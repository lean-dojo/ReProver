"""Ligihtning module for the premise retriever."""
import os
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm
from lean_dojo import Pos
from loguru import logger
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Union
from transformers import T5EncoderModel, AutoTokenizer

from common import (
    Premise,
    Context,
    Corpus,
    get_optimizers,
    load_checkpoint,
    zip_strict,
    cpu_checkpointing_enabled,
)


torch.set_float32_matmul_precision("medium")


class PremiseRetriever(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        max_seq_len: int,
        num_retrieved: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_retrieved = num_retrieved
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.embeddings_staled = True

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "PremiseRetriever":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def load_corpus(self, path_or_corpus: Union[str, Corpus]) -> None:
        """Associate the retriever with a corpus."""
        if isinstance(path_or_corpus, Corpus):
            self.corpus = path_or_corpus
            self.corpus_embeddings = None
            self.embeddings_staled = True
            return

        path = path_or_corpus
        if path.endswith(".jsonl"):  # A raw corpus without embeddings.
            self.corpus = Corpus(path)
            self.corpus_embeddings = None
            self.embeddings_staled = True
        else:  # A corpus with pre-computed embeddings.
            indexed_corpus = pickle.load(open(path, "rb"))
            self.corpus = indexed_corpus.corpus
            self.corpus_embeddings = indexed_corpus.embeddings
            self.embeddings_staled = False

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.encoder.config.hidden_size

    def _encode(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a premise or a context into a feature vector."""
        if cpu_checkpointing_enabled(self):
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.encoder, input_ids, attention_mask, use_reentrant=False
            )[0]
        else:
            hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

        # Masked average.
        lens = attention_mask.sum(dim=1)
        features = (hidden_states * attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        # Normalize the feature vector to have unit norm.
        return F.normalize(features, dim=1)

    def forward(
        self,
        context_ids: torch.LongTensor,
        context_mask: torch.LongTensor,
        pos_premise_ids: torch.LongTensor,
        pos_premise_mask: torch.LongTensor,
        neg_premises_ids: torch.LongTensor,
        neg_premises_mask: torch.LongTensor,
        label: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the contrastive loss for premise retrieval."""
        # Encode the query and positive/negative documents.
        context_emb = self._encode(context_ids, context_mask)
        pos_premise_emb = self._encode(pos_premise_ids, pos_premise_mask)
        neg_premise_embs = [
            self._encode(ids, mask)
            for ids, mask in zip_strict(neg_premises_ids, neg_premises_mask)
        ]
        all_premise_embs = torch.cat([pos_premise_emb, *neg_premise_embs], dim=0)

        # Cosine similarities for unit-norm vectors are just inner products.
        similarity = torch.mm(context_emb, all_premise_embs.t())
        assert -1 <= similarity.min() <= similarity.max() <= 1
        loss = F.mse_loss(similarity, label)
        return loss

    ############
    # Training #
    ############

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            logger.info(f"Logging to {self.trainer.log_dir}")

        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True

    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        loss = self(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )
        self.log(
            "loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch)
        )
        return loss

    def on_train_batch_end(self, outputs, batch, _) -> None:
        """Mark the embeddings as staled after a training batch."""
        self.embeddings_staled = True

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    ##############
    # Validation #
    ##############

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        if not self.embeddings_staled:
            return
        logger.info("Re-indexing the retrieval corpus")

        self.corpus_embeddings = torch.zeros(
            len(self.corpus.all_premises),
            self.embedding_size,
            dtype=self.encoder.dtype,
            device=self.device,
        )

        for i in tqdm(range(0, len(self.corpus), batch_size)):
            batch_premises = self.corpus.all_premises[i : i + batch_size]
            tokenized_premises = self.tokenizer(
                [p.serialize() for p in batch_premises],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.corpus_embeddings[i : i + batch_size] = self._encode(
                tokenized_premises.input_ids, tokenized_premises.attention_mask
            )

        self.embeddings_staled = False

    def on_validation_start(self) -> None:
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate metrics such as Recall@K and MRR."""
        # Retrieval.
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, _ = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []
        num_with_premises = 0
        tb = self.logger.experiment

        for i, (all_pos_premises, premises) in enumerate(
            zip_strict(batch["all_pos_premises"], retrieved_premises)
        ):
            # Only log the first example in the batch.
            if i == 0:
                msg_gt = "\n\n".join([p.serialize() for p in all_pos_premises])
                msg_retrieved = "\n\n".join(
                    [f"{j}. {p.serialize()}" for j, p in enumerate(premises)]
                )
                TP = len(set(premises).intersection(all_pos_premises))
                if len(all_pos_premises) == 0:
                    r = math.nan
                else:
                    r = float(TP) / len(all_pos_premises)
                msg = f"Recall@{self.num_retrieved}: {r}\n\nGround truth:\n\n```\n{msg_gt}\n```\n\nRetrieved:\n\n```\n{msg_retrieved}\n```"
                tb.add_text(f"premises_val", msg, self.global_step)

            all_pos_premises = set(all_pos_premises)
            if len(all_pos_premises) == 0:
                continue
            else:
                num_with_premises += 1
            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premises.intersection(premises[: (j + 1)]))
                recall[j].append(float(TP) / len(all_pos_premises))
                if premises[j] in all_pos_premises and not first_match_found:
                    MRR.append(1.0 / (j + 1))
                    first_match_found = True
            if not first_match_found:
                MRR.append(0.0)

        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            self.log(
                f"Recall@{j+1}_val",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=num_with_premises,
            )

        self.log(
            "MRR",
            np.mean(MRR),
            on_epoch=True,
            sync_dist=True,
            batch_size=num_with_premises,
        )

    ##############
    # Prediction #
    ##############

    def on_predict_start(self) -> None:
        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)
        self.predict_step_outputs = []

    def predict_step(self, batch: Dict[str, Any], _):
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )

        for (
            url,
            commit,
            file_path,
            full_name,
            start,
            tactic_idx,
            ctx,
            pos_premises,
            premises,
            s,
        ) in zip_strict(
            batch["url"],
            batch["commit"],
            batch["file_path"],
            batch["full_name"],
            batch["start"],
            batch["tactic_idx"],
            batch["context"],
            batch["all_pos_premises"],
            retrieved_premises,
            scores,
        ):
            self.predict_step_outputs.append(
                {
                    "url": url,
                    "commit": commit,
                    "file_path": file_path,
                    "full_name": full_name,
                    "start": start,
                    "tactic_idx": tactic_idx,
                    "context": ctx,
                    "all_pos_premises": pos_premises,
                    "retrieved_premises": premises,
                    "scores": s,
                }
            )

    def on_predict_epoch_end(self) -> None:
        if self.trainer.log_dir is not None:
            path = os.path.join(self.trainer.log_dir, "predictions.pickle")
            with open(path, "wb") as oup:
                pickle.dump(self.predict_step_outputs, oup)
            logger.info(f"Retrieval predictions saved to {path}")

        self.predict_step_outputs.clear()

    def retrieve(
        self,
        state: List[str],
        file_name: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        k: int,
    ) -> Tuple[List[Premise], List[float]]:
        """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        self.reindex_corpus(batch_size=32)

        ctx = [
            Context(*_)
            for _ in zip_strict(file_name, theorem_full_name, theorem_pos, state)
        ]
        ctx_tokens = self.tokenizer(
            [_.serialize() for _ in ctx],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        context_emb = self._encode(
            ctx_tokens.input_ids.to(self.device),
            ctx_tokens.attention_mask.to(self.device),
        )

        if self.corpus_embeddings.device != context_emb.device:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.device)
        if self.corpus_embeddings.dtype != context_emb.dtype:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.dtype)

        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            ctx,
            context_emb,
            k,
        )
        return retrieved_premises, scores
