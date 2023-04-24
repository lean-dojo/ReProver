import pdb
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Tuple
from transformers import T5EncoderModel, AutoTokenizer
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy


from common import (
    Premise,
    Context,
    Corpus,
    get_optimizers,
    load_checkpoint,
    zip_strict,
    MARK_START_SYMBOL,
)


torch.set_float32_matmul_precision("medium")


class PremiseRetriever(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_retrieved: int,
        max_seq_len: int,
        corpus_path: Union[str, Path, None] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_retrieved = num_retrieved
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        if corpus_path is not None:
            self.corpus = Corpus(corpus_path)
            self._init_corpus_embeddings()
        else:
            self.corpus = None
            self.embeddings_staled = True

        self.validation_step_outputs = []

    @classmethod
    def load(
        cls, ckpt_path: Union[str, Path], device, freeze: bool
    ) -> "PremiseRetriever":
        return load_checkpoint(cls, Path(ckpt_path), device, freeze)

    @property
    def embedding_size(self) -> int:
        return self.encoder.config.hidden_size

    def _cpu_checkpointing_enabled(self) -> bool:
        try:
            trainer = self.trainer
            return (
                trainer.strategy is not None
                and isinstance(trainer.strategy, DeepSpeedStrategy)
                and trainer.strategy.config["activation_checkpointing"][
                    "cpu_checkpointing"
                ]
            )
        except RuntimeError:
            return False

    def _encode(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a tokenized sequence represented by ``input_ids`` and ``attention_mask``
        into a feature vector using ``encoder``.
        """
        if self._cpu_checkpointing_enabled():
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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:  # type: ignore
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

    def _init_corpus_embeddings(self) -> None:
        corpus_embeddings = torch.zeros(
            len(self.corpus.all_premises),
            self.embedding_size,
            dtype=self.encoder.dtype,
            device=self.device,
        )
        self.register_buffer("corpus_embeddings", corpus_embeddings)
        self.embeddings_staled = True

    def on_fit_start(self) -> None:
        self.corpus = self.trainer.datamodule.corpus
        self._init_corpus_embeddings()

        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        assert not self.embeddings_staled
        checkpoint["corpus"] = self.corpus

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.corpus = checkpoint["corpus"]

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            if "corpus_embeddings" in checkpoint["state_dict"]:
                assert state_dict["corpus_embeddings"].size() == (
                    len(self.corpus),
                    self.embedding_size,
                )
                self.register_buffer(
                    "corpus_embeddings", state_dict["corpus_embeddings"]
                )
                self.embeddings_staled = False

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.embeddings_staled = True

    def on_validation_start(self) -> None:
        if self.embeddings_staled:
            self.reindex_corpus(8 * self.trainer.datamodule.batch_size)

    def on_validation_end(self) -> None:
        outputs = []

        for _ in self.validation_step_outputs:
            for context, pos_premise, retrieved_premises, scores in zip_strict(*_):
                outputs.append(
                    {
                        "context": context,
                        "all_pos_premises": pos_premise,
                        "retrieved_premises": retrieved_premises,
                        "scores": scores,
                    }
                )

        if self.trainer.log_dir is not None:
            path = (
                Path(self.trainer.log_dir)
                / f"epoch{self.current_epoch}_validation_outputs.pickle"
            )
            with path.open("wb") as oup:
                pickle.dump(outputs, oup)
            logger.info(f"Validation outputs saved to {path}")

        self.validation_step_outputs.clear()

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        logger.info("Re-indexing the retrieval corpus")

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

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate Recall@K evaluation metrics."""
        # Retrieval.
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings, batch["context"], context_emb, self.num_retrieved
        )

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []
        tb = self.logger.experiment

        for i, (all_pos_premises, premises) in enumerate(
            zip(batch["all_pos_premises"], retrieved_premises)
        ):
            # Only log the first example in the batch.
            if i == 0:
                msg_gt = "\n\n".join(
                    [p.serialize() for p in all_pos_premises]
                )
                msg_retrieved = "\n\n".join(
                    [f"{j}. {p.serialize()}" for j, p in enumerate(premises)]
                )
                TP = len(set(premises).intersection(all_pos_premises))
                r = float(TP) / len(all_pos_premises)
                msg = f"Recall@{self.num_retrieved}: {r}\n\nGround truth:\n\n`{msg_gt}`\n\n Retrieved:\n\n```\n{msg_retrieved}\n```"
                tb.add_text(f"premises_val", msg, self.global_step)

            all_pos_premises = set(all_pos_premises)
            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premises.intersection(premises[: (j + 1)]))
                recall[j].append(float(TP) / len(all_pos_premises))
                if premises[j] in all_pos_premises and not first_match_found:
                    MRR.append(1.0 / j)
                    first_match_found = True
            if not first_match_found:
                MRR.append(0.0)

        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            self.log(
                f"Recall@{j+1}_val (%)",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch),
            )

        self.log(
            "MRR", np.mean(MRR), on_epoch=True, sync_dist=True, batch_size=len(batch)
        )

        self.validation_step_outputs.append(
            (batch["context"], batch["all_pos_premises"], retrieved_premises, scores)
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    def retrieve(
        self,
        state: str,
        file_name,
        theorem_full_name,
        theorem_pos,
        tactic_prefix: str,
        k: int,
    ) -> Tuple[List[Premise], List[float]]:
        # """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        raise NotImplementedError
        assert tactic_prefix.endswith(MARK_START_SYMBOL)
        ctx = Context(file_name, theorem_full_name, theorem_pos, tactic_prefix, state)
        ctx_tokens = self.tokenizer(
            ctx.serialize(),
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        context_emb = self._encode(
            ctx_tokens.input_ids.to(self.device),
            ctx_tokens.attention_mask.to(self.device),
        )
        assert not self.embeddings_staled
        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings, [ctx], context_emb, k
        )
        return retrieved_premises[0], scores[0]
