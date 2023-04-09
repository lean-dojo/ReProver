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
    get_optimizers,
    load_checkpoint,
    to_path,
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_retrieved = num_retrieved
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.corpus = None
        self.stale_corpus_embeddings = True
        self.validation_step_outputs = []
        # TODO: Do we need re-ranking?

    @classmethod
    def load(
        cls, ckpt_path: Union[str, Path], device, freeze: bool
    ) -> "PremiseRetriever":
        return load_checkpoint(cls, to_path(ckpt_path), device, freeze)

    def _cpu_checkpointing_enabled(self) -> bool:
        try:
            trainer = self.trainer
            return (
                trainer.strategy is not None
                and isinstance(trainer.strategy, DeepSpeedStrategy)
                and "cpu_checkpointing" in trainer.strategy.config["zero_optimization"]
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

    def on_fit_start(self) -> None:
        self.corpus = self.trainer.datamodule.corpus
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.stale_corpus_embeddings = True

    def on_validation_start(self) -> None:
        if self.stale_corpus_embeddings:
            self.reindex_corpus(16 * self.trainer.datamodule.batch_size)
            self.stale_corpus_embeddings = False
        self.corpus.to(self.device)

    def on_validation_end(self) -> None:
        self.corpus.cpu()
        outputs = []

        for _ in self.validation_step_outputs:
            for context, pos_premise, retrieved_premises, scores in zip_strict(*_):
                outputs.append(
                    {
                        "context": context,
                        "pos_premise": pos_premise,
                        "retrieved_premises": retrieved_premises,
                        "scores": scores,
                    }
                )

        path = (
            Path(self.trainer.log_dir)
            / f"epoch{self.current_epoch}_validation_outputs.pickle"
        )
        with path.open("wb") as oup:
            pickle.dump(outputs, oup)
        logger.info(f"Validation outputs saved to {path}")

        self.validation_step_outputs.clear()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        assert self.corpus.has_embeddings and not self.stale_corpus_embeddings
        checkpoint["corpus"] = self.corpus

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "corpus" in checkpoint:
            self.corpus = checkpoint["corpus"]
            self.stale_corpus_embeddings = False
        else:
            assert self.stale_corpus_embeddings

    def reindex_corpus(self, batch_size: int) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        logger.info("Re-indexing the retrieval corpus")

        @torch.no_grad()
        def corpus_encoder(all_premises: List[Premise]) -> torch.Tensor:
            premise_embeddings = []

            for i in tqdm(range(0, len(all_premises), batch_size)):
                batch_premises = all_premises[i : i + batch_size]
                tokenized_premises = self.tokenizer(
                    [p.serialize() for p in batch_premises],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                emb = self._encode(
                    tokenized_premises.input_ids, tokenized_premises.attention_mask
                )
                premise_embeddings.append(emb)

            return torch.cat(premise_embeddings).float().cpu()

        self.corpus.update_embeddings(corpus_encoder)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate Recall@K evaluation metrics."""
        # Retrieval.
        context_emb = self._encode(batch["context_ids"], batch["context_mask"]).float()
        retrieved_premises, scores = self.corpus.get_nearest_premises(
            batch["context"], context_emb, self.num_retrieved
        )

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        tb = self.logger.experiment

        for i, (premise_gt, premises) in enumerate(
            zip(batch["pos_premise"], retrieved_premises)
        ):
            # Only log the first example in the batch.
            if i == 0:
                msg = "\n\n".join(
                    [f"{j}. {p.serialize()}" for j, p in enumerate(premises)]
                )
                msg = f"{premise_gt in premises}\nGround truth:\n `{premise_gt.serialize()}`\n Retrieved:\n```\n{msg}\n```"
                tb.add_text(f"premises_val", msg, self.global_step)

            for j in range(self.num_retrieved):
                if premise_gt in premises[: (j + 1)]:
                    recall[j].append(1.0)
                else:
                    recall[j].append(0.0)

        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            self.log(
                f"Recall@{j+1}_val",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch),
            )

        self.validation_step_outputs.append(
            (batch["context"], batch["pos_premise"], retrieved_premises, scores)
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
        assert tactic_prefix.endswith(MARK_START_SYMBOL)
        ctx = Context(file_name, theorem_full_name, theorem_pos, state, tactic_prefix)
        ctx_tokens = self.tokenizer(
            ctx.serialize(),
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        context_emb = self._encode(ctx_tokens.input_ids, ctx_tokens.attention_mask)
        retrieved_premises, scores = self.corpus.get_nearest_premises(
            [ctx], context_emb, k
        )
        return retrieved_premises[0], scores[0]
