import pdb
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import List, Dict, Any
from retrieval.datamodule import Premise
from transformers import T5EncoderModel, ByT5Tokenizer
from transformers import get_cosine_schedule_with_warmup
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy


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

        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        # TODO: Try adding a linear project layer for dimensionality reduction.
        # TODO: Retrieval and generation can share the backbone.
        # TODO: Do we need re-ranking?

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode a tokenized sequence represented by ``input_ids`` and ``attention_mask``
        into a feature vector using ``encoder``.
        """
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        # Masked average.
        lens = attention_mask.sum(dim=1)
        features = (hidden_states * attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
        # Normalize the feature vector to have unit norm.
        return F.normalize(features, dim=1)

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        pos_premise_ids: torch.Tensor,
        pos_premise_mask: torch.Tensor,
        neg_premises_ids: torch.Tensor,
        neg_premises_mask: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        # Encode the query and positive/negative documents.
        context_emb = self._encode(context_ids, context_mask)
        pos_premise_emb = self._encode(pos_premise_ids, pos_premise_mask)
        assert len(neg_premises_ids) == len(neg_premises_mask)
        neg_premise_embs = [
            self._encode(ids, mask)
            for ids, mask in zip(neg_premises_ids, neg_premises_mask)
        ]
        all_premise_embs = torch.cat([pos_premise_emb, *neg_premise_embs], dim=0)

        # Cosine similarities for unit-norm vectors are just inner products.
        similarity = torch.mm(context_emb, all_premise_embs.t())
        assert -1 <= similarity.min() <= similarity.max() <= 1

        # Cosine similarity loss.
        # mask = torch.zeros_like(similarity)
        # mask.fill_diagonal_(1.0)
        # loss = (torch.eye(batch_size, device=self.device) - mask * similarity + (1.0 - mask) * torch.maximum(torch.zeros_like(similarity), similarity)).mean()

        # target = torch.zeros_like(similarity)
        # target.fill_diagonal_(1.0)
        # loss = F.binary_cross_entropy_with_logits(similarity, target)
        # loss = -F.log_softmax(similarity, dim=1).diag().mean()
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

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

    def on_validation_start(self) -> None:
        self.reindex_corpus()
        self.trainer.datamodule.corpus.to(self.device)

    def on_test_start(self) -> None:
        self.reindex_corpus()
        self.trainer.datamodule.corpus.to(self.device)

    def on_validation_end(self) -> None:
        self.trainer.datamodule.corpus.cpu()

    def on_test_end(self) -> None:
        self.trainer.datamodule.corpus.cpu()

    def reindex_corpus(self) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        logger.info("Re-indexing the retrieval corpus")

        def corpus_encoder(all_premises: List[Premise]) -> torch.Tensor:
            # OK to use larger batch size since it is less expensive than training the model.
            batch_size = 8 * self.trainer.datamodule.batch_size
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

            return torch.cat(premise_embeddings).cpu()

        with torch.no_grad():
            self.trainer.datamodule.corpus.update_embeddings(corpus_encoder)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self.val_test_step("val", batch, batch_idx)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self.val_test_step("test", batch, batch_idx)

    def val_test_step(self, split: str, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate Recall@K evaluation metrics."""
        # Retrieval.
        corpus = self.trainer.datamodule.corpus
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        retrieved_premises, _ = corpus.get_nearest_premises(
            batch["path"], batch["pos"], context_emb, self.num_retrieved
        )

        # Evaluation & logging.
        batch_size = len(batch)
        recall = [[] for _ in range(self.num_retrieved)]
        tb = self.logger.experiment

        for i, (premise_gt, premises) in enumerate(
            zip(batch["pos_premise"], retrieved_premises)
        ):
            n = batch_size * batch_idx + i
            if i == 0:
                tb.add_text(f"premise_gt_{split}", premise_gt, n)

            for j in range(self.num_retrieved):
                if i == 0:
                    tb.add_text(f"premises_{j + 1}_{split}", premises[j], n)
                # TODO: Only check the path and the name.
                if premise_gt in premises[: (j + 1)]:
                    recall[j].append(1.0)
                else:
                    recall[j].append(0.0)

        recall = [100 * np.mean(_) for _ in recall]
        for j in range(self.num_retrieved):
            self.log(
                f"Recall@{j+1}_{split}",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure an AdamW optimizer with cosine warmup learning rate schedule."""
        parameters = self.parameters()
        strategy = self.trainer.strategy

        if isinstance(strategy, DeepSpeedStrategy):
            if "offload_optimizer" in strategy.config["zero_optimization"]:
                logger.info("Optimizing with DeepSpeedCPUAdam")
                optimizer = DeepSpeedCPUAdam(parameters, lr=self.lr, adamw_mode=True)
            else:
                logger.info("Optimizing with FusedAdam")
                optimizer = FusedAdam(parameters, lr=self.lr, adam_w_mode=True)
        else:
            logger.info("Optimizing with AdamW")
            optimizer = torch.optim.AdamW(parameters, lr=self.lr)

        if self.trainer.max_steps != -1:
            max_steps = self.trainer.max_steps
        else:
            assert self.trainer.max_epochs is not None
            max_steps = (
                self.trainer.max_epochs
                * len(self.trainer.datamodule.train_dataloader())
                // self.trainer.accumulate_grad_batches
            )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
