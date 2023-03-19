import pdb
import torch
import numpy as np
import faiss
from loguru import logger
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import T5EncoderModel, ByT5Tokenizer
from transformers import get_cosine_schedule_with_warmup
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy


torch.set_float32_matmul_precision("medium")


def _encode(encoder, input_ids, attention_mask):
    hidden_states = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).last_hidden_state
    features = hidden_states.mean(dim=1)
    return F.normalize(features, dim=1)


class PremiseRetriever(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        dual_encoder: bool,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_seq_len = max_seq_len

        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
        self.query_encoder = T5EncoderModel.from_pretrained(model_name)
        if dual_encoder:
            self.doc_encoder = T5EncoderModel.from_pretrained(model_name)
        else:
            self.doc_encoder = self.query_encoder
        # TODO: Try adding a linear project layer for dimensionality reduction.
        # TODO: Retrieval and generation can share a model.
        # TODO: Do we need re-ranking?

    def forward(
        self,
        query_ids,
        query_mask,
        positive_doc_ids,
        positive_doc_mask,
        negative_docs_ids,
        negative_docs_mask,
        label,
    ):
        query_emb = _encode(self.query_encoder, query_ids, query_mask)
        doc_emb = _encode(self.doc_encoder, positive_doc_ids, positive_doc_mask)

        assert len(negative_docs_ids) == len(negative_docs_mask)
        negative_embs = [
            _encode(self.doc_encoder, ids, mask)
            for ids, mask in zip(negative_docs_ids, negative_docs_mask)
        ]

        all_doc_embs = torch.cat([doc_emb, *negative_embs], dim=0)

        # query_emb = F.normalize(query_emb, dim=1)
        # doc_emb = F.normalize(doc_emb, dim=1)
        # batch_size = query_emb.size(0)

        similarity = torch.mm(query_emb, all_doc_embs.t())
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

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        loss = self(
            batch["query_ids"],
            batch["query_mask"],
            batch["positive_doc_ids"],
            batch["positive_doc_mask"],
            batch["negative_docs_ids"],
            batch["negative_docs_mask"],
            batch["label"],
        )
        self.log("loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch))
        return loss

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

    def on_validation_start(self) -> None:
        self.reindex_corpus()

    def on_test_start(self) -> None:
        self.reindex_corpus()

    def reindex_corpus(self) -> None:
        logger.info("Re-indexing the retrieval corpus")

        def embed(examples):
            positive_doc = self.tokenizer(
                examples["doc"],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                doc_emb = (
                    _encode(
                        self.doc_encoder,
                        positive_doc.input_ids,
                        positive_doc.attention_mask,
                    )
                    .cpu()
                    .to(dtype=torch.float32)
                    .numpy()
                )
            return {"doc_emb": doc_emb}

        self.indexed_corpus = self.trainer.datamodule.corpus.all_premises.map(
            embed, batched=True, batch_size=self.trainer.datamodule.batch_size
        ).add_faiss_index(column="doc_emb", metric_type=faiss.METRIC_INNER_PRODUCT)

    def validation_step(self, batch, batch_idx: int) -> None:
        self.val_test_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx: int) -> None:
        self.val_test_step("test", batch, batch_idx)

    def val_test_step(self, split: str, batch, batch_idx: int) -> None:
        # Perform retrieval.
        query_emb = _encode(self.query_encoder, batch["query_ids"], batch["query_mask"])
        query_emb = query_emb.cpu().to(dtype=torch.float32).numpy()

        # TODO: Try retrieving from only feasible premises.
        _, results = self.indexed_corpus.get_nearest_examples_batch(
            "doc_emb", query_emb, k=10
        )

        retrieved_docs = [r["doc"] for r in results]
        recall = [[] for j in range(10)]
        tb = self.logger.experiment
        batch_size = len(batch)

        for i, (doc_gt, docs) in enumerate(zip(batch["positive_doc"], retrieved_docs)):
            n = batch_size * batch_idx + i
            if i == 0:
                tb.add_text(f"doc_gt_{split}", doc_gt, n)

            k = len(docs)
            for j in range(1, k+1):
                if i == 0:
                    tb.add_text(f"docs_{j}_{split}", docs[j - 1], n)
                # TODO: Only check the path and the name.
                if doc_gt in docs[:j]:
                    recall[j - 1].append(1.0)
                else:
                    recall[j - 1].append(0.0)

        recall = [100 * np.mean(_) for _ in recall]
        for j in range(1, 11):
            self.log(
                f"Recall@{j}_{split}",
                recall[j - 1],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self):
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
