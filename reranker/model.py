import os
import pdb
import torch
import pickle
import numpy as np
import torch.nn as nn
from loguru import logger
from typing import Dict, Any
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from transformers import T5EncoderModel, AutoModel, AutoTokenizer
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

from common import get_optimizers, zip_strict, cpu_checkpointing_enabled


torch.set_float32_matmul_precision("medium")


class PremiseReranker(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        max_seq_len: int,
        num_retrieved: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_seq_len = max_seq_len
        self.num_retrieved = num_retrieved
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        # self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.classifier = nn.Linear(h, 1)

        self.metrics = {
            "train": {
                "accuracy": BinaryAccuracy(),
                "f1": BinaryF1Score(),
            },
            "val": {
                "accuracy": BinaryAccuracy(),
                "f1": BinaryF1Score(),
            },
        }
        for split, metrics in self.metrics.items():
            for name, m in metrics.items():
                self.add_module(f"{name}_{split}", m)

        self.predict_step_outputs = []

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

    def forward(
        self,
        seq_ids: torch.LongTensor,
        seq_mask: torch.LongTensor,
    ) -> torch.FloatTensor:
        if cpu_checkpointing_enabled(self):
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.encoder, seq_ids, seq_mask, use_reentrant=False
            )[0]
        else:
            hidden_states = self.encoder(seq_ids, seq_mask).last_hidden_state

        # Masked average.
        lens = seq_mask.sum(dim=1).unsqueeze(1)
        features = (hidden_states * seq_mask.unsqueeze(2)).sum(dim=1) / lens

        logits = self.classifier(features).squeeze(dim=1)
        return logits

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:  # type: ignore
        logits = self(batch["seq_ids"], batch["seq_mask"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"])

        self.log(
            "loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch)
        )
        self._log_metrics("train", logits, batch["label"])
        print(loss.item())
        return loss

    def on_train_epoch_end(self) -> None:
        self.trainer.datamodule.ds_train.reload_data()

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

    def _log_metrics(
        self, split: str, logits: torch.Tensor, label: torch.Tensor
    ) -> None:
        for name, metric in self.metrics[split].items():
            metric(logits.to(torch.float32), label.to(torch.float32))
            self.log(f"{name}_{split}", metric, on_step=False, on_epoch=True)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate Recall@K evaluation metrics."""
        assert len(batch["seqs"][0]) == self.num_retrieved
        all_scores = [
            self(batch[f"seq_{i}_ids"], batch[f"seq_{i}_mask"])
            for i in range(self.num_retrieved)
        ]

        reranked_premises = []
        batch_size = len(all_scores[0])

        for j in range(batch_size):
            scores = [all_scores[i][j].item() for i in range(self.num_retrieved)]
            premises = batch["retrieved_premises"][j]
            reranked_premises.append([premises[k] for k in np.argsort(scores)[::-1]])

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []
        tb = self.logger.experiment

        for i, (all_pos_premises, premises) in enumerate(
            zip_strict(batch["all_pos_premises"], reranked_premises)
        ):
            # Only log the first example in the batch.
            if i == 0:
                msg_gt = "\n\n".join([p.serialize() for p in all_pos_premises])
                msg_reranked = "\n\n".join(
                    [f"{j}. {p.serialize()}" for j, p in enumerate(premises)]
                )
                msg = f"Ground truth:\n\n`{msg_gt}`\n\n Reranked:\n\n```\n{msg_reranked}\n```"
                tb.add_text(f"premises_val", msg, self.global_step)

            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premises.intersection(premises[: (j + 1)]))
                if len(all_pos_premises) == 0:
                    continue
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
                batch_size=len(batch),
            )

        self.log(
            "MRR", np.mean(MRR), on_epoch=True, sync_dist=True, batch_size=len(batch)
        )

    def predict_step(self, batch: Dict[str, Any], batch_idx: int):
        assert len(batch["seqs"][0]) == self.num_retrieved
        all_scores = [
            self(batch[f"seq_{i}_ids"], batch[f"seq_{i}_mask"])
            for i in range(self.num_retrieved)
        ]

        reranked_premises = []
        batch_size = len(all_scores[0])

        for j in range(batch_size):
            scores = [all_scores[i][j].item() for i in range(self.num_retrieved)]
            premises = batch["retrieved_premises"][j]
            reranked_premises.append([premises[k] for k in np.argsort(scores)[::-1]])

        pdb.set_trace()
        pred = (batch["context"], batch["all_pos_premises"], reranked_premises, scores)
        self.predict_step_outputs.append(pred)
        return pred

    def on_predict_epoch_end(self) -> None:
        outputs = self._unpack_outputs(self.predict_step_outputs)

        if self.trainer.log_dir is not None:
            path = os.path.join(self.trainer.log_dir, "predictions.pickle")
            with open(path, "wb") as oup:
                pickle.dump(outputs, oup)
            logger.info(f"Predictions saved to {path}")

        self.predict_step_outputs.clear()

    def on_fit_end(self) -> None:
        logger.info("Using the trained model to make predictions.")
        self.trainer.predict(self, self.trainer.datamodule.predict_dataloader())

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )
