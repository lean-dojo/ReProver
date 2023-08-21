"""Lightning module for the tactic generator."""
import re
import time
import torch
import openai
import pickle
from lean_dojo import Pos
from loguru import logger
import pytorch_lightning as pl
from torchmetrics import Metric
from lean_dojo.utils import execute
from abc import ABC, abstractmethod
from subprocess import CalledProcessError
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    zip_strict,
    remove_marks,
    IndexedCorpus,
    get_optimizers,
    load_checkpoint,
    format_augmented_state,
)
from retrieval.model import PremiseRetriever


torch.set_float32_matmul_precision("medium")


class TopkAccuracy(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch_preds: List[List[str]], batch_gt: List[str]):
        assert len(batch_preds) == len(batch_gt)
        for preds, gt in zip(batch_preds, batch_gt):
            # This still doesn't account for short names vs. full names.
            gt = remove_marks(gt)
            preds = [remove_marks(p) for p in preds]
            self.correct += gt in preds[: self.k]
        self.total += len(batch_gt)

    def compute(self) -> float:
        return self.correct.float() / self.total


class TacticGenerator(ABC):
    """A tactic generator takes a state and generates multiple tactic candidates."""

    @abstractmethod
    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError


class RetrievalAugmentedGenerator(TacticGenerator, pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        eval_num_retrieved: int,
        eval_num_cpus: int,
        eval_num_theorems: int,
        max_seq_len: int,
        length_penalty: float = 0.0,
        ret_ckpt_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.eval_num_retrieved = eval_num_retrieved
        self.eval_num_cpus = eval_num_cpus
        self.eval_num_theorems = eval_num_theorems
        self.max_seq_len = max_seq_len

        if ret_ckpt_path is None:
            logger.info("Without retrieval")
            self.retriever = None
        else:
            logger.info(f"Loading the retriever from {ret_ckpt_path}")
            self.retriever = PremiseRetriever.load(
                ret_ckpt_path, self.device, freeze=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = AutoModelForCausalLM.from_pretrained(model_name)

        self.topk_accuracies = dict()
        for k in range(1, num_beams + 1):
            acc = TopkAccuracy(k)
            self.topk_accuracies[k] = acc
            self.add_module(f"top{k}_acc_val", acc)

    @classmethod
    def load(
        cls, ckpt_path: str, device, freeze: bool
    ) -> "RetrievalAugmentedGenerator":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        input_labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=input_labels,
        )[0]

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["input_ids"],
            batch["input_mask"],
            batch["input_labels"],
        )
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        self._log_io_texts("train", batch["state_ids"], batch["tactic_ids"])
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    def _log_io_texts(
        self,
        split: str,
        state_ids: torch.LongTensor,
        tactic_ids: torch.LongTensor,
    ) -> None:
        tb = self.logger.experiment
        inp = self.tokenizer.decode(state_ids[0], skip_special_tokens=True)
        oup_ids = torch.where(
            tactic_ids[0] == -100, self.tokenizer.pad_token_id, tactic_ids[0]
        )
        oup = self.tokenizer.decode(oup_ids, skip_special_tokens=True)
        tb.add_text(f"{split}_state", f"```\n{inp}\n```", self.global_step)
        tb.add_text(f"{split}_tactic", f"`{oup}`", self.global_step)

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

        if self.retriever is not None:
            self.retriever.load_corpus(self.trainer.datamodule.corpus)

    ##############
    # Validation #
    ##############

    def validation_step(self, batch: Dict[str, Any], _) -> None:
        state_ids = batch["state_ids"]
        state_mask = batch["state_mask"]
        tactic_ids = batch["tactic_ids"]
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        input_labels = batch["input_labels"]

        loss = self(input_ids, input_mask, input_labels)
        self.log(f"loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
        self._log_io_texts("val", state_ids, tactic_ids)

        # Generate topk tactic candidates via Beam Search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=self.num_beams,
            do_sample=False,
            num_return_sequences=self.num_beams,
            early_stopping=False,
        )
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        batch_size = state_ids.size(0)
        assert len(output_text) == batch_size * self.num_beams
        tactics_pred = [
            output_text[i * self.num_beams : (i + 1) * self.num_beams]
            for i in range(batch_size)
        ]

        tb = self.logger.experiment
        msg = "\n".join(tactics_pred[0])
        tb.add_text(f"preds_val", f"```\n{msg}\n```", self.global_step)

        # Log the topk accuracies.
        for k in range(1, self.num_beams + 1):
            topk_acc = self.topk_accuracies[k]
            topk_acc(tactics_pred, batch["tactic"])
            self.log(f"top{k}_acc_val", topk_acc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last.ckpt"
        self.trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

        data_path = self.trainer.datamodule.data_path
        if self.retriever is None:
            cmd = f"python prover/evaluate.py --data-path {data_path} --num-cpus {self.eval_num_cpus} --num-theorems {self.eval_num_theorems} --ckpt_path {ckpt_path}"
        else:
            self.retriever.reindex_corpus(self.trainer.datamodule.eval_batch_size)
            corpus_path = f"{self.trainer.log_dir}/checkpoints/indexed_corpus.pickle"
            pickle.dump(
                IndexedCorpus(
                    self.retriever.corpus, self.retriever.corpus_embeddings.cpu()
                ),
                open(corpus_path, "wb"),
            )
            cmd = f"python prover/evaluate.py --data-path {data_path} --num-cpus {self.eval_num_cpus} --num-theorems {self.eval_num_theorems} --ckpt_path {ckpt_path} --indexed-corpus-path {corpus_path}"

        logger.info(cmd)

        wait_time = 3600
        while True:
            try:
                _, err = execute(cmd, capture_output=True)
                break
            except CalledProcessError as ex:
                logger.error(ex)
                logger.error(
                    f"Failed to evaluate. Retrying in {wait_time / 3600} hour..."
                )
                time.sleep(wait_time)
                wait_time *= 2

        m = re.search(r"Pass@1: (\S+)", err)
        assert m is not None, err
        acc = float(m.group(1))
        self.log("Pass@1_val", acc, on_step=False, on_epoch=True)
        logger.info(f"Pass@1: {acc}")

    ##############
    # Prediction #
    ##############

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return self.batch_generate(
            [state], [file_path], [theorem_full_name], [theorem_pos], num_samples
        )[0]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        logger.debug(state)
        if self.retriever is not None:
            retrieved_premises, _ = self.retriever.retrieve(
                state,
                file_path,
                theorem_full_name,
                theorem_pos,
                self.eval_num_retrieved,
            )
            state = [
                format_augmented_state(s, premises, self.max_seq_len, p_drop=0.0) +  + "\u0002\u0002\u0002"
                for s, premises in zip_strict(state, retrieved_premises)
            ]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=num_samples,
            length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        raw_scores = output.sequences_scores.tolist()
        tactics_with_scores = []

        for i in range(len(state)):
            output_text = []
            output_score = []

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = remove_marks(raw_output_text[j])
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores
