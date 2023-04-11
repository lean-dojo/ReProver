import pdb
import math
import torch
from copy import deepcopy
from pathlib import Path
from lean_dojo import Pos
from loguru import logger
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    ByT5Tokenizer,
    BeamSearchScorer,
    StoppingCriteriaList,
    MaxLengthCriteria,
    LogitsProcessor,
    LogitsProcessorList,
)
import pytorch_lightning as pl
from torchmetrics import Metric
from collections import defaultdict
from abc import ABC, abstractmethod
from retrieval.model import PremiseRetriever
from typing import List, Dict, Any, Optional, Tuple, Union


from common import (
    get_optimizers,
    remove_marks,
    is_well_formed,
    find_open_mark,
    load_checkpoint,
    to_path,
    zip_strict,
    MARK_START_SYMBOL,
    MARK_END_SYMBOL,
)


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
    @abstractmethod
    def generate(
        self,
        state: str,
        file_path: Path,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError


class TransformerTacticGenerator(TacticGenerator, pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        topk: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.topk = topk
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seq2seq = T5ForConditionalGeneration.from_pretrained(model_name)

        self.topk_accuracies = dict()
        for k in range(1, topk + 1):
            acc = TopkAccuracy(k)
            self.topk_accuracies[k] = acc
            self.add_module(f"val_top{k}_acc", acc)

    @classmethod
    def load(
        cls, ckpt_path: Union[str, Path], device, freeze: bool
    ) -> "TransformerTacticGenerator":
        return load_checkpoint(cls, to_path(ckpt_path), device, freeze)

    def generate(
        self,
        state: str,
        file_path: Path,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        # Prepare the input.
        # logger.debug(f"Input state: {state}")
        # TODO: Should we apply any truncation here?
        tokenized_state = self.tokenizer(
            state, truncation=True, max_length=self.max_seq_len
        )
        if len(tokenized_state.input_ids) >= self.max_seq_len:
            logger.warning(f"The tactic_state is truncated: {state}")
        input_ids = torch.tensor(
            tokenized_state.input_ids, device=self.device
        ).unsqueeze(0)
        attention_mask = torch.tensor(
            tokenized_state.attention_mask, device=self.device
        ).unsqueeze(0)

        # Perform Beam Search.
        output = self.seq2seq.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_seq_len,
            num_beams=num_samples,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping="never",
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        output_text = [
            remove_marks(_)
            for _ in self.tokenizer.batch_decode(
                output.sequences, skip_special_tokens=True
            )
        ]
        assert num_samples > 1
        output_score = output.sequences_scores.tolist()
        tactics_with_scores = list(zip_strict(output_text, output_score))
        # logger.debug(f"Predicted tactics: {str(tactics_with_scores)}")
        return tactics_with_scores

    def forward(
        self,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        tactic_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.seq2seq(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=tactic_ids,
        ).loss

    def training_step(self, batch, batch_idx: int):
        """
        # Don't apply the loss on <a>...</a>
        assert isinstance(self.tokenizer, ByT5Tokenizer)
        masked_tactic_ids = tactic_ids.clone().detach()
        tactic_utf8 = [t.encode("utf-8") for t in batch["tactic"]]
        for i, t in enumerate(tactic_utf8):
            assert len(t) >= self.max_seq_len or (
                masked_tactic_ids[i, len(t)] == self.tokenizer.eos_token_id
                and masked_tactic_ids[i, len(t) - 1] != self.tokenizer.eos_token_id
            )
            for m in re.finditer(b"(?<=<a>).+?</a>", t):
                if m.start() < self.max_seq_len:
                    end = min(m.end(), self.max_seq_len)
                    masked_tactic_ids[i, m.start() : end] = -100
        """

        loss = self(batch["state_ids"], batch["state_mask"], batch["tactic_ids"])
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        self._log_io_texts("train", batch["tactic_ids"], batch["tactic_ids"])
        return loss

    def _log_io_texts(
        self, split: str, state_ids: torch.LongTensor, tactic_ids: torch.LongTensor
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

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        state_ids = batch["state_ids"]
        state_mask = batch["state_mask"]
        tactic_ids = batch["tactic_ids"]

        loss = self(state_ids, state_mask, tactic_ids)

        self.log(f"loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
        self._log_io_texts("val", state_ids, tactic_ids)

        # Generate topk tactic candidates via Beam Search.
        output = self.seq2seq.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=self.num_beams,
            do_sample=False,
            num_return_sequences=self.topk,
            early_stopping="never",
        )
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        batch_size = state_ids.size(0)
        assert len(output_text) == batch_size * self.topk
        tactics_pred = [
            output_text[i * self.topk : (i + 1) * self.topk] for i in range(batch_size)
        ]

        tb = self.logger.experiment
        msg = "\n".join(tactics_pred[0])
        tb.add_text(f"val_preds", f"```\n{msg}\n```", self.global_step)

        # Log the topk accuracies.
        for k in range(1, self.topk + 1):
            topk_acc = self.topk_accuracies[k]
            topk_acc(tactics_pred, batch["tactic"])
            self.log(f"val_top{k}_acc", topk_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )


class RetrievalAugmentedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        state,
        file_path: Path,
        theorem_full_name,
        theorem_pos,
        tokenizer,
        retriever,
        num_beams: int,
    ) -> None:
        super().__init__()
        self.state = state
        self.file_path = file_path
        self.theorem_full_name = theorem_full_name
        self.theorem_pos = theorem_pos
        assert isinstance(tokenizer, ByT5Tokenizer)
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.num_beams = num_beams
        self.premise_names = [None for _ in range(num_beams)]
        self.premise_scores = [None for _ in range(num_beams)]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        prefixes = self.tokenizer.batch_decode(input_ids[:, 1:])

        for i, tactic_prefix in enumerate(prefixes):
            if not is_well_formed(tactic_prefix):
                pdb.set_trace()
                continue
            elif tactic_prefix.endswith(MARK_START_SYMBOL):
                premises, premise_scores = self.retriever.retrieve(
                    self.state,
                    self.file_path,
                    self.theorem_full_name,
                    self.theorem_pos,
                    tactic_prefix,
                    self.num_beams,
                )
                premise_names = [p.full_name + MARK_END_SYMBOL for p in premises]
                # logger.info(f"tactic_prefix: {tactic_prefix}")
                # logger.info(f"premise_names: {premise_names}")
                try:
                    self._update_retrieved_premises(i, premise_names, premise_scores)
                except Exception:
                    pdb.set_trace()
            elif tactic_prefix.endswith(MARK_END_SYMBOL):
                # logger.info(f"tactic_prefix: {tactic_prefix}")
                # logger.info(f"premise_names: {premise_names}")
                try:
                    self._update_retrieved_premises(i, None, None)
                except Exception:
                    pdb.set_trace()

            name_prefix = find_open_mark(tactic_prefix)
            if name_prefix is not None:
                # Modify scores[i].
                possible_suffixes = []
                suffix_scores = []
                for s, p in zip_strict(self.premise_names[i], self.premise_scores[i]):
                    if s.startswith(name_prefix) and s != name_prefix:
                        possible_suffixes.append(s[len(name_prefix) :])
                        suffix_scores.append(p)

                assert len(possible_suffixes) > 0
                suffix_probs = torch.tensor(suffix_scores).softmax(dim=0).tolist()
                byte_probs = defaultdict(float)
                for s, p in zip_strict(possible_suffixes, suffix_probs):
                    byte_probs[s[0].encode("utf-8")] += p

                scores[i].fill_(-float("inf"))
                for b, p in byte_probs.items():
                    b_id = self.tokenizer.convert_tokens_to_ids([b])[0]
                    assert b_id != self.tokenizer.unk_token_id
                    scores[i, b_id] = math.log(p)

        return scores

    def _update_retrieved_premises(
        self,
        beam_idx: int,
        premise_names: Optional[List[str]],
        premise_scores: Optional[List[float]],
    ) -> None:
        assert (premise_names is None) == (premise_scores is None)
        if premise_names is None:
            assert self.premise_names[beam_idx] is not None
            assert self.premise_scores[beam_idx] is not None
        else:
            assert self.premise_names[beam_idx] is None
            assert self.premise_scores[beam_idx] is None
        self.premise_names[beam_idx] = premise_names
        self.premise_scores[beam_idx] = premise_scores

    def process(self, next_beam_indices: List[int]) -> None:
        # if next_beam_indices != list(range(self.num_beams)):
        #    pdb.set_trace()
        self.premise_names = [
            deepcopy(self.premise_names[i]) for i in next_beam_indices
        ]
        self.premise_scores = [
            deepcopy(self.premise_scores[i]) for i in next_beam_indices
        ]


class BeamSearchHelper:
    def __init__(
        self,
        state: str,
        file_path: Path,
        theorem_full_name: str,
        theorem_pos: Pos,
        tokenizer: ByT5Tokenizer,
        retriever,
        num_beams,
        device,
        max_length,
    ) -> None:
        self.logits_processor = RetrievalAugmentedLogitsProcessor(
            state,
            file_path,
            theorem_full_name,
            theorem_pos,
            tokenizer,
            retriever,
            num_beams,
        )

        self.beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=num_beams,
            device=device,
            do_early_stopping="never",
            num_beam_hyps_to_keep=num_beams,
            max_length=max_length,
        )

    @property
    def _beam_hyps(self):
        return self.beam_scorer._beam_hyps

    @property
    def num_beams(self):
        return self.beam_scorer.num_beams

    @property
    def is_done(self):
        return self.beam_scorer.is_done

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.logits_processor(input_ids, scores)

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor]:
        result = self.beam_scorer.process(
            input_ids,
            next_scores,
            next_tokens,
            next_indices,
            pad_token_id,
            eos_token_id,
            beam_indices,
        )
        self.logits_processor.process(result["next_beam_indices"].tolist())
        return result

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor]:
        return self.beam_scorer.finalize(
            input_ids,
            final_beam_scores,
            final_beam_tokens,
            final_beam_indices,
            max_length,
            pad_token_id,
            eos_token_id,
            beam_indices,
        )


class RetrivalAugmentedTacticGenerator(TacticGenerator):
    def __init__(
        self,
        gen_ckpt: Union[str, Path],
        ret_ckpt: Union[str, Path],
        device,
    ) -> None:
        super().__init__()
        self.generator = TransformerTacticGenerator.load(gen_ckpt, device, freeze=True)
        self.retriever = PremiseRetriever.load(ret_ckpt, device, freeze=True)

    def generate(
        self,
        state: str,
        file_path: Path,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        logger.info(state)
        state_ids = self.generator.tokenizer(
            state,
            truncation=True,
            max_length=self.generator.max_seq_len,
            return_tensors="pt",
        ).input_ids.to(self.generator.device)
        if len(state_ids) >= self.generator.max_seq_len:
            logger.warning(f"The tactic_state is truncated: {state}")

        decoder_input_ids = torch.full(
            (num_samples, 1),
            fill_value=self.generator.seq2seq.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.generator.device,
        )
        encoder_outputs = self.generator.seq2seq.encoder(
            state_ids.repeat_interleave(num_samples, dim=0)
        )
        helper = BeamSearchHelper(
            state,
            file_path,
            theorem_full_name,
            theorem_pos,
            self.generator.tokenizer,
            self.retriever,
            num_beams=num_samples,
            device=self.generator.device,
            max_length=self.generator.max_seq_len,
        )
        logits_processor = LogitsProcessorList([helper])
        stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(self.generator.max_seq_len)]
        )
        output = self.generator.seq2seq.beam_search(
            decoder_input_ids,
            helper,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            output_scores=True,
            return_dict_in_generate=True,
            encoder_outputs=encoder_outputs,
        )

        # Return the output.
        output_text = [
            remove_marks(_)
            for _ in self.generator.tokenizer.batch_decode(
                output.sequences, skip_special_tokens=True
            )
        ]
        assert num_samples > 1
        output_score = output.sequences_scores.tolist()
        assert len(output_text) == len(output_score)
        tactics_with_scores = list(zip(output_text, output_score))
        logger.info(f"Predicted tactics: {str(tactics_with_scores)}")
        return tactics_with_scores
