import pdb
import openai
import math
import torch
from time import monotonic
from copy import copy
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
import torch.nn.functional as F
from torchmetrics import Metric
from collections import defaultdict
from abc import ABC, abstractmethod
from retrieval.model import PremiseRetriever
from transformers.generation import BeamHypotheses, BeamSearchEncoderDecoderOutput
from typing import List, Dict, Any, Optional, Tuple, Union


from common import (
    get_optimizers,
    remove_marks,
    is_well_formed,
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

    @abstractmethod
    def batch_generate(
        self,
        state: List[str],
        file_path: Path,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError


class TransformerTacticGenerator(TacticGenerator, pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        self.topk_accuracies = dict()
        for k in range(1, num_beams + 1):
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
        return self.batch_generate(
            [state], file_path, theorem_full_name, theorem_pos, num_samples
        )[0]

    def batch_generate(
        self,
        state: List[str],
        file_path: Path,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        # Prepare the input.
        # logger.debug(f"Input state: {state}")
        assert num_samples > 1

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        if tokenized_state.input_ids.size(1) >= self.max_seq_len:
            logger.warning(f"The tactic_state is truncated: {state}")

        # Perform Beam Search.
        # TODO: Try different temperature and length penalty.
        output = self.t5.generate(
            input_ids=tokenized_state.input_ids.to(self.device),
            attention_mask=tokenized_state.attention_mask.to(self.device),
            max_length=self.max_seq_len,
            num_beams=num_samples,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=-0.5,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        raw_scores = output.sequences_scores.tolist()
        batch_size = len(state)
        tactics_with_scores = []

        for i in range(batch_size):
            raw_output_text_i = raw_output_text[i * num_samples : (i + 1) * num_samples]
            raw_scores_i = raw_scores[i * num_samples : (i + 1) * num_samples]
            output_text = []
            output_score = []
            for t, s in zip_strict(raw_output_text_i, raw_scores_i):
                t = remove_marks(t)
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(s)
            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        # logger.debug(f"Predicted tactics: {str(tactics_with_scores)}")
        return tactics_with_scores

    def forward(
        self,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        tactic_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.t5(
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
        self._log_io_texts("train", batch["state_ids"], batch["tactic_ids"])
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
        output = self.t5.generate(
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
        tb.add_text(f"val_preds", f"```\n{msg}\n```", self.global_step)

        # Log the topk accuracies.
        for k in range(1, self.num_beams + 1):
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
        self.mark_start_ids = bytes(
            tokenizer.encode(MARK_START_SYMBOL, add_special_tokens=False)
        )
        self.mark_end_ids = bytes(
            tokenizer.encode(MARK_END_SYMBOL, add_special_tokens=False)
        )
        self.time_retrieval = 0.0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for beam_idx in range(self.num_beams):
            tactic_prefix = bytes(input_ids[beam_idx, 1:].tolist())
            tactic_prefix_str = self.tokenizer.decode(tactic_prefix)
            assert is_well_formed(tactic_prefix_str)

            if self._in_generation_mode(beam_idx) and tactic_prefix.endswith(
                self.mark_end_ids[:-1]
            ):
                # Prevent MARK_END_SYMBOL from being generated.
                scores[beam_idx, self.mark_end_ids[-1]] = -float("inf")
                scores[beam_idx] = F.log_softmax(scores[beam_idx], dim=0)

            if tactic_prefix.endswith(self.mark_start_ids):
                # TODO: Don't have to retrieve self.num_beams items.
                # TODO: Dont' discount the beam score using premise_scores.
                assert self._in_generation_mode(beam_idx)
                time_start = monotonic()
                premises, premise_scores = self.retriever.retrieve(
                    self.state,
                    self.file_path,
                    self.theorem_full_name,
                    self.theorem_pos,
                    tactic_prefix_str,
                    self.num_beams,
                )
                self.time_retrieval += monotonic() - time_start
                premise_names = [
                    bytes(
                        self.tokenizer.encode(
                            p.full_name + MARK_END_SYMBOL, add_special_tokens=False
                        )
                    )
                    for p in premises
                ]
                self._update_retrieved_premises(beam_idx, premise_names, premise_scores)
            elif tactic_prefix.endswith(self.mark_end_ids):
                assert self._in_retrival_mode(beam_idx)
                self._update_retrieved_premises(beam_idx, None, None)

            name_prefix = self._find_open_mark(tactic_prefix)

            if name_prefix is not None:
                # Modify scores[i].
                possible_suffixes = []
                suffix_scores = []
                for s, p in zip_strict(
                    self.premise_names[beam_idx], self.premise_scores[beam_idx]
                ):
                    if s.startswith(name_prefix) and s != name_prefix:
                        possible_suffixes.append(s[len(name_prefix) :])
                        suffix_scores.append(p)
                assert len(possible_suffixes) > 0

                suffix_probs = torch.tensor(suffix_scores).softmax(dim=0).tolist()
                byte_probs = defaultdict(float)
                for s, p in zip_strict(possible_suffixes, suffix_probs):
                    byte_probs[s[0]] += p

                scores[beam_idx].fill_(-float("inf"))
                for b_id, p in byte_probs.items():
                    scores[beam_idx, b_id] = math.log(p)

        return scores

    def _find_open_mark(self, s: bytes) -> Optional[bytes]:
        """Check if ``s`` has an open :code:`<a>` that is not closed by :code:`</a>`.
        If so, return the substring from the open :code:`<a>` to the end of ``s``."""
        if s.count(self.mark_start_ids) > s.count(self.mark_end_ids):
            return s[s.rfind(self.mark_start_ids) + len(self.mark_start_ids) :]
        else:
            return None

    def _in_generation_mode(self, beam_idx: int) -> bool:
        return self.premise_names[beam_idx] is None

    def _in_retrival_mode(self, beam_idx: int) -> bool:
        return self.premise_names[beam_idx] is not None

    def _update_retrieved_premises(
        self,
        beam_idx: int,
        premise_names: Optional[List[List[int]]],
        premise_scores: Optional[List[float]],
    ) -> None:
        assert (premise_names is None) == (premise_scores is None)
        self.premise_names[beam_idx] = premise_names
        self.premise_scores[beam_idx] = premise_scores

    def process(self, next_beam_indices: List[int]) -> None:
        self.premise_names = [copy(self.premise_names[i]) for i in next_beam_indices]
        self.premise_scores = [copy(self.premise_scores[i]) for i in next_beam_indices]


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
            do_early_stopping=False,
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
        logger.debug(f"time_retrieval: {self.logits_processor.time_retrieval}")
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
        logger.debug(state)

        state_ids = self.generator.tokenizer(
            state,
            truncation=True,
            max_length=self.generator.max_seq_len,
            return_tensors="pt",
        ).input_ids.to(self.generator.device)

        if len(state_ids) >= self.generator.max_seq_len:
            logger.warning(f"The tactic_state is truncated: {state}")

        encoder_outputs = self.generator.t5.encoder(
            state_ids.repeat_interleave(num_samples, dim=0)
        )

        time_start = monotonic()
        # TODO: Make `length_penalty` a parameter.
        sequences, scores = self.beam_search(
            encoder_outputs,
            num_samples,
            length_penalty=-0.5,
            early_stopping=False,
            max_length=self.generator.max_seq_len,
        )
        logger.debug(f"Beam search finished in {monotonic() - time_start:.2f} seconds.")

        # Return the output.
        output_text = self.generator.tokenizer.batch_decode(
            sequences, skip_special_tokens=True
        )
        logger.debug(f"Predicted tactics: {output_text}")
        output_text = [remove_marks(_) for _ in output_text]
        assert len(set(output_text)) == len(output_text)
        assert num_samples > 1
        tactics_with_scores = list(zip_strict(output_text, scores.tolist()))
        # logger.debug(f"Predicted tactics: {str(tactics_with_scores)}")
        return tactics_with_scores

    def batch_generate(
        self,
        state: List[str],
        file_path: Path,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError

    def beam_search(
        self,
        encoder_outputs,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool,
        max_length: int,
    ):
        device = self.generator.device
        decoder_input_ids = torch.full(
            (num_beams, 1),
            fill_value=self.generator.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((num_beams,), dtype=torch.float, device=device)
        beam_scores[1:] = -1e9

        beam_hyps = BeamHypotheses(
            num_beams, length_penalty, early_stopping, max_length
        )

        while True:
            outputs = self.generator.t5(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                return_dict=True,
            )

            next_token_scores = F.log_softmax(outputs.logits[:, -1, :], dim=-1)
            # next_scores_processed = logits_processor(input_ids, next_scores)
            # renormalize_logits=True,
            next_scores = next_token_scores + beam_scores.unsqueeze(-1)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            vocab_size = next_scores.size(-1)
            next_scores, next_tokens = torch.topk(
                next_scores.view(-1), 2 * num_beams, largest=True, sorted=True
            )
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            beam_idx = 0
            next_beam_scores = torch.zeros(
                (num_beams,), dtype=next_scores.dtype, device=device
            )
            next_beam_tokens = torch.zeros(
                (num_beams,), dtype=next_tokens.dtype, device=device
            )
            next_beam_indices = torch.zeros(
                (num_beams,), dtype=next_indices.dtype, device=device
            )

            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens, next_scores, next_indices)
            ):
                # Add to generated hypotheses if end of sentence.
                if next_token.item() == self.generator.t5.config.eos_token_id:
                    if beam_token_rank >= num_beams:
                        continue
                    beam_hyps.add(
                        decoder_input_ids[next_index].clone(), next_score.item()
                    )
                else:
                    next_beam_scores[beam_idx] = next_score
                    next_beam_tokens[beam_idx] = next_token
                    next_beam_indices[beam_idx] = next_index
                    beam_idx += 1

                # Once the beam for next step is full, don't add more tokens to it.
                if beam_idx >= num_beams:
                    break

            decoder_input_ids = torch.cat(
                [
                    decoder_input_ids[next_beam_indices, :],
                    next_beam_tokens.unsqueeze(-1),
                ],
                dim=-1,
            )
            beam_scores = next_beam_scores

            cur_len = decoder_input_ids.size(-1)
            if cur_len >= max_length or beam_hyps.is_done(
                next_scores.max().item(), cur_len
            ):
                break

        assert len(beam_hyps) == num_beams
        final_beams = sorted(beam_hyps.beams, key=lambda x: x[0], reverse=True)
        max_len = 1 + max(len(x[1]) for x in final_beams)
        sequences = torch.full(
            (num_beams, max_len),
            fill_value=self.generator.t5.config.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        scores = torch.zeros((num_beams,), dtype=torch.float, device=device)

        for i, (s, seq, _) in enumerate(final_beams):
            sequences[i, : len(seq)] = seq
            sequences[i, len(seq)] = self.generator.t5.config.eos_token_id
            scores[i] = s

        return sequences, scores


class GPT4TacticGenerator(TacticGenerator):
    def __init__(
        self,
        organization: str = "org-cOXBemL38ej1bDIYcLdzQaIC",
        api_key: str = "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",
        model: str = "gpt-4",
        max_tokens: int = 512,
    ):
        openai.organization = organization
        openai.api_key = api_key
        self.model = model
        self.default_prompt = "Given the Lean theorem `THEOREM_FULL_NAME` in the mathlib file path `FILE_PATH`, we are currently at the tactic state\n```\nTACTIC_STATE\n```\nBased on this context, return exactly NUM_SAMPLES unique comma-separated tuples with each tuple surrounded by a '#' character on both ends. Each tuple should contains a string representing a tactic which would make the given tactic state easier to solve, and a float between 0 and 1 containing the confidence level that this tactic will succeed and be in the final proof."
        self.max_tokens = max_tokens

    def generate(
        self,
        state: str,
        file_path: Path,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        prompt = (
            self.default_prompt.replace("TACTIC_STATE", state)
            .replace("FILE_PATH", str(file_path))
            .replace("THEOREM_FULL_NAME", theorem_full_name)
            .replace("NUM_SAMPLES", str(num_samples + 3))
        )
        logger.debug(prompt)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )

        output = response["choices"][0]["message"]["content"]
        logger.debug(output)

        indices = []

        for i, chr in enumerate(output):
            if chr == "#":
                indices.append(i)

        tactics = []

        for i in range(1, len(indices), 2):
            tactic_and_confidence = output[indices[i - 1] + 1 : indices[i]]

            try:
                tactic = tactic_and_confidence.split(",")[0].strip()

                if tactic[0] == "(":
                    tactic = tactic[1:]

                if tactic[0] == '"':
                    tactic = tactic[1:]

                if tactic[-1] == '"':
                    tactic = tactic[:-1]
            except:
                print(
                    f"{self.model} output {tactic_and_confidence} was not formatted correctly and tactic {tactic_and_confidence.split(',')[0].strip()} could not be parsed."
                )
                continue

            try:
                confidence = tactic_and_confidence.split(",")[-1].strip()

                if confidence[0] == "(":
                    confidence = confidence[1:]

                if confidence[-1] == ")":
                    confidence = confidence[:-1]

                confidence = float(confidence)
            except:
                print(
                    f"{self.model} output {tactic_and_confidence} was not formatted correctly and confidence {tactic_and_confidence.split(',')[-1].strip()} could not be parsed."
                )
                continue

            tactics.append((tactic, confidence))

        tactics_with_scores = sorted(tactics, key=lambda x: x[1], reverse=True)[
            : min(num_samples, len(tactics))
        ]
        logger.debug(tactics_with_scores)
        return tactics_with_scores

    def batch_generate(
        self,
        state: List[str],
        file_path: Path,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, file_path, theorem_full_name, theorem_pos, num_samples)
            for s in state
        ]
