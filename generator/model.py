import pdb
import openai
import math
import torch
import itertools
from time import monotonic
from copy import copy
from pathlib import Path
from lean_dojo import Pos
from loguru import logger
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Metric
from collections import defaultdict
from abc import ABC, abstractmethod
from retrieval.model import PremiseRetriever
from transformers.generation import BeamHypotheses
from typing import List, Dict, Any, Optional, Tuple, Union


from common import (
    get_optimizers,
    remove_marks,
    load_checkpoint,
    zip_strict,
    find_marks,
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
        file_path: List[Path],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
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
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
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
        return load_checkpoint(cls, Path(ckpt_path), device, freeze)

    def generate(
        self,
        state: str,
        file_path: Path,
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
        file_path: List[Path],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        # Prepare the input.
        # logger.debug(f"Input state: {state}")
        assert num_samples > 1
        max_seq_len = self.max_seq_len

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        if tokenized_state.input_ids.size(1) >= max_seq_len:
            logger.warning(f"The tactic_state is truncated: {state}")

        # Perform Beam Search.
        # TODO: Try different temperature and length penalty.
        output = self.t5.generate(
            input_ids=tokenized_state.input_ids.to(self.device),
            attention_mask=tokenized_state.attention_mask.to(self.device),
            max_length=max_seq_len,
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


class LengthDiscountedBeamHypotheses(BeamHypotheses):
    def add(self, hyp: torch.LongTensor, sum_logprobs: float, l: int):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (l**self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, None))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _, _) in enumerate(self.beams)]
                )
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)


class RetrivalAugmentedTacticGenerator(TacticGenerator):
    def __init__(
        self,
        gen_ckpt: Union[str, Path],
        ret_ckpt: Union[str, Path],
        device,
        length_penalty: float,
        temperature: float,
        retrieval_weight: float,
    ) -> None:
        super().__init__()
        logger.debug(f"Loading the generator from {gen_ckpt}")
        self.generator = TransformerTacticGenerator.load(gen_ckpt, device, freeze=True)
        logger.debug(f"Loading the retriever from {ret_ckpt}")
        self.retriever = PremiseRetriever.load(ret_ckpt, device, freeze=True)
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.retrieval_weight = retrieval_weight

        assert isinstance(self.generator.tokenizer, ByT5Tokenizer)
        num_special_tokens = self.generator.tokenizer._num_special_tokens
        self.mark_start_ids = bytes(
            x - num_special_tokens
            for x in self.generator.tokenizer.encode(
                MARK_START_SYMBOL, add_special_tokens=False
            )
        )
        self.mark_end_ids = bytes(
            x - num_special_tokens
            for x in self.generator.tokenizer.encode(
                MARK_END_SYMBOL, add_special_tokens=False
            )
        )

    @property
    def device(self):
        return self.generator.device

    @property
    def tokenizer(self):
        assert isinstance(self.generator.tokenizer, ByT5Tokenizer)
        assert isinstance(self.retriever.tokenizer, ByT5Tokenizer)
        return self.generator.tokenizer

    def generate(
        self,
        state: str,
        file_path: Path,
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
        file_path: List[Path],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        logger.debug(state)

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.generator.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device).repeat_interleave(
            num_samples, dim=0
        )
        encoder_outputs = self.generator.t5.encoder(
            input_ids=state_ids.repeat_interleave(num_samples, dim=0),
            attention_mask=state_mask,
        )

        # TODO: Make `length_penalty` a parameter.
        sequences, scores = self.beam_search(
            state,
            file_path,
            theorem_full_name,
            theorem_pos,
            encoder_outputs,
            state_mask,
            num_samples,
            length_penalty=self.length_penalty,
            early_stopping=False,
            max_length=self.generator.max_seq_len,
        )

        # Return the output.
        raw_output_text = self.generator.tokenizer.batch_decode(
            sequences, skip_special_tokens=True
        )
        raw_scores = scores.tolist()
        logger.debug(
            f"Raw predicted tactics: {str(list(zip(raw_output_text, raw_scores)))}"
        )
        tactics_with_scores = []

        for i in range(len(state)):
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

        return tactics_with_scores

    @torch.no_grad()
    def beam_search(
        self,
        state: List[str],
        file_path: List[Path],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        encoder_outputs,
        attention_mask,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool,
        max_length: int,
    ):
        batch_beam_size = encoder_outputs.last_hidden_state.size(0)
        self._init_retrieved_premises(batch_beam_size)
        batch_size = batch_beam_size // num_beams
        device = self.device
        pad_token_id = self.generator.t5.config.pad_token_id
        eos_token_id = self.generator.t5.config.eos_token_id
        decoder_start_token_id = self.generator.t5.config.decoder_start_token_id

        input_ids = torch.full(
            (batch_beam_size, 1), fill_value=decoder_start_token_id, device=device
        )

        # Initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.flatten()

        beam_hyps = [
            LengthDiscountedBeamHypotheses(
                num_beams, length_penalty, early_stopping, max_length
            )
            for _ in range(batch_size)
        ]
        done = [False for _ in range(batch_size)]
        past_key_values = None

        while True:
            decoder_input_ids = (
                input_ids if past_key_values is None else input_ids[:, -1:]
            )
            outputs = self.generator.t5(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                return_dict=True,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_scores = F.log_softmax(outputs.logits[:, -1, :], dim=-1)
            next_scores_processed = self._process_logits(
                state,
                file_path,
                theorem_full_name,
                theorem_pos,
                batch_size,
                num_beams,
                done,
                input_ids,
                next_token_scores,
            )
            # next_scores_processed = next_token_scores
            next_scores = next_scores_processed + beam_scores.unsqueeze(-1)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            vocab_size = next_scores.size(-1)
            next_scores, next_tokens = torch.topk(
                next_scores.view(batch_size, -1),
                2 * num_beams,
                dim=1,
                largest=True,
                sorted=True,
            )
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            next_beam_scores = torch.zeros(
                (batch_size, num_beams), dtype=next_scores.dtype, device=device
            )
            next_beam_tokens = torch.zeros(
                (batch_size, num_beams), dtype=next_tokens.dtype, device=device
            )
            next_beam_indices = torch.zeros(
                (batch_size, num_beams), dtype=next_indices.dtype, device=device
            )

            for batch_idx, beam_hyp in enumerate(beam_hyps):
                if done[batch_idx]:
                    assert num_beams <= len(beam_hyp)
                    # Pad the batch.
                    next_beam_scores[batch_idx, :] = 0
                    next_beam_tokens[batch_idx, :] = pad_token_id
                    next_beam_indices[batch_idx, :] = 0
                    continue

                beam_idx = 0
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(
                        next_tokens[batch_idx],
                        next_scores[batch_idx],
                        next_indices[batch_idx],
                    )
                ):
                    batch_beam_idx = batch_idx * num_beams + next_index
                    # Add to generated hypotheses if end of sentence.
                    if next_token.item() == eos_token_id:
                        if beam_token_rank >= num_beams:
                            continue
                        tac = self.generator.tokenizer.decode(
                            input_ids[batch_beam_idx], skip_special_tokens=True
                        )
                        premises_length = sum(
                            len(m.group())
                            for m in find_marks(tac, include_symbols=True)
                        )
                        length = 1 + len(tac) - premises_length
                        beam_hyp.add(
                            input_ids[batch_beam_idx].clone(), next_score.item(), length
                        )
                    else:
                        next_beam_scores[batch_idx, beam_idx] = next_score
                        next_beam_tokens[batch_idx, beam_idx] = next_token
                        next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                        beam_idx += 1

                    # Once the beam for next step is full, don't add more tokens to it.
                    if beam_idx >= num_beams:
                        break

                done[batch_idx] = done[batch_idx] or beam_hyp.is_done(
                    next_scores[batch_idx].max().item(), 1 + input_ids.size(-1)
                )

            next_beam_scores = next_beam_scores.flatten()
            next_beam_tokens = next_beam_tokens.flatten()
            next_beam_indices = next_beam_indices.flatten()

            input_ids = torch.cat(
                [
                    input_ids[next_beam_indices, :],
                    next_beam_tokens.unsqueeze(-1),
                ],
                dim=-1,
            )
            past_key_values = self.generator.t5._reorder_cache(
                outputs.past_key_values, next_beam_indices
            )
            self._reorder_retrieved_premises(next_beam_indices)
            beam_scores = next_beam_scores

            if input_ids.size(-1) >= max_length - 1 or all(done):
                break

        # Finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(beam_hyps):
            if done[batch_idx]:
                continue
            # All open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams.
            for beam_id in range(num_beams):
                batch_beam_idx = batch_idx * num_beams + beam_id
                final_score = beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                tac = self.generator.tokenizer.decode(
                    final_tokens, skip_special_tokens=True
                )
                premises_length = sum(
                    len(m.group()) for m in find_marks(tac, include_symbols=True)
                )
                length = 1 + len(tac) - premises_length
                beam_hyp.add(final_tokens, final_score, length)

        # Select the best hypotheses.
        final_beams = list(
            itertools.chain.from_iterable(
                sorted(hyp.beams, key=lambda x: x[0], reverse=True)[:num_beams]
                for hyp in beam_hyps
            )
        )
        assert len(final_beams) == batch_beam_size
        max_len = 1 + max(len(x[1]) for x in final_beams)
        sequences = torch.full(
            (batch_beam_size, max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=device,
        )
        scores = torch.zeros((batch_beam_size,), dtype=torch.float, device=device)
        for i, (s, seq, _) in enumerate(final_beams):
            sequences[i, : len(seq)] = seq
            sequences[i, len(seq)] = eos_token_id
            scores[i] = s

        logger.debug(f"Beam search finished in {monotonic() - self.time_start:.2f} s")
        logger.debug(f"Retrieval time: {self.time_retrieval:.2f} s")
        return sequences, scores

    def _init_retrieved_premises(self, batch_beam_size: int):
        self.premise_names = [None for _ in range(batch_beam_size)]
        self.premise_scores = [None for _ in range(batch_beam_size)]
        self.time_start = monotonic()
        self.time_retrieval = 0.0

    def _reorder_retrieved_premises(self, next_beam_indices: List[int]) -> None:
        self.premise_names = [copy(self.premise_names[i]) for i in next_beam_indices]
        self.premise_scores = [copy(self.premise_scores[i]) for i in next_beam_indices]

    def _in_generation_mode(self, beam_idx: int) -> bool:
        return self.premise_names[beam_idx] is None

    def _in_retrival_mode(self, beam_idx: int) -> bool:
        return self.premise_names[beam_idx] is not None

    def _find_open_mark(self, s: bytes) -> Optional[bytes]:
        """Check if ``s`` has an open :code:`<a>` that is not closed by :code:`</a>`.
        If so, return the substring from the open :code:`<a>` to the end of ``s``."""
        if s.count(self.mark_start_ids) > s.count(self.mark_end_ids):
            return s[s.rfind(self.mark_start_ids) + len(self.mark_start_ids) :]
        else:
            return None

    def _update_retrieved_premises(
        self,
        beam_idx: int,
        premise_names: Optional[List[List[int]]],
        premise_scores: Optional[List[float]],
    ) -> None:
        assert (premise_names is None) == (premise_scores is None)
        self.premise_names[beam_idx] = premise_names
        self.premise_scores[beam_idx] = premise_scores

    def _process_logits(
        self,
        state: List[str],
        file_path: List[Path],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        batch_size: int,
        num_beams: int,
        done: List[bool],
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        device = scores.device
        num_special_tokens = self.generator.tokenizer._num_special_tokens
        scores[:, 256 + num_special_tokens :].fill_(-math.inf)
        scores[:, self.tokenizer.pad_token_id] = -math.inf
        scores[:, self.tokenizer.unk_token_id] = -math.inf
        retrieval_probs = torch.zeros_like(scores)

        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            for beam_idx in range(num_beams):
                batch_beam_idx = batch_idx * num_beams + beam_idx
                tactic_prefix_ids = input_ids[batch_beam_idx, 1:]
                tactic_prefix = bytes((tactic_prefix_ids - num_special_tokens).tolist())
                tactic_prefix_str = self.generator.tokenizer.decode(tactic_prefix_ids)

                if self._in_retrival_mode(batch_beam_idx) and tactic_prefix.endswith(
                    self.mark_start_ids[:-1]
                ):
                    # Prevent MARK_START_SYMBOL from being generated.
                    scores[
                        batch_beam_idx, self.mark_start_ids[-1] + num_special_tokens
                    ] = -math.inf

                if self._in_generation_mode(batch_beam_idx) and tactic_prefix.endswith(
                    self.mark_end_ids[:-1]
                ):
                    # Prevent MARK_END_SYMBOL from being generated.
                    scores[
                        batch_beam_idx, self.mark_end_ids[-1] + num_special_tokens
                    ] = -math.inf

                if tactic_prefix.endswith(self.mark_start_ids):  # <a> detected.
                    # TODO: Don't have to retrieve self.num_beams items.
                    # TODO: Dont' discount the beam score using premise_scores.
                    assert self._in_generation_mode(batch_beam_idx)
                    params = (
                        state[batch_idx],
                        file_path[batch_idx],
                        theorem_full_name[batch_idx],
                        theorem_pos[batch_idx],
                        tactic_prefix_str,
                        num_beams,
                    )
                    time_start = monotonic()
                    # TODO: Retrieval is taking ~50% of the time. Optimize!
                    premises, premise_scores = self.retriever.retrieve(
                        state[batch_idx],
                        file_path[batch_idx],
                        theorem_full_name[batch_idx],
                        theorem_pos[batch_idx],
                        tactic_prefix_str,
                        num_beams,
                    )
                    self.time_retrieval += monotonic() - time_start

                    premise_names = [
                        bytes(
                            x - num_special_tokens
                            for x in self.generator.tokenizer.encode(
                                p.full_name + MARK_END_SYMBOL, add_special_tokens=False
                            )
                        )
                        for p in premises
                    ]
                    self._update_retrieved_premises(
                        batch_beam_idx, premise_names, premise_scores
                    )
                elif tactic_prefix.endswith(self.mark_end_ids):
                    assert self._in_retrival_mode(batch_beam_idx)
                    self._update_retrieved_premises(batch_beam_idx, None, None)

                name_prefix = self._find_open_mark(tactic_prefix)
                if name_prefix is None:
                    continue

                possible_suffixes = []
                suffix_scores = []

                for s, p in zip_strict(
                    self.premise_names[batch_beam_idx],
                    self.premise_scores[batch_beam_idx],
                ):
                    if s.startswith(name_prefix) and s != name_prefix:
                        possible_suffixes.append(s[len(name_prefix) :])
                        suffix_scores.append(p)

                if len(possible_suffixes) > 0:
                    suffix_probs = (
                        (torch.tensor(suffix_scores, device=device) / self.temperature)
                        .softmax(dim=0)
                        .tolist()
                    )
                    for suffix, prob in zip_strict(possible_suffixes, suffix_probs):
                        retrieval_probs[
                            batch_beam_idx, suffix[0] + num_special_tokens
                        ] += prob
                else:
                    retrieval_probs[batch_beam_idx] = F.softmax(
                        scores[batch_beam_idx], dim=0
                    )

                """
                t = 0.5  # TODO: Tune
                suffix_probs = (
                    (torch.tensor(suffix_scores) / t).softmax(dim=0).tolist()
                )
                byte_probs = defaultdict(float)
                for s, p in zip_strict(possible_suffixes, suffix_probs):
                    byte_probs[s[0] + num_special_tokens] += p

                scores[
                    batch_beam_idx, scores[batch_beam_idx] < math.log(0.3)
                ] = -math.inf
                for b_id, p in byte_probs.items():
                    scores[batch_beam_idx, b_id] = max(
                        math.log(p), scores[batch_beam_idx, b_id]
                    )

                # TODOs: Using generator scores does not make sense for novel premises.
                # TODOs: Let's try retrieved premises scores followed by softmax with temperature 0.5, followed by thresholding.
                """
                """
                mask_keep = scores[batch_beam_idx] >= math.log(0.2)
                for s in possible_suffixes:
                    mask_keep[s[0] + num_special_tokens] = True
                scores[batch_beam_idx, ~mask_keep] = -math.inf
                """

        scores = (
            self.retrieval_weight * retrieval_probs
            + (1 - self.retrieval_weight) * F.softmax(scores, dim=1)
        ).log()
        scores = F.log_softmax(scores, dim=1)
        assert not scores.isnan().any()
        return scores


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
