"""Lightning module for the tactic generator."""

import os
import torch
import shutil
import openai
import pickle
from lean_dojo import Pos
from loguru import logger
import pytorch_lightning as pl
from torchmetrics import Metric
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM


from common import (
    zip_strict,
    remove_marks,
    IndexedCorpus,
    get_optimizers,
    load_checkpoint,
    format_augmented_state,
    format_inputs_decoder,
    extract_inputs_decoder,
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
        inputs: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def batch_generate(
        self,
        inputs: List[str],
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
        eval_num_workers: int,
        eval_num_gpus: int,
        eval_num_theorems: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        length_penalty: float = 0.0,
        ret_ckpt_path: Optional[str] = None,
        decoder_only: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.eval_num_retrieved = eval_num_retrieved
        self.eval_num_workers = eval_num_workers
        self.eval_num_gpus = eval_num_gpus
        self.eval_num_theorems = eval_num_theorems
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.decoder_only = decoder_only

        if ret_ckpt_path is None:
            logger.info("Without retrieval")
            self.retriever = None
        else:
            logger.info(f"Loading the retriever from {ret_ckpt_path}")
            self.retriever = PremiseRetriever.load(
                ret_ckpt_path, self.device, freeze=True
            )

        # Set the generator
        if self.decoder_only:
            # Load general decoder-only models
            self.generator = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            # Load t5 specifically
            if 't5' in self.model_name.lower():
                self.generator = T5ForConditionalGeneration.from_pretrained(self.model_name)
            # Load as general seq2seq models
            else:
                self.generator = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Set the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.generator.config.pad_token_id = self.tokenizer.eos_token_id
        
        if self.decoder_only:
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side = "left"

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
        output_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.decoder_only:
            labels = input_ids  # CausalLM will handle label shift internally
        else:
            labels = output_ids
            
        return self.generator(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=labels,
        ).loss


    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        output_ids = batch["output_ids"] if not self.decoder_only else None
        
        loss = self(
            batch["input_ids"],
            batch["input_mask"],
            output_ids,
        )
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        self._log_io_texts("train", batch["input_ids"], output_ids)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    def _log_io_texts(
        self,
        split: str,
        input_ids: torch.LongTensor,
        output_ids: Optional[torch.LongTensor] = None,
    ) -> None:
        tb = self.logger.experiment
        inp = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        tb.add_text(f"{split}_input", f"```\n{inp}\n```", self.global_step)
        
        if not self.decoder_only:
            oup_ids = torch.where(
                output_ids[0] == -100, self.tokenizer.pad_token_id, output_ids[0]
            )
            oup = self.tokenizer.decode(oup_ids, skip_special_tokens=True)
            tb.add_text(f"{split}_output", f"`{oup}`", self.global_step)

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
        """Get the validation loss, and the accuracy in generated outputs."""
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        output_ids = batch["output_ids"] if not self.decoder_only else None

        # Validation for loss
        loss = self(input_ids, input_mask, output_ids)
        self.log(f"loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
        self._log_io_texts("val", input_ids, output_ids)
        
        # Validation for accuracy
        if self.decoder_only:
            # Format the inputs to get the conditional input (everything before proofstep)
            inputs = extract_inputs_decoder(batch["inputs"], ex_type='condition')
            tokenized_inputs = self.tokenizer(
                inputs,
                padding="longest",
                max_length=self.max_inp_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokenized_inputs.input_ids.to(self.device)
            input_mask = tokenized_inputs.attention_mask.to(self.device)
            
            # Set the max_length for generation to include the input and output lengths
            max_length = self.max_inp_seq_len + self.max_oup_seq_len
        else:
            max_length = self.max_oup_seq_len      

        # Generate topk tactic candidates via Beam Search.
        output = self.generator.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            max_length=max_length,
            num_beams=self.num_beams,
            do_sample=False,
            num_return_sequences=self.num_beams,
            early_stopping=False,
        )
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        batch_size = input_ids.size(0)
        assert len(output_text) == batch_size * self.num_beams

        # Extract true tactics and predicted tactics for comparison
        if self.decoder_only:
            # Extract only the tactic part from the inputs for decoder-only models
            tactics_true = extract_inputs_decoder(batch["inputs"], ex_type='tactic')
        else:
            tactics_true = batch["outputs"]
        
        tactics_pred = [
            output_text[i * self.num_beams : (i + 1) * self.num_beams]
            for i in range(batch_size)
        ]

        # If decoder-only, extract the tactic part from predicted outputs
        if self.decoder_only:
            tactics_pred = [
                extract_inputs_decoder(tactic_preds, ex_type='tactic')
                for tactic_preds in tactics_pred
            ]

        tb = self.logger.experiment
        msg = "\n".join(tactics_pred[0])
        tb.add_text(f"preds_val", f"```\n{msg}\n```", self.global_step)

        # Log the topk accuracies.
        for k in range(1, self.num_beams + 1):
            topk_acc = self.topk_accuracies[k]
            topk_acc(tactics_pred, tactics_true)
            self.log(
                f"top{k}_acc_val",
                topk_acc,
                on_step=False, 
                on_epoch=True, 
                sync_dist=True,
            )

    def on_validation_epoch_end(self) -> None:
        if self.eval_num_theorems == 0:
            return

        from prover.evaluate import evaluate  # Avoid circular import.

        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last-tmp.ckpt"
        self.trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}. Evaluating...")
        torch.cuda.empty_cache()

        data_path = self.trainer.datamodule.data_path
        if self.retriever is None:
            acc = evaluate(
                data_path=data_path,
                num_workers=self.eval_num_workers,
                num_gpus=self.eval_num_gpus,
                num_theorems=self.eval_num_theorems,
                ckpt_path=ckpt_path,
            )
        else:
            self.retriever.reindex_corpus(self.trainer.datamodule.eval_batch_size)
            corpus_path = f"{self.trainer.log_dir}/checkpoints/indexed_corpus.pickle"
            pickle.dump(
                IndexedCorpus(
                    self.retriever.corpus, self.retriever.corpus_embeddings.cpu()
                ),
                open(corpus_path, "wb"),
            )
            acc = evaluate(
                data_path=data_path,
                num_workers=self.eval_num_workers,
                num_gpus=self.eval_num_gpus,
                num_theorems=self.eval_num_theorems,
                ckpt_path=ckpt_path,
                indexed_corpus_path=corpus_path,
            )

        self.log("Pass@1_val", acc, on_step=False, on_epoch=True, sync_dist=True)
        logger.info(f"Pass@1: {acc}")

        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)

    ##############
    # Prediction #
    ##############

    def generate(
        self,
        inputs: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return self.batch_generate(
            inputs=[inputs], 
            file_path=[file_path], 
            theorem_full_name=[theorem_full_name], 
            theorem_pos=[theorem_pos], 
            num_samples=num_samples,
        )[0]

    def batch_generate(
        self,
        inputs: List[str],  # tactic states
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        logger.debug(inputs)
        
        if self.retriever is not None:
            if self.decoder_only:  # Extract the states
                inputs = extract_inputs_decoder(inputs, ex_type='state')
                
            retrieved_premises, _ = self.retriever.retrieve(
                inputs,
                file_path,
                theorem_full_name,
                theorem_pos,
                self.eval_num_retrieved,
            )
            inputs = [
                format_augmented_state(s, premises, self.max_inp_seq_len, p_drop=0.0)
                for s, premises in zip_strict(inputs, retrieved_premises)
            ]
            
            if self.decoder_only:  # Add back the condition
                inputs = extract_inputs_decoder(inputs, ex_type='condition_with_states')
            
        else:
            if self.decoder_only:  # Truncate the inputs
                inputs = extract_inputs_decoder(inputs, ex_type='condition')

        tokenized_inputs = self.tokenizer(
            inputs,
            padding="longest",
            max_length=self.max_inp_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized_inputs.input_ids.to(self.device)
        input_mask = tokenized_inputs.attention_mask.to(self.device)

        # Handle the max output length
        if not self.decoder_only:
            max_length = self.max_oup_seq_len
        else: 
            max_length = self.max_inp_seq_len + self.max_oup_seq_len

        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            max_length=max_length,
            num_beams=num_samples,
            length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Get the default output text and scores
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        raw_scores = output.sequences_scores.tolist()
        
        # Handle the case for decoder-only models
        if self.decoder_only:
            # Get the raw_output_text with only generated tactics
            raw_output_text = [
                output_str.split("[PROOFSTEP]\n", 1)[-1].strip() 
                if "[PROOFSTEP]\n" in output_str else ''
                for output_str in raw_output_text
            ]
            # Get the probs with only generated tactics
            probs = torch.stack(output.scores, dim=1).softmax(-1)
            gen_seqs = output.sequences[:, input_ids.size(1):]
            gen_probs = torch.gather(probs, 2, gen_seqs[:, :, None]).squeeze(-1)
            eos_mask = gen_seqs != self.tokenizer.eos_token_id
            raw_scores = (gen_probs * eos_mask).sum(-1).tolist()
        
        # Generate the outputs with scores (only tactics are retained in outputs)
        outputs_with_scores = []

        for i in range(len(inputs)):
            output_text = []
            output_score = []

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = remove_marks(raw_output_text[j])
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            outputs_with_scores.append(list(zip_strict(output_text, output_score)))

        return outputs_with_scores


class GPT4TacticGenerator(TacticGenerator):
    def __init__(
        self,
        organization: str,
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 1024,
        num_retries: int = 3,
        threshold: float = 0.9,
    ):
        super().__init__()
        openai.organization = organization
        openai.api_key = api_key
        self.model = model
        self.default_prompt = "You are an expert in theorem proving in Lean. We are trying to solve the Lean theorem 'THEOREM_FULL_NAME' from the mathlib file 'FILE_PATH'. The current tactic state is: 'TACTIC_STATE'. Suggest exactly NUM_SAMPLES unique tactics to progress in solving 'THEOREM_FULL_NAME', along with their confidence levels as a float between 0 and 1. Rank them in order of effectiveness. Present the tactics and their confidence levels as comma-separated tuples in this format: #(tactic_{1}, confidence_{1})#, #(tactic_{2}, confidence_{2})#, ..., #(tactic_{NUM_SAMPLES}, confidence_{NUM_SAMPLES})#."
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.threshold = threshold

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        prompt = (
            self.default_prompt.replace("TACTIC_STATE", state)
            .replace("FILE_PATH", file_path)
            .replace("THEOREM_FULL_NAME", theorem_full_name)
            .replace("NUM_SAMPLES", str(int(num_samples / self.threshold)))
        )
        logger.info(prompt)

        for _ in range(self.num_retries):
            response = None
            # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=0,
                    max_tokens=self.max_tokens,
                    # stop="E:" #
                )
            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                logger.info(f"OpenAI API returned an API Error: {e}")
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                logger.info(f"Failed to connect to OpenAI API: {e}")
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                logger.info(f"OpenAI API request exceeded rate limit: {e}")
                continue
            except Exception as e:
                logger.info(e)
                continue

            if response is None:
                continue

            logger.info(f"GPT-4 response: {response}")
            output = response["choices"][0]["message"]["content"]
            indices = []

            for i, c in enumerate(output):
                if c == "#":
                    indices.append(i)

            tactics_with_scores = []

            for i in range(1, len(indices), 2):
                tactic_and_confidence = output[indices[i - 1] + 1 : indices[i]].strip()

                try:
                    while tactic_and_confidence[0] == "(":
                        tactic_and_confidence = tactic_and_confidence[1:]

                    if tactic_and_confidence[-1] == ")":
                        tactic_and_confidence = tactic_and_confidence[:-1]

                    split_index = tactic_and_confidence.rindex(",")
                    tactic = tactic_and_confidence[:split_index].strip()
                    confidence = float(tactic_and_confidence[split_index + 1 :].strip())
                except Exception as e:
                    logger.info(e)
                    logger.info(
                        f"{self.model} output {output[indices[i-1]+1:indices[i]]} was not formatted correctly and could not be parsed."
                    )
                    continue

                tactics_with_scores.append((tactic, confidence))

            if len(tactics_with_scores) < int(self.threshold * num_samples):
                continue

            tactics_with_scores = sorted(
                tactics_with_scores, key=lambda x: x[1], reverse=True
            )[: min(num_samples, len(tactics_with_scores))]
            logger.debug(f"GPT-4 tactics: {tactics_with_scores}")
            logger.debug(
                f"GPT-4 tactic count requested: {num_samples} / {self.threshold} = {int(num_samples / self.threshold)}"
            )
            logger.debug(
                f"GPT-4 tactic count received and parsed: {len(tactics_with_scores)}"
            )
            return tactics_with_scores

        raise ValueError("GPT-4 outputs are unparsable.")

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, t, p, num_samples)
            for s, f, t, p in zip_strict(
                state, file_path, theorem_full_name, theorem_pos
            )
        ]


class FixedTacticGenerator(TacticGenerator):
    def __init__(self, tactic, module) -> None:
        self.tactic = tactic
        self.module = module

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return [(f"{{ {self.tactic} }}", 1.0)]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, tfn, tp, num_samples)
            for s, f, tfn, tp in zip(state, file_path, theorem_full_name, theorem_pos)
        ]
