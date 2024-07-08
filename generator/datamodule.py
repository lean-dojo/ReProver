"""Data module for the tactic generator."""

import os
import json
import pickle
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ByT5Tokenizer

from common import (
    Batch,
    Corpus,
    Example,
    format_state,
    remove_marks,
    format_tactic,
    format_augmented_state,
    format_inputs_decoder,
)


class GeneratorDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        corpus: Corpus,
        keep_marks: bool,
        preds: List[Dict[str, Any]],
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        normalize_tactics: bool,
        tokenizer: ByT5Tokenizer,
        is_train: bool,
        decoder_only: bool = False,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.keep_marks = keep_marks
        self.preds = preds
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = self._load_data(data_path, normalize_tactics)
        self.decoder_only = decoder_only
        
        # Default setup for the tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_data(self, data_path: str, normalize_tactics: bool) -> List[Example]:
        data = []
        for thm in tqdm(json.load(open(data_path))):
            for tac in thm["traced_tactics"]:
                if "annotated_tactic" in tac:
                    tactic = format_tactic(*tac["annotated_tactic"], normalize_tactics)
                else:
                    tactic = format_tactic(tac["tactic"], [], normalize_tactics)
                if not self.keep_marks:
                    tactic = remove_marks(tactic)
                data.append(
                    {
                        "url": thm["url"],
                        "commit": thm["commit"],
                        "file_path": thm["file_path"],
                        "full_name": thm["full_name"],
                        "state": format_state(tac["state_before"]),
                        "tactic": tactic,
                    }
                )

        logger.info(f"{len(data)} examples loaded")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]

        if self.preds is not None:
            file_path = ex["file_path"]
            pred = self.preds[(file_path, ex["full_name"], ex["state"])]
            ex["state"] = format_augmented_state(
                ex["state"],
                pred["retrieved_premises"],
                self.max_inp_seq_len,
                self.p_drop if self.is_train else 0.0,
            )

        if not self.keep_marks:
            ex["state"] = remove_marks(ex["state"])

        return ex

    def collate(self, examples: List[Example]) -> Batch:
        batch = {}
        
        # Set up for inputs
        if not self.decoder_only:
            inputs = [ex["state"] for ex in examples]
            max_length = self.max_inp_seq_len  
        else:
            inputs = format_inputs_decoder(examples)
            
            # By default we assume max_inp_seq_len to be the the state length in config
            max_length = self.max_inp_seq_len + self.max_oup_seq_len
            
        tokenized_inputs = self.tokenizer(
            inputs,
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        batch["inputs"] = inputs
        batch["input_ids"] = tokenized_inputs.input_ids
        batch["input_mask"] = tokenized_inputs.attention_mask
        
        # Set up for separate outputs for encoder-decoder models
        if not self.decoder_only:
            outputs = [ex["tactic"] for ex in examples]
            tokenized_outputs = self.tokenizer(
                outputs,
                padding="longest",
                max_length=self.max_oup_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            output_ids = tokenized_outputs.input_ids
            output_ids[output_ids == self.tokenizer.pad_token_id] = -100
            
            batch["outputs"] = outputs
            batch["output_ids"] = output_ids
            batch["output_mask"] = tokenized_outputs.attention_mask

        # Copy other fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        keep_marks: bool,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        normalize_tactics: bool,
        num_workers: int,
        corpus_path: Optional[str] = None,
        preds_path: Optional[str] = None,
        decoder_only: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        if corpus_path is not None:
            self.corpus = Corpus(corpus_path)
        else:
            self.corpus = None
        self.keep_marks = keep_marks
        self.model_name = model_name
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        self.normalize_tactics = normalize_tactics
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.decoder_only = decoder_only

        if preds_path is None:
            logger.info("Without retrieval data")
            self.preds = None
        else:
            logger.info("With retrieval data")
            self.preds = {}
            for pred in pickle.load(open(preds_path, "rb")):
                ctx = pred["context"]
                self.preds[ctx.path, ctx.theorem_full_name, ctx.state] = pred

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = GeneratorDataset(
                data_path=os.path.join(self.data_path, "train.json"),
                corpus=self.corpus,
                keep_marks=self.keep_marks,
                preds=self.preds,
                max_inp_seq_len=self.max_inp_seq_len,
                max_oup_seq_len=self.max_oup_seq_len,
                p_drop=self.p_drop,
                normalize_tactics=self.normalize_tactics,
                tokenizer=self.tokenizer,
                is_train=True,
                decoder_only=self.decoder_only,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = GeneratorDataset(
                data_path=os.path.join(self.data_path, "val.json"),
                corpus=self.corpus,
                keep_marks=self.keep_marks,
                preds=self.preds,
                max_inp_seq_len=self.max_inp_seq_len,
                max_oup_seq_len=self.max_oup_seq_len,
                p_drop=self.p_drop,
                normalize_tactics=self.normalize_tactics,
                tokenizer=self.tokenizer,
                is_train=False,
                decoder_only=self.decoder_only,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
