import pdb
import math
import json
import torch
import random
import pickle
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from copy import deepcopy
from lean_dojo import Pos
import pytorch_lightning as pl
from typing import Union, Optional, List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ByT5Tokenizer


from common import (
    Context,
    Corpus,
    format_state,
    format_tactic,
    Example,
    Batch,
    MARK_START_SYMBOL,
    find_marks,
)


class RerankerDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        preds: List[Dict[str, Any]],
        max_seq_len: int,
        tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.preds = preds
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.data = self._load_data(data_path)
        pdb.set_trace()

    def _load_data(self, data_path: Path) -> List[Example]:
        data = []
        logger.info(f"Loading data from {data_path}")

        for thm in tqdm(json.load(data_path.open())):
            repo_name = thm["url"].split("/")[-1]
            file_path = Path(repo_name) / thm["file_path"]

            for tac in thm["traced_tactics"]:
                state = tac["state_before"]
                try:
                    pred = self.preds[(file_path, thm["full_name"], state)]
                except KeyError:
                    logger.warning("skip")
                for premise in pred["all_pos_premises"]:
                    data.append({"state": state, "premise": premise, "label": True})
                for premise in pred["retrieved_premises"]:
                    if premise not in pred["all_pos_premises"]:
                        data.append(
                            {"state": state, "premise": premise, "label": False}
                        )

        logger.info(f"{len(data)} examples loaded.")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        if not self.is_train:
            return self.data[idx]

        pdb.set_trace()

        return ex

    def collate(self, examples: List[Example]) -> Batch:
        batch = {}

        # Tokenize the context.
        context = [ex["context"] for ex in examples]
        tokenized_context = self.tokenizer(
            [c.serialize() for c in context],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        batch["context"] = context
        batch["context_ids"] = tokenized_context.input_ids
        batch["context_mask"] = tokenized_context.attention_mask

        # Tokenize the label and premises.
        if self.is_train:
            pos_premise = [ex["pos_premise"] for ex in examples]
            tokenized_pos_premise = self.tokenizer(
                [p.serialize() for p in pos_premise],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            batch["pos_premise"] = pos_premise
            batch["pos_premise_ids"] = tokenized_pos_premise.input_ids
            batch["pos_premise_mask"] = tokenized_pos_premise.attention_mask

            batch_size = len(examples)
            label = torch.zeros(batch_size, batch_size * (1 + self.num_negatives))

            # Check if one's negative is another's positive
            for j in range(batch_size):
                all_pos_premises = examples[j]["all_pos_premises"]
                for k in range(batch_size * (1 + self.num_negatives)):
                    if k < batch_size:
                        pos_premise_k = examples[k]["pos_premise"]
                    else:
                        pos_premise_k = examples[k % batch_size]["neg_premises"][
                            k // batch_size - 1
                        ]
                    label[j, k] = float(pos_premise_k in all_pos_premises)

            batch["label"] = label
            batch["neg_premises"] = []
            batch["neg_premises_ids"] = []
            batch["neg_premises_mask"] = []

            for i in range(self.num_negatives):
                neg_premise = [ex["neg_premises"][i] for ex in examples]
                tokenized_neg_premise = self.tokenizer(
                    [p.serialize() for p in neg_premise],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                batch["neg_premises"].append(neg_premise)
                batch["neg_premises_ids"].append(tokenized_neg_premise.input_ids)
                batch["neg_premises_mask"].append(tokenized_neg_premise.attention_mask)

        # Copy the rest of the fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class RerankerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Union[str, Path],
        preds_path: Union[str, Path],
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.preds = {}
        for pred in pickle.load(Path(preds_path).open("rb")):
            ctx = pred["context"]
            self.preds[ctx.path, ctx.theorem_full_name, ctx.state] = pred

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RerankerDataset(
                self.data_path / "train.json",
                self.preds,
                self.max_seq_len,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = RerankerDataset(
                self.data_path / "val.json",
                self.preds,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

        if stage in (None, "test"):
            self.ds_val = RerankerDataset(
                self.data_path / "test.json",
                self.preds,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


if __name__ == "__main__":
    dm = RerankerDataModule(
        data_path="data/lean_bench/random/",
        preds_path="lightning_logs/version_15/predictions.pickle",
        model_name="google/byt5-small",
        batch_size=8,
        eval_batch_size=64,
        max_seq_len=1024,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup("fit")

    for i, data_batch in tqdm(
        enumerate(dm.train_dataloader()), total=len(dm.train_dataloader())
    ):
        if i == 0:
            print(data_batch)

    for i, data_batch in tqdm(enumerate(dm.val_dataloader())):
        if i == 0:
            print(data_batch)

    for i, data_batch in tqdm(enumerate(dm.test_dataloader())):
        if i == 0:
            print(data_batch)
