import pdb
import os
import json
import torch
import pickle
import random
import itertools
from tqdm import tqdm
from copy import deepcopy
from loguru import logger
import pytorch_lightning as pl
from transformers import AutoTokenizer
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset, DataLoader


from common import format_state, Example, Batch, MARK_START_SYMBOL


class RerankerDataset(Dataset):
    def __init__(
        self,
        data_paths: str,
        preds: List[Dict[str, Any]],
        num_retrieved: int,
        num_negatives: int,
        max_seq_len: int,
        tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.data_paths = data_paths
        self.preds = preds
        self.num_retrieved = num_retrieved
        self.num_negatives = num_negatives
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.reload_data()

    def reload_data(self) -> None:
        self.data = list(
            itertools.chain.from_iterable(
                self._load_data(path, self.is_train) for path in self.data_paths
            )
        )

    def _load_data(self, data_path: str, is_train: bool) -> List[Example]:
        data = []
        logger.info(f"Loading data from {data_path}")

        for thm in tqdm(json.load(open(data_path))):
            repo_name = thm["url"].split("/")[-1]
            file_path = os.path.join(repo_name, thm["file_path"])

            for tac in thm["traced_tactics"]:
                state = format_state(tac["state_before"])
                pred = self.preds[(file_path, thm["full_name"], state)]
                retrieved_premises = pred["retrieved_premises"][: self.num_retrieved]
                all_pos_premises = set(pred["all_pos_premises"])

                if is_train:
                    for premise in all_pos_premises:
                        data.append({"state": state, "premise": premise, "label": True})
                    if len(all_pos_premises) == 0:
                        continue
                    if not all_pos_premises.issubset(retrieved_premises):
                        neg_premises = [
                            p for p in retrieved_premises if p not in all_pos_premises
                        ]
                    else:
                        last_idx = -1
                        for i, p in enumerate(retrieved_premises):
                            if p in all_pos_premises:
                                last_idx = i
                                break
                        last_idx = max(
                            last_idx, len(all_pos_premises) + self.num_negatives
                        )
                        neg_premises = [
                            p
                            for p in retrieved_premises[:last_idx]
                            if p not in all_pos_premises
                        ]
                    for p in random.sample(neg_premises, self.num_negatives):
                        data.append({"state": state, "premise": p, "label": False})
                else:
                    data.append(
                        {
                            "state": state,
                            "all_pos_premises": all_pos_premises,
                            "retrieved_premises": retrieved_premises,
                        }
                    )

        if is_train:
            num_examples = len(data)
            num_positives = sum(ex["label"] for ex in data)
            logger.info(
                f"{num_examples} training examples loaded, including {num_positives} positive examples"
            )

        return data

    def __len__(self) -> int:
        return len(self.data)

    def _format_seq(self, state: str, premise: str) -> str:
        return f"$PREMISE$ {premise.serialize()} $STATE$ {state}"

    def __getitem__(self, idx: int) -> Example:
        ex = deepcopy(self.data[idx])
        if self.is_train:
            ex["seq"] = self._format_seq(ex["state"], ex["premise"])
        else:
            ex["seqs"] = [
                self._format_seq(ex["state"], p) for p in ex["retrieved_premises"]
            ]
        return ex

    def collate(self, examples: List[Example]) -> Batch:
        batch = {}

        if self.is_train:
            tokenized_seq = self.tokenizer(
                [ex["seq"] for ex in examples],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )

            batch["seq_ids"] = tokenized_seq.input_ids
            batch["seq_mask"] = tokenized_seq.attention_mask
            batch["label"] = torch.tensor(
                [ex["label"] for ex in examples], dtype=torch.float32
            )
        else:
            num_retrieved = len(examples[0]["seqs"])
            for i in range(num_retrieved):
                tokenized_seq = self.tokenizer(
                    [ex["seqs"][i] for ex in examples],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                batch[f"seq_{i}_ids"] = tokenized_seq.input_ids
                batch[f"seq_{i}_mask"] = tokenized_seq.attention_mask

        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class RerankerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        preds_path: str,
        num_retrieved: int,
        num_negatives: int,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_retrieved = num_retrieved
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.preds = {}
        for pred in pickle.load(open(preds_path, "rb")):
            ctx = pred["context"]
            self.preds[ctx.path, ctx.theorem_full_name, ctx.state] = pred

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RerankerDataset(
                [os.path.join(self.data_path, "train.json")],
                self.preds,
                self.num_retrieved,
                self.num_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = RerankerDataset(
                [os.path.join(self.data_path, "val.json")],
                self.preds,
                self.num_retrieved,
                self.num_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

        if stage in (None, "test"):
            self.ds_val = RerankerDataset(
                [os.path.join(self.data_path, "test.json")],
                self.preds,
                self.num_retrieved,
                self.num_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

        if stage in (None, "fit", "predict"):
            self.ds_pred = RerankerDataset(
                [
                    os.path.join(self.data_path, f"{split}.json")
                    for split in ("train", "val", "test")
                ],
                self.preds,
                self.num_retrieved,
                self.num_negatives,
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

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_pred,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_pred.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


if __name__ == "__main__":
    dm = RerankerDataModule(
        data_path="data/lean_bench/random/",
        preds_path="lightning_logs/version_28/predictions.pickle",
        model_name="google/byt5-small",
        batch_size=8,
        eval_batch_size=64,
        max_seq_len=2048,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup("fit")

    for i, data_batch in tqdm(
        enumerate(dm.train_dataloader()), total=len(dm.train_dataloader())
    ):
        if i == 0:
            print(data_batch)
        print(data_batch["seq_ids"].shape)

    for i, data_batch in tqdm(enumerate(dm.val_dataloader())):
        if i == 0:
            print(data_batch)

    for i, data_batch in tqdm(enumerate(dm.test_dataloader())):
        if i == 0:
            print(data_batch)
