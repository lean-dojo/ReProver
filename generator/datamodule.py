import os
import pdb
import json
import pickle
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ByT5Tokenizer
from common import (
    format_state,
    format_augmented_state,
    format_tactic,
    remove_marks,
    Example,
    Batch,
)


class GeneratorDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        keep_marks: bool,
        preds: List[Dict[str, Any]],
        max_seq_len: int,
        tokenizer: ByT5Tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.keep_marks = keep_marks
        self.preds = preds
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Example]:
        data = []
        for thm in tqdm(json.load(open(data_path))):
            repo_name = thm["url"].split("/")[-1]
            file_path = os.path.join(repo_name, thm["file_path"])

            for tac in thm["traced_tactics"]:
                state = format_state(tac["state_before"])
                if self.preds is not None:
                    pred = self.preds[(file_path, thm["full_name"], state)]
                    state = format_augmented_state(
                        tac["state_before"], pred["retrieved_premises"]
                    )
                tactic = format_tactic(*tac["annotated_tactic"])
                if not self.keep_marks:
                    tactic = remove_marks(tactic)
                data.append(
                    {
                        "url": thm["url"],
                        "commit": thm["commit"],
                        "file_path": thm["file_path"],
                        "full_name": thm["full_name"],
                        "state": state,
                        "tactic": tactic,
                    }
                )

        logger.info(f"{len(data)} examples loaded")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        return self.data[idx]

    def collate(self, examples: List[Example]) -> Batch:
        state = [ex["state"] for ex in examples]
        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tactic = [ex["tactic"] for ex in examples]
        tokenized_tactic = self.tokenizer(
            tactic,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tactic_ids = tokenized_tactic.input_ids
        tactic_ids[tactic_ids == self.tokenizer.pad_token_id] = -100

        batch = {}
        batch["state"] = state
        batch["state_ids"] = tokenized_state.input_ids
        batch["state_mask"] = tokenized_state.attention_mask
        batch["tactic"] = tactic
        batch["tactic_ids"] = tactic_ids
        batch["tactic_mask"] = tokenized_tactic.attention_mask

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
        preds_path: Optional[str],
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.keep_marks = keep_marks
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
                os.path.join(self.data_path, "train.json"),
                self.keep_marks,
                self.preds,
                self.max_seq_len,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = GeneratorDataset(
                os.path.join(self.data_path, "val.json"),
                self.keep_marks,
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
    dm = GeneratorDataModule(
        data_path="data/lean_bench/random/",
        keep_marks=True,
        preds_path="lightning_logs/version_31/predictions.pickle",
        model_name="google/byt5-small",
        batch_size=8,
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
        print("state: ", data_batch["state_ids"].size())
        # if data_batch["tactic_ids"].size(1) > 256:
        #    print("tactic: ", data_batch["tactic_ids"].size())

    for i, data_batch in tqdm(enumerate(dm.val_dataloader())):
        if i == 0:
            print(data_batch)

    for i, data_batch in tqdm(enumerate(dm.test_dataloader())):
        if i == 0:
            print(data_batch)
