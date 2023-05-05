import os
import json
import torch
import random
import itertools
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from lean_dojo import Pos
import pytorch_lightning as pl
from typing import Optional, List
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ByT5Tokenizer


from common import (
    Context,
    Corpus,
    format_state,
    Example,
    Batch,
)


class RetrievalDataset(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        corpus: Corpus,
        num_negatives: int,
        num_in_file_negatives: int,
        accessible_premises_only: bool,
        max_seq_len: int,
        tokenizer: ByT5Tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.accessible_premises_only = accessible_premises_only
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = list(
            itertools.chain.from_iterable(self._load_data(path) for path in data_paths)
        )

    def _load_data(self, data_path: str) -> List[Example]:
        data = []
        num_discarded = 0
        logger.info(f"Loading data from {data_path}")

        for thm in tqdm(json.load(open(data_path))):
            repo_name = thm["url"].split("/")[-1]
            file_path = os.path.join(repo_name, thm["file_path"])
            deps = self.corpus.get_dependencies(file_path)

            for tac in thm["traced_tactics"]:
                _, provenances = tac["annotated_tactic"]
                all_pos_premises = set()

                for prov in provenances:
                    def_path = prov["def_path"]
                    assert def_path == file_path or def_path in deps
                    p = self.corpus.locate_premise(def_path, Pos(*prov["def_pos"]))
                    if p is None:  # Cannot find the premise.
                        num_discarded += 1
                    else:
                        all_pos_premises.add(p)

                if self.is_train and len(all_pos_premises) == 0:
                    continue

                all_pos_premises = list(all_pos_premises)
                state = format_state(tac["state_before"])
                context = Context(
                    file_path, thm["full_name"], Pos(*thm["start"]), state
                )

                if not self.is_train:
                    data.append(
                        {"context": context, "all_pos_premises": all_pos_premises}
                    )
                else:
                    for pos_premise in all_pos_premises:
                        data.append(
                            {
                                "context": context,
                                "pos_premise": pos_premise,
                                "all_pos_premises": all_pos_premises,
                            }
                        )

        logger.info(
            f"{len(data)} examples remain after discarding {num_discarded} examples."
        )
        assert num_discarded / len(data) < 0.01
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        if not self.is_train:
            return self.data[idx]

        # Sample negative premises from all accessible premises.
        ex = deepcopy(self.data[idx])
        premises_in_file = []
        premises_outside_file = []

        for p in self.corpus.get_premises(ex["context"].path):
            if p == ex["pos_premise"]:
                continue
            if p.end < ex["context"].theorem_pos:
                if ex["pos_premise"].path == ex["context"].path:
                    premises_in_file.append(p)
                else:
                    premises_outside_file.append(p)

        for p in self.corpus.transitive_dep_graph.successors(ex["context"].path):
            if p == ex["pos_premise"].path:
                premises_in_file += [
                    _p for _p in self.corpus.get_premises(p) if _p != ex["pos_premise"]
                ]
            else:
                premises_outside_file += self.corpus.get_premises(p)

        if not self.accessible_premises_only:
            premises_outside_file = self.corpus.all_premises

        num_in_file_negatives = min(len(premises_in_file), self.num_in_file_negatives)

        ex["neg_premises"] = random.sample(
            premises_in_file, num_in_file_negatives
        ) + random.sample(
            premises_outside_file, self.num_negatives - num_in_file_negatives
        )
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


class RetrievalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        corpus_path: str,
        num_negatives: int,
        num_in_file_negatives: int,
        accessible_premises_only: bool,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_negatives = num_negatives
        assert 0 <= num_in_file_negatives <= num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.accessible_premises_only = accessible_premises_only
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.corpus = Corpus(corpus_path)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RetrievalDataset(
                [os.path.join(self.data_path, "train.json")],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.accessible_premises_only,
                self.max_seq_len,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = RetrievalDataset(
                [os.path.join(self.data_path, "val.json")],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.accessible_premises_only,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

        if stage in (None, "fit", "predict"):
            self.ds_pred = RetrievalDataset(
                [
                    os.path.join(self.data_path, f"{split}.json")
                    for split in ("train", "val", "test")
                ],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.accessible_premises_only,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.eval_batch_size,
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
    dm = RetrievalDataModule(
        data_path="data/lean_bench/random/",
        corpus_path="data/lean_bench/corpus.jsonl",
        num_negatives=3,
        model_name="google/byt5-small",
        batch_size=8,
        eval_batch_size=64,
        max_seq_len=1024,
        num_workers=8,
    )
    dm.prepare_data()
    dm.setup("fit")

    for i, data_batch in tqdm(
        enumerate(dm.train_dataloader()), total=len(dm.train_dataloader())
    ):
        if i == 0:
            print(data_batch)
        print("context: ", data_batch["context_ids"].size())
        print("pos: ", data_batch["pos_premise_ids"].size())
        for ids in data_batch["neg_premises_ids"]:
            print("neg:", ids.size())

    for i, data_batch in tqdm(enumerate(dm.val_dataloader())):
        if i == 0:
            print(data_batch)

    for i, data_batch in tqdm(enumerate(dm.test_dataloader())):
        if i == 0:
            print(data_batch)
