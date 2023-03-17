import os
import re
import pdb
import random
import torch
import datasets
import json
import networkx as nx
from loguru import logger
from tqdm import tqdm
import unicodedata
from copy import deepcopy
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Union
from transformers import AutoTokenizer


def _format_query(ex):
    # TODO: Remove <a>
    # TODO: Can also include the partial proof?
    return f"$FILE$ = {ex['file']} $THEOREM$ = {ex['theorem']} $TACTIC$ = {ex['tactic_prefix']} $STATE$ = {ex['state']}"


def _format_doc(d):
    return f"$FILE$ = {d['file']} $NAME$ = {d['full_name']} $CODE$ = {d['code']}"


def _normalize(text: str) -> str:
    """Deal with unicode-related artifacts."""
    return unicodedata.normalize("NFD", text)


def _lt(x: List[int], y: List[int]) -> bool:
    x_line, x_col = x
    y_line, y_col = y
    return x_line < y_line or (x_line == y_line and x_col < y_col)


def _le(x: List[int], y: List[int]) -> bool:
    x_line, x_col = x
    y_line, y_col = y
    return x_line < y_line or (x_line == y_line and x_col <= y_col)


def _between(x: List[int], y: List[int], z: List[int]) -> bool:
    return _le(x, y) and _le(y, z)


class Corpus:
    dep_graph: nx.DiGraph
    all_premises: datasets.Dataset

    def __init__(self, jsonl_path: Path):
        self.dep_graph = nx.DiGraph()
        all_premises = []

        for line in jsonl_path.open():
            data = json.loads(line)
            path = data["path"]
            assert not self.dep_graph.has_node(path)
            premises = data["premises"]
            for prem in premises:
                all_premises.append(
                    _format_doc(
                        {
                            "file": path,
                            "full_name": prem["full_name"],
                            "code": prem["code"],
                        }
                    )
                )
            self.dep_graph.add_node(path, premises=premises)
            for p in data["imports"]:
                assert self.dep_graph.has_node(p)
                self.dep_graph.add_edge(path, p)
            nx.is_directed_acyclic_graph(self.dep_graph)

        self.all_premises = datasets.Dataset.from_dict({"doc": all_premises})

    def get_premises(self, path):
        return [
            p
            for p in self.dep_graph.nodes[path]["premises"]
            if not p["full_name"].startswith("user__.")
        ]

    def num_premises(self, path):
        return len(self.get_premises(path))

    def get_positive_premise(self, full_name: str, def_path: Path, def_pos: List[int]):
        potential_premises = self.get_premises(def_path)
        for p in potential_premises:
            if _between(p["start"], def_pos, p["end"]):
                return {
                    "file": def_path,
                    "full_name": p["full_name"],
                    "code": p["code"],
                }
        logger.warning(f"Unable to find {full_name} in {def_path} at {def_pos}")
        return None

    def sample_negative_premises(
        self, num_negatives: int, def_path: Path, def_pos: List[int]
    ):
        accessible_files = list(self.dep_graph.nodes)

        # TODO: Alternative strategies include hard negatives, sample premises in the current file, etc.
        # accessible_files = nx.descendants(self.dep_graph, def_path)
        # same_file_premises = [p for p in self.get_premises(def_path) if _lt(p["end"], def_pos)]
        # if same_file_premises != []:
        #    accessible_files.add(def_path)
        # accessible_files = list(accessible_files)
        nums = [self.num_premises(p) for p in accessible_files]
        # Sample with replacement.
        sampled_files = random.choices(accessible_files, weights=nums, k=num_negatives)
        negs = []
        for path in sampled_files:
            premises = self.get_premises(path)
            # if path == def_path:
            #    premises = same_file_premises
            p = random.choice(premises)
            negs.append(
                {
                    "file": path,
                    "full_name": p["full_name"],
                    "code": p["code"],
                }
            )

        return negs


class RetrievalDataset(Dataset):  # type: ignore
    def __init__(
        self,
        data_path: Path,
        corpus: Corpus,
        num_negatives: int,
        model_name: str,
        max_seq_len: int,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.num_negatives = num_negatives
        self.max_seq_len = max_seq_len
        self.is_train = is_train

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.data = []
        num_discarded = 0
        logger.info(f"Loading data from {data_path}")

        for thm in json.load(data_path.open()):
            repo_name = thm["url"].split("/")[-1]
            file_path = os.path.join(repo_name, thm["file_path"])
            deps = nx.descendants(self.corpus.dep_graph, file_path)

            for tac in thm["traced_tactics"]:
                annot_tac, provenances = tac["annotated_tactic"]
                marks = list(re.finditer(r"(?<=<a>).+?(?=</a>)", annot_tac))
                for m, prov in zip(marks, provenances):
                    if prov["def_path"] != file_path and prov["def_path"] not in deps:
                        pdb.set_trace()
                    assert prov["def_path"] == file_path or prov["def_path"] in deps
                    positive_doc = self.corpus.get_positive_premise(**prov)
                    if positive_doc is None:
                        num_discarded += 1
                    else:
                        self.data.append(
                            {
                                "file": file_path,
                                "theorem": thm["full_name"],
                                "pos": thm["start"],
                                "state": tac["state_before"],
                                "tactic_prefix": annot_tac[: m.start()],
                                "tactic_arg": m.group(),
                                "positive_doc": positive_doc,
                            }
                        )

        logger.info(
            f"{len(self.data)} examples remain after discarding {num_discarded} examples"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        ex = deepcopy(self.data[idx])
        ex["query"] = _format_query(ex)
        positive_doc = ex["positive_doc"]
        ex["positive_doc"] = _format_doc(positive_doc)

        if self.is_train:
            negative_docs = self.corpus.sample_negative_premises(
                self.num_negatives, ex["file"], ex["pos"]
            )
            ex["negative_docs"] = [_format_doc(d) for d in negative_docs]

        return ex

    def collate(self, examples):
        batch = {}

        q = [ex["query"] for ex in examples]
        query = self.tokenizer(
            q,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        pd = [ex["positive_doc"] for ex in examples]
        positive_doc = self.tokenizer(
            pd,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        batch["query"] = q
        batch["query_ids"] = query.input_ids
        batch["query_mask"] = query.attention_mask
        batch["positive_doc"] = pd
        batch["positive_doc_ids"] = positive_doc.input_ids
        batch["positive_doc_mask"] = positive_doc.attention_mask

        if self.is_train:
            batch["negative_docs_ids"] = []
            batch["negative_docs_mask"] = []

            batch_size = len(examples)
            label = torch.zeros(batch_size, batch_size * (1 + self.num_negatives))
            for j in range(batch_size):
                label[j, j] = 1.0
            # TODO: Check if one's negative is another's positive

            for i in range(self.num_negatives):
                neg_doc = self.tokenizer(
                    [ex["negative_docs"][i] for ex in examples],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                batch["negative_docs_ids"].append(neg_doc.input_ids)
                batch["negative_docs_mask"].append(neg_doc.attention_mask)

            batch["label"] = label

        for k in examples[0].keys():
            if k not in ("query", "positive_doc", "negative_docs"):
                batch[k] = [ex[k] for ex in examples]

        return batch


class RetrievalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Union[str, Path],
        corpus_path: Union[str, Path],
        num_negatives: int,
        model_name: str,
        batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if isinstance(corpus_path, str):
            corpus_path = Path(corpus_path)

        self.data_path = data_path
        self.num_negatives = num_negatives
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.corpus = Corpus(corpus_path)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RetrievalDataset(
                self.data_path / "train.json",
                self.corpus,
                self.num_negatives,
                self.model_name,
                self.max_seq_len,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = RetrievalDataset(
                self.data_path / "val.json",
                self.corpus,
                self.num_negatives,
                self.model_name,
                self.max_seq_len,
                is_train=False,
            )

        if stage in (None, "test"):
            self.ds_test = RetrievalDataset(
                self.data_path / "test.json",
                self.corpus,
                self.num_negatives,
                self.model_name,
                self.max_seq_len,
                is_train=False,
            )

    def train_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_train,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_val,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_test,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_test.collate,
            pin_memory=True,
            drop_last=False,
        )


if __name__ == "__main__":
    dm = RetrievalDataModule(
        data_path="data/lean_bench/random/",
        corpus_path="data/lean_bench/corpus.jsonl",
        num_negatives=10,
        model_name="google/byt5-small",
        batch_size=32,
        max_seq_len=2048,
        num_workers=8,
    )
    dm.prepare_data()
    dm.setup("fit")

    # max_len = 0
    for i, data_batch in tqdm(
        enumerate(dm.train_dataloader()), total=len(dm.train_dataloader())
    ):
        if i == 0:
            print(data_batch)
        # max_len = max(max_len, data_batch["positive_doc_ids"].size(1))
        # print("pos: ", data_batch["positive_doc_ids"].size())
        # for ids in data_batch["negative_docs_ids"]:
        #    max_len = max(max_len, ids.size(1))
        #    print("neg:", ids.size())

    for i, data_batch in tqdm(enumerate(dm.val_dataloader())):
        if i == 0:
            print(data_batch)

    for i, data_batch in tqdm(enumerate(dm.test_dataloader())):
        if i == 0:
            print(data_batch)

    # print(max_len)
