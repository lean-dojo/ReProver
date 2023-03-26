import re
import pdb
import json
import torch
import random
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from lean_dojo import Pos
import pytorch_lightning as pl
from dataclasses import dataclass, field
from transformers import ByT5Tokenizer
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Union, Dict, Any, Tuple, Generator


@dataclass(frozen=True)
class Premise:
    """Premises are "documents" in our retrieval setup."""

    path: Path
    """The ``*.lean`` file this premise comes from.
    """

    full_name: str
    """Fully qualified name.
    """

    code: str = field(compare=False)
    """Raw, human-written code for defining the premise.
    """

    start: Pos = field(repr=False, compare=False)
    """Start position of the premise's definition in the ``*.lean`` file.
    """

    end: Pos = field(repr=False, compare=False)
    """End position of the premise's definition in the ``*.lean`` file.
    """

    def serialize(self) -> str:
        """Serialize the premise into a string for Transformers."""
        return f"$NAME$ = {self.full_name} $CODE$ = {self.code}"


@dataclass(frozen=True)
class File:
    """A file defines 0 or multiple premises."""

    path: Path
    """Path of the ``*.lean`` file.
    """

    premises: List[Premise]
    """A list of premises defined in this file.
    """

    @classmethod
    def from_data(cls, file_data: Dict[str, Any]) -> "File":
        """Construct a :class:`File` object from ``file_data``."""
        path = Path(file_data["path"])
        premises = [
            Premise(path, p["full_name"], p["code"], Pos(*p["start"]), Pos(*p["end"]))
            for p in file_data["premises"]
            if "user__.n" not in p["full_name"]
        ]
        return cls(path, premises)

    @property
    def is_empty(self) -> bool:
        """Check whether the file contains no premise."""
        return self.premises == []


class Corpus:
    """Our retrieval corpus is a DAG of files. Each file consists of
    premises (theorems, definitoins, etc.) that can be retrieved.
    """

    transitive_dep_graph: nx.DiGraph
    """Transitive closure of the dependency graph among files. 
    There is an edge from file X to Y iff X import Y (directly or indirectly).
    """

    all_premises: List[Premise]
    """All premises in the entire corpus.
    """

    premise_embeddings: Optional[torch.Tensor] = None
    """Vector embeddings of all premises produced by some machine learning model.
    """

    def __init__(self, jsonl_path: Union[str, Path]) -> None:
        """Construct a :class:`Corpus` object from a ``corpus.jsonl`` data file."""
        if isinstance(jsonl_path, str):
            jsonl_path = Path(jsonl_path)

        dep_graph = nx.DiGraph()
        self.all_premises = []

        logger.info(f"Building the corpus from {jsonl_path}")
        lines = list(jsonl_path.open())

        for line in tqdm(lines):
            file_data = json.loads(line)
            path = Path(file_data["path"])
            assert not dep_graph.has_node(path)
            file = File.from_data(file_data)

            dep_graph.add_node(path, file=file)
            self.all_premises.extend(file.premises)

            for p in file_data["imports"]:
                p = Path(p)
                assert dep_graph.has_node(p)
                dep_graph.add_edge(path, p)

        assert nx.is_directed_acyclic_graph(dep_graph)
        self.transitive_dep_graph = nx.transitive_closure_dag(dep_graph)

    def _get_file(self, path: Path) -> File:
        return self.transitive_dep_graph.nodes[path]["file"]

    @property
    def files(self) -> List[File]:
        return [self._get_file(p) for p in self.transitive_dep_graph.dep_graph.nodes]

    def get_dependencies(self, path: Union[str, Path]) -> List[Path]:
        """Return a list of (direct and indirect) dependencies of the file ``path``."""
        if isinstance(path, str):
            path = Path(path)
        return list(self.transitive_dep_graph.successors(path))

    def get_premises(self, path: Union[str, Path]) -> List[Premise]:
        """Return a list of premises defined in the file ``path``."""
        if isinstance(path, str):
            path = Path(path)
        return self._get_file(path).premises

    def num_premises(self, path: Union[str, Path]) -> int:
        """Return the number of premises defined in the file ``path``."""
        return len(self.get_premises(path))

    def locate_premise(self, path: Union[str, Path], pos: Pos) -> Optional[Premise]:
        """Return a premise at position ``pos`` in file ``path``.

        Return None if no such premise can be found.
        """
        if isinstance(path, str):
            path = Path(path)

        for p in self.get_premises(path):
            assert p.path == path
            if p.start <= pos <= p.end:
                return p

        return None

    def iter_accessible_premises(
        self, path: Union[str, Path], pos: Pos
    ) -> Generator[Premise, None, None]:
        """Return an iterator of premises accessible at position ``pos`` in file ``path``,
        i.e., all premises defined in the (transitively) imported files or earlier in the same file.
        """
        if isinstance(path, str):
            path = Path(path)
        for p in self.get_premises(path):
            if p.end < pos:
                yield p
        for p in self.transitive_dep_graph.successors(path):
            yield from self._get_file(p).premises

    def get_accessible_premise_indexes(
        self, path: Union[str, Path], pos: Pos
    ) -> List[int]:
        if isinstance(path, str):
            path = Path(path)
        return {
            i
            for i, prem in enumerate(self.all_premises)
            if (prem.path == path and prem.end < pos)
            or self.transitive_dep_graph.has_edge(path, prem.path)
        }

    def get_nearest_premises(
        self,
        batch_path: List[Union[str, Path]],
        batch_pos: List[Pos],
        batch_context_emb: torch.Tensor,
        k: int,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """Perform a batch of nearest neighbour search.

        Args:
            batch_path (List[Union[str, Path]]): _description_
            batch_pos (List[Pos]): _description_
            batch_context_emb (torch.Tensor): _description_
            k (int): _description_

        Returns:
            Tuple[List[List[str]], List[List[float]]]: _description_
        """
        assert self.premise_embeddings is not None
        similarities = batch_context_emb @ self.premise_embeddings.t()
        idxs_batch = similarities.argsort(dim=1, descending=True)
        assert len(batch_path) == len(batch_pos) == len(idxs_batch)
        results = [[] for _ in batch_path]
        scores = [[] for _ in batch_path]

        for j, (path, pos, idxs) in enumerate(zip(batch_path, batch_pos, idxs_batch)):
            accessible_premises = set(self.iter_accessible_premises(path, pos))
            assert len(accessible_premises) >= k
            for i in idxs:
                p = self.all_premises[i]
                if p in accessible_premises:
                    results[j].append(p.serialize())
                    scores[j].append(similarities[j, i].item())
                    if len(results[j]) >= k:
                        break
            else:
                raise ValueError

        return results, scores

    def update_embeddings(self, encoder) -> None:
        self.premise_embeddings = encoder(self.all_premises)

    def to(self, device):
        self.premise_embeddings = self.premise_embeddings.to(device)

    def cpu(self):
        self.premise_embeddings = self.premise_embeddings.cpu()


@dataclass(frozen=True)
class Context:
    path: Path
    theorem_full_name: str
    theorem_pos: Pos
    tactic_prefix: str
    state: str

    def serialize(self) -> str:
        """Serialize the context into a string for Transformers."""
        # TODO: Do file names and theorem names actually help?
        return f"$THEOREM$ = {self.theorem_full_name} $TACTIC$ = {self.tactic_prefix} $STATE$ = {self.state}"


class RetrievalDataset(Dataset):  # type: ignore
    def __init__(
        self,
        data_path: Path,
        corpus: Corpus,
        num_negatives: int,
        max_seq_len: int,
        tokenizer: str,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.num_negatives = num_negatives
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.is_train = is_train

        self.data = []
        num_discarded = 0
        logger.info(f"Loading data from {data_path}")

        for thm in tqdm(json.load(data_path.open())):
            repo_name = thm["url"].split("/")[-1]
            file_path = Path(repo_name) / thm["file_path"]
            deps = self.corpus.get_dependencies(file_path)

            for tac in thm["traced_tactics"]:
                annot_tac, provenances = tac["annotated_tactic"]
                marks = list(re.finditer(r"(?<=<a>).+?(?=</a>)", annot_tac))

                for m, prov in zip(marks, provenances):
                    def_path = Path(prov["def_path"])
                    assert def_path == file_path or def_path in deps
                    pos_premise = self.corpus.locate_premise(
                        def_path, Pos(*prov["def_pos"])
                    )
                    if pos_premise is None:
                        num_discarded += 1
                    else:
                        tactic_prefix = annot_tac[: m.start()]
                        context = Context(
                            file_path,
                            thm["full_name"],
                            Pos(*thm["start"]),
                            tactic_prefix,
                            tac["state_before"],
                        )
                        self.data.append(
                            {
                                "context": context,
                                "tactic_arg": m.group(),
                                "pos_premise": pos_premise,
                            }
                        )

        logger.info(
            f"{len(self.data)} examples remain after discarding {num_discarded} examples"
        )
        assert num_discarded / len(self.data) < 0.01

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        ex = self.data[idx]
        context = ex["context"]
        pos_premise = ex["pos_premise"]
        item = {
            "path": context.path,
            "pos": context.theorem_pos,
            "context": context.serialize(),
            "pos_premise": pos_premise.serialize(),
        }

        if self.is_train:
            premises = [
                p
                for p in self.corpus.iter_accessible_premises(
                    context.path, context.theorem_pos
                )
                if p != pos_premise
            ]
            neg_premises = random.sample(premises, self.num_negatives)
            item["neg_premises"] = [p.serialize() for p in neg_premises]

        return item

    def collate(self, examples):
        batch = {}

        c = [ex["context"] for ex in examples]
        context = self.tokenizer(
            c,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        pd = [ex["pos_premise"] for ex in examples]
        pos_premise = self.tokenizer(
            pd,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        batch["context"] = c
        batch["context_ids"] = context.input_ids
        batch["context_mask"] = context.attention_mask
        batch["pos_premise"] = pd
        batch["pos_premise_ids"] = pos_premise.input_ids
        batch["pos_premise_mask"] = pos_premise.attention_mask

        if self.is_train:
            batch["neg_premises_ids"] = []
            batch["neg_premises_mask"] = []

            batch_size = len(examples)
            label = torch.zeros(batch_size, batch_size * (1 + self.num_negatives))
            # Check if one's negative is another's positive
            for j in range(batch_size):
                pos_premise = examples[j]["pos_premise"]
                for k in range(batch_size * (1 + self.num_negatives)):
                    if k < batch_size:
                        label[j, k] = float(pos_premise == examples[k]["pos_premise"])
                    else:
                        label[j, k] = float(
                            pos_premise
                            == examples[k % batch_size]["neg_premises"][
                                k // batch_size - 1
                            ]
                        )

            for i in range(self.num_negatives):
                neg_premise = self.tokenizer(
                    [ex["neg_premises"][i] for ex in examples],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                batch["neg_premises_ids"].append(neg_premise.input_ids)
                batch["neg_premises_mask"].append(neg_premise.attention_mask)

            batch["label"] = label

        for k in examples[0].keys():
            if k not in ("context", "pos_premise", "neg_premises"):
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
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
        self.corpus = Corpus(corpus_path)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RetrievalDataset(
                self.data_path / "train.json",
                self.corpus,
                self.num_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = RetrievalDataset(
                self.data_path / "val.json",
                self.corpus,
                self.num_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

        if stage in (None, "test"):
            self.ds_test = RetrievalDataset(
                self.data_path / "test.json",
                self.corpus,
                self.num_negatives,
                self.max_seq_len,
                self.tokenizer,
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
        num_negatives=3,
        model_name="google/byt5-small",
        batch_size=8,
        max_seq_len=1024,
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
        # max_len = max(max_len, data_batch["pos_premise_ids"].size(1))
        # print("pos: ", data_batch["pos_premise_ids"].size())
        # for ids in data_batch["negative_premises_ids"]:
        #    max_len = max(max_len, ids.size(1))
        #    print("neg:", ids.size())

    for i, data_batch in tqdm(enumerate(dm.val_dataloader())):
        if i == 0:
            print(data_batch)

    for i, data_batch in tqdm(enumerate(dm.test_dataloader())):
        if i == 0:
            print(data_batch)

    # print(max_len)
