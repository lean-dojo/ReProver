import re
import sys
import json
import torch
import random
import tempfile
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from lean_dojo import Pos
import pytorch_lightning as pl
from dataclasses import dataclass, field
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from transformers import get_cosine_schedule_with_warmup
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from typing import Optional, List, Union, Dict, Any, Tuple, Generator


Example = Dict[str, Any]
Batch = Dict[str, Any]

MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"
_MARK_REGEX_WITH_SYMBOLS = re.compile(f"{MARK_START_SYMBOL}.+?{MARK_END_SYMBOL}")
_MARK_REGEX_WITHOUT_SYMBOLS = re.compile(
    f"(?<={MARK_START_SYMBOL}).+?(?={MARK_END_SYMBOL})"
)


def find_marks(s: str, include_symbols: bool) -> List[re.Match]:
    """Find all :code:`<a>...</a>` marks in ``s``."""
    if include_symbols:
        return list(_MARK_REGEX_WITH_SYMBOLS.finditer(s))
    else:
        return list(_MARK_REGEX_WITHOUT_SYMBOLS.finditer(s))


def remove_marks(s: str) -> str:
    """Remove all :code:`<a>` and :code:`</a>` from ``s``."""
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")


def is_well_formed(s: str) -> bool:
    return 0 <= s.count(MARK_START_SYMBOL) - s.count(MARK_END_SYMBOL) <= 1 and not any(
        MARK_START_SYMBOL in m.group() or MARK_END_SYMBOL in m.group()
        for m in find_marks(s, include_symbols=False)
    )


def to_path(p: Union[str, Path]) -> Path:
    """Convert ``p`` to a :class:`Path` object."""
    if isinstance(p, Path):
        return p
    else:
        return Path(p)


@dataclass
class Context:
    """Contexts are "queries" in our retrieval setup."""

    path: Path
    theorem_full_name: str
    theorem_pos: Pos
    tactic_prefix: str
    state: str

    def __post_init__(self) -> None:
        assert isinstance(self.path, Path)
        assert isinstance(self.theorem_full_name, str)
        assert isinstance(self.theorem_pos, Pos)
        assert isinstance(self.tactic_prefix, str)
        assert (
            isinstance(self.state, str)
            and "⊢" in self.state
            and MARK_START_SYMBOL not in self.state
            and MARK_END_SYMBOL not in self.state
        )

    def serialize(self) -> str:
        """Serialize the context into a string for Transformers."""
        # TODO: Make sure the goal is not truncated.
        return f"$TACTIC$ = {self.tactic_prefix} $STATE$ = {self.state}"


@dataclass
class Premise:
    """Premises are "documents" in our retrieval setup."""

    path: Path
    """The ``*.lean`` file this premise comes from.
    """

    full_name: str
    """Fully qualified name.
    """

    start: Pos = field(repr=False)
    """Start position of the premise's definition in the ``*.lean`` file.
    """

    end: Pos = field(repr=False, compare=False)
    """End position of the premise's definition in the ``*.lean`` file.
    """

    code: str = field(compare=False)
    """Raw, human-written code for defining the premise.
    """

    def __post_init__(self) -> None:
        assert isinstance(self.path, Path)
        assert isinstance(self.full_name, str)
        assert (
            isinstance(self.start, Pos)
            and isinstance(self.end, Pos)
            and self.start <= self.end
        )
        assert isinstance(self.code, str)

    def serialize(self) -> str:
        """Serialize the premise into a string for Transformers."""
        return self.code

    def __hash__(self) -> int:
        return (
            hash(self.path)
            ^ hash(self.full_name)
            ^ hash(self.start.line_nb)
            ^ hash(self.start.column_nb)
        )


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
            Premise(path, p["full_name"], Pos(*p["start"]), Pos(*p["end"]), p["code"])
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

    def __init__(self, jsonl_path: Union[str, Path]) -> None:
        """Construct a :class:`Corpus` object from a ``corpus.jsonl`` data file."""
        jsonl_path = to_path(jsonl_path)

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

    def __len__(self) -> int:
        return len(self.all_premises)

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
        premise_embeddings: torch.FloatTensor,
        batch_context: List[Context],
        batch_context_emb: torch.Tensor,
        k: int,
    ) -> Tuple[List[List[Premise]], List[List[float]]]:
        """Perform a batch of nearest neighbour search.

        Args:
            batch_path (List[Union[str, Path]]): _description_
            batch_pos (List[Pos]): _description_
            batch_context_emb (torch.Tensor): _description_
            k (int): _description_

        Returns:
            Tuple[List[List[str]], List[List[float]]]: _description_
        """
        similarities = batch_context_emb @ premise_embeddings.t()
        idxs_batch = similarities.argsort(dim=1, descending=True).tolist()
        assert len(batch_context) == len(idxs_batch)
        results = [[] for _ in batch_context]
        scores = [[] for _ in batch_context]

        for j, (ctx, idxs) in enumerate(zip(batch_context, idxs_batch)):
            accessible_premises = set(
                self.iter_accessible_premises(ctx.path, ctx.theorem_pos)
            )
            assert len(accessible_premises) >= k
            for i in idxs:
                p = self.all_premises[i]
                if p in accessible_premises:
                    results[j].append(p)
                    scores[j].append(similarities[j, i].item())
                    if len(results[j]) >= k:
                        break
            else:
                raise ValueError

        return results, scores


_SPACES_REGEX = re.compile(r"\s+", re.DOTALL)


def normalize_spaces(s: str) -> str:
    """Repalce any consecutive block of whitespace characters in ``s`` with a single whitespace."""
    return _SPACES_REGEX.sub(" ", s).strip()


def format_tactic(annot_tac: str, provenances) -> str:
    """USE full names for the all <a>...</a>."""
    annot_tac = normalize_spaces(annot_tac)
    if len(provenances) == 0:
        return annot_tac

    tac = ""
    marks = list(re.finditer(r"<a>(?P<ident>.+?)</a>", annot_tac))

    for i, (m, prov) in enumerate(zip_strict(marks, provenances)):
        last_end = marks[i - 1].end() if i > 0 else 0
        tac += annot_tac[last_end : m.start()] + "<a>" + prov["full_name"] + "</a>"

    tac += annot_tac[marks[-1].end() :]
    return tac


def format_state(s: str) -> str:
    # TODO: Try putting the goal before the context.
    assert "⊢" in s
    m = re.match(r"\d+ goals", s)
    if m is None:
        return s
    else:
        return s[m.end() :].strip()


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def get_optimizers(
    parameters, trainer: pl.Trainer, lr: float, warmup_steps: int
) -> Dict[str, Any]:
    """Return an AdamW optimizer with cosine warmup learning rate schedule."""
    strategy = trainer.strategy

    if isinstance(strategy, DeepSpeedStrategy):
        if "offload_optimizer" in strategy.config["zero_optimization"]:
            logger.info("Optimizing with DeepSpeedCPUAdam")
            optimizer = DeepSpeedCPUAdam(parameters, lr=lr, adamw_mode=True)
        else:
            logger.info("Optimizing with FusedAdam")
            optimizer = FusedAdam(parameters, lr=lr, adam_w_mode=True)
    else:
        logger.info("Optimizing with AdamW")
        optimizer = torch.optim.AdamW(parameters, lr=lr)

    if trainer.max_steps != -1:
        max_steps = trainer.max_steps
    else:
        assert trainer.max_epochs is not None
        max_steps = (
            trainer.max_epochs
            * len(trainer.datamodule.train_dataloader())
            // trainer.accumulate_grad_batches
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }


def _is_deepspeed_checkpoint(path: Path):
    if not path.exists():
        raise FileExistsError(f"Checkpoint {path} does not exist.")
    return path.is_dir() and (path / "zero_to_fp32.py").exists()


def load_checkpoint(model_cls, ckpt_path: Path, device, freeze: bool):
    """Handle DeepSpeed checkpoints in model loading."""
    if not _is_deepspeed_checkpoint(ckpt_path):
        model = model_cls.load_from_checkpoint(ckpt_path, strict=False).to(device)
    else:
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "lightning.cpkt"
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, path)
            model = model_cls.load_from_checkpoint(path, strict=False)
            model = model.to(device)
    if freeze:
        model.freeze()
    return model


def zip_strict(*args):
    assert len(args) > 1 and all(len(args[0]) == len(a) for a in args[1:])
    return zip(*args)


def set_logger(verbose: bool) -> None:
    """
    Set the logging level of loguru.
    The effect of this function is global, and it should
    be called only once in the main function
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
