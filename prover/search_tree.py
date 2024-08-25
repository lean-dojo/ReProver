"""Definitions of the search tree used by the prover.
"""

import math
from enum import Enum
from lean_dojo import (
    TacticState,
    LeanError,
    TimeoutError,
    ProofGivenUp,
    ProofFinished,
)
from abc import ABC, abstractmethod
from functools import total_ordering
from dataclasses import dataclass, field
from typing import Optional, List, Iterable, Union


class Status(Enum):
    """Status of a node or a proof search."""

    PROVED = "Proved"  # This node (or search) has at least one known proof.
    FAILED = "Failed"  # This node (or search) has exhausted its options and cannot be proved within the current run.
    OPEN = "Open"  # This node (or search) has not been proven or given up on yet.


class Node(ABC):
    @property
    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError

    @property
    @abstractmethod
    def distance_to_proof(self) -> int:
        "The smallest number of steps to a proof."
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError


@dataclass
class ProofFinishedNode(Node):
    inner: ProofFinished
    status = Status.PROVED
    distance_to_proof = 0
    is_terminal = True


@dataclass
class ErrorNode(Node):
    inner: Union[LeanError, TimeoutError, ProofGivenUp]
    status = Status.FAILED
    distance_to_proof = math.inf
    is_terminal = True


@total_ordering
@dataclass(unsafe_hash=True)
class InternalNode(Node):
    """
    An internal node in the search tree, representing a nonterminal state.

    Nodes are sorted by _inverse_ priority, for compatibility with the `heapq` library.
    That is, node_a < node_b is true if node_a has _higher_ priority than node_b.
    """

    # Goal state this node represents. Two nodes are considered equal if their states
    # are equal; this is the only hashed field and must not be changed.
    state: TacticState = field(compare=True)

    # The sum of action logprobs along edges from the root to this node
    cumulative_logprob: float = field(compare=False, repr=False)

    # All edges known to lead to this node.
    # May change at any time as other nodes are explored.
    in_edges: List["Edge"] = field(
        default_factory=list, init=False, compare=False, repr=False
    )

    out_edges: List["Edge"] = field(
        default_factory=list, init=False, compare=False, repr=False
    )

    status: Status = field(default=Status.OPEN, init=False, compare=False, repr=True)

    is_terminal = False  # type: ignore[override]

    # Number of steps separating this node from the end of a proof along the
    # optimal path. If unproved, infinity. Updated as needed by children.
    _distance_to_proof: float = field(
        default=math.inf, init=False, compare=False, repr=False
    )

    def add_out_edge(self, e: "Edge", max_num_edges: int) -> None:
        assert e.src is self and e not in self.out_edges
        self.out_edges.append(e)
        self._recompute_status(max_num_edges)
        self._recompute_distance_to_proof()

    # A node is considered explored if we've evaluated the actor in the node to generate
    # a list of candidate children. Explored nodes are never re-searched.
    @property
    def is_explored(self) -> bool:
        return self.out_edges is not None

    def _recompute_status(self, max_num_edges: int):
        """
        Recursively update the status of the current node and its ancestors.
        """
        assert self.is_explored and self.out_edges is not None

        # If this node is proved or failed, nothing can change that
        if self.status != Status.OPEN:
            return

        # If any child is proved, this node is proved, and so are parents recursively
        if any(edge.dst.status == Status.PROVED for edge in self.out_edges):
            self.status = Status.PROVED

        # If all children failed, this node is failed. This may fail some parents too.
        if len(self.out_edges) == max_num_edges and all(
            edge.dst.status == Status.FAILED for edge in self.out_edges
        ):
            self.status = Status.FAILED

        # If this node was proved or failed, parents may need to recompute.
        # This is guaranteed to terminate because only open nodes can change, and
        # there are a finite number of open nodes in the tree.
        if self.status != Status.OPEN:
            for edge in self.in_edges:
                edge.src._recompute_status()

    @property
    def distance_to_proof(self) -> float:
        return self._distance_to_proof

    def _recompute_distance_to_proof(self):
        """
        Recursively update the distance_to_proof of the current node and its ancestors.
        """
        if self.out_edges:
            distance = min(edge.distance_to_proof() for edge in self.out_edges)
        else:
            distance = math.inf

        if distance < self._distance_to_proof:
            self._distance_to_proof = distance
            for edge in self.in_edges:
                edge.src._recompute_distance_to_proof()

    # NOTE: Nodes are compared by _negative_ priority, to make heapq act as a max-priority-queue.
    @property
    def priority(self) -> float:
        return self.cumulative_logprob

    def __lt__(self, other: "InternalNode") -> bool:
        return self.priority > other.priority

    def extract_proof(self) -> Optional[List["Edge"]]:
        """
        Extract a proof of the current node as a sequence of edges.
        """
        if self.status != Status.PROVED:
            return None
        assert self.is_explored

        proving_edge = min(
            self.out_edges,
            key=Edge.distance_to_proof,
        )

        if proving_edge.dst.is_terminal:
            # Base case: this edge is all that's required to finish the proof
            assert isinstance(proving_edge.dst, ProofFinishedNode)
            return [proving_edge]
        else:
            # Recursive case: prove the child, then add this edge
            assert isinstance(proving_edge.dst, InternalNode)
            child_proof = proving_edge.dst.extract_proof()
            assert child_proof
            return [proving_edge, *child_proof]


@dataclass
class Edge:
    """An edge in the search tree, representing a tactic."""

    tactic: str
    src: InternalNode = field(repr=False)
    dst: Node = field(repr=False)

    def distance_to_proof(self) -> float:
        return 1 + self.dst.distance_to_proof
