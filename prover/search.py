import math
import pdb
import ray
import time
import heapq
import torch
import graphviz
import itertools
from pathlib import Path
from enum import Enum
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    TacticState,
    TacticError,
    IncompleteSolve1,
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
)
from pathlib import Path
from loguru import logger
from abc import ABC, abstractmethod
from functools import total_ordering
from dataclasses import dataclass, field
from ray.util.actor_pool import ActorPool
from ray.util.queue import Queue
from typing import List, Optional, Iterable, Tuple, Union
from lean_dojo import ProofFinished, TacticState, TacticError


from common import zip_strict, to_path
from generator.model import (
    TacticGenerator,
    TransformerTacticGenerator,
    RetrivalAugmentedTacticGenerator,
    GPT4TacticGenerator,
)


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
    def distance_to_proof(self) -> float:
        "The smallest number of steps to a proof."
        raise NotImplementedError

    @property
    @abstractmethod
    def time_to_proof(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    def add_to_graph(self, graph: graphviz.Digraph):
        graph.node(
            name=str(id(self)),
            shape="box",
            label=self._graphviz_label,
            fillcolor=self._graphviz_color,
            style=self._graphviz_style,
            penwidth=str(5 if self.status == Status.PROVED else 1),
        )

    @property
    @abstractmethod
    def _graphviz_label(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def _graphviz_color(self) -> str:
        raise NotImplementedError

    @property
    def _graphviz_style(self) -> str:
        return "filled"


@dataclass
class ProofFinishedNode(Node):
    inner: ProofFinished

    status = Status.PROVED
    distance_to_proof = 0.0
    time_to_proof = 0.0
    is_terminal = True

    _graphviz_label = "[Proved]"
    _graphviz_color = "green"


@dataclass
class ErrorNode(Node):
    inner: TacticError

    status = Status.FAILED
    distance_to_proof = math.inf
    time_to_proof = math.inf
    is_terminal = True

    @property
    def _graphviz_label(self) -> str:
        return f"[Error]\n" + self.inner.error.replace("\t", "\n")

    _graphviz_color = "red"


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

    # The score the critic assigned to this node upon creation
    critic_score: Optional[float] = field(compare=False, repr=False)

    # All edges known to lead to this node.
    # May change at any time as other nodes are explored.
    in_edges: List["Edge"] = field(
        default_factory=list, init=False, compare=False, repr=False
    )

    # All edges out of this node that we've considered, or None for unexplored nodes.
    # When a node is explored, this list is populated, and must not change after that.
    _out_edges: Optional[List["Edge"]] = field(
        default=None, init=False, compare=False, repr=False
    )

    # A node is proved if any child is proved, and failed if every child is failed
    # (or there are no children). A node that is proved or failed cannot change status
    # because nothing is ever added to out_edges. _status is recomputed on an as-needed
    # basis by children, since proving or failing a child may prove or fail this node.
    _status: Status = field(default=Status.OPEN, init=False, compare=False, repr=True)

    is_terminal = False  # type: ignore[override]

    # Environment time separating this node from the end of a proof along the
    # optimal path. If unproved, infinity. Updated as needed by children.
    _time_to_proof: float = field(
        default=math.inf, init=False, compare=False, repr=False
    )

    # Number of steps separating this node from the end of a proof along the
    # optimal path. If unproved, infinity. Updated as needed by children.
    _distance_to_proof: float = field(
        default=math.inf, init=False, compare=False, repr=False
    )

    @property
    def out_edges(self):
        return self._out_edges

    # This setter implements exploring this node
    @out_edges.setter
    def out_edges(self, out_edges: Iterable["Edge"]) -> Optional[List["Edge"]]:
        if self.is_explored:
            raise RuntimeError("Node is already explored.")

        self._out_edges = list(out_edges)
        self._recompute_status()
        self._recompute_distance_to_proof()
        self._recompute_time_to_proof()

    # A node is considered explored if we've evaluated the actor in the node to generate
    # a list of candidate children. Explored nodes are never re-searched.
    @property
    def is_explored(self) -> bool:
        return self.out_edges is not None

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, s):
        self._status = s

    def _recompute_status(self):
        """
        Recursively update the status of the current node and its ancestors.
        """
        assert self.is_explored and self.out_edges is not None

        # If this node is proved or failed, nothing can change that
        if self._status != Status.OPEN:
            return

        # If any child is proved, this node is proved, and so are parents recursively
        if any(edge.dst.status == Status.PROVED for edge in self.out_edges):
            self._status = Status.PROVED

        # If all children failed, this node is failed. This may fail some parents too.
        if all(edge.dst.status == Status.FAILED for edge in self.out_edges):
            self._status = Status.FAILED

        # If this node was proved or failed, parents may need to recompute.
        # This is guaranteed to terminate because only open nodes can change, and
        # there are a finite number of open nodes in the tree.
        if self._status != Status.OPEN:
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

    @property
    def time_to_proof(self) -> float:
        return self._time_to_proof

    def _recompute_time_to_proof(self):
        """
        Recursively update the time_to_proof of the current node and its ancestors.
        """
        if self.out_edges:
            time = min(edge.time_to_proof() for edge in self.out_edges)
        else:
            time = math.inf

        if time < self._time_to_proof:
            self._time_to_proof = time
            for edge in self.in_edges:
                edge.src._recompute_time_to_proof()

    # Nodes are sorted by the critic score if available, and cumulative logprob otherwise.
    # NOTE: Nodes are compared by _negative_ priority, to make heapq act as a max-priority-queue.
    @property
    def priority(self) -> float:
        return self.critic_score or self.cumulative_logprob

    def __lt__(self, other: "InternalNode") -> bool:
        return self.priority > other.priority

    def extract_proof(self, min_by_time: bool = False) -> Optional[List["Edge"]]:
        """
        Extract a proof of the current node as a sequence of edges.
        """
        if self.status != Status.PROVED:
            return None
        assert self.is_explored

        # Select the "best" edge, either by total time to proof or total steps to proof.
        # This has an analogy to the Q function, discounted either by time or steps.
        proving_edge = min(
            self.out_edges,
            key=Edge.time_to_proof if min_by_time else Edge.distance_to_proof,
        )

        if proving_edge.dst.is_terminal:
            # Base case: this edge is all that's required to finish the proof
            assert isinstance(proving_edge.dst, ProofFinishedNode)
            return [proving_edge]
        else:
            # Recursive case: prove the child, then add this edge
            assert isinstance(proving_edge.dst, InternalNode)
            child_proof = proving_edge.dst.extract_proof(min_by_time)
            assert child_proof
            return [proving_edge, *child_proof]

    #########
    # Debug #
    #########

    def check_invariants(self):
        """
        Perform some sanity checks.
        """
        if not self.is_explored:
            assert self.status == Status.OPEN
            return  # Nothing more can be said about unexplored nodes

        for edge in self.in_edges:
            assert edge.dst is self

        if self.out_edges == []:
            assert self.status == Status.FAILED
        else:
            for edge in self.out_edges:  # type: ignore
                assert edge.src is self

        if self.status == Status.PROVED:
            assert self.out_edges
            assert any(edge.dst.status == Status.PROVED for edge in self.out_edges)
            assert all(edge.dst.status == Status.PROVED for edge in self.in_edges)

            proof_by_steps = self.extract_proof(False)
            assert proof_by_steps is not None
            assert self.distance_to_proof == len(proof_by_steps)

            proof_by_time = self.extract_proof(True)
            assert proof_by_time is not None
            assert self.time_to_proof == sum(edge.time for edge in proof_by_time)

            assert len(proof_by_steps) <= len(proof_by_time)
            assert sum(edge.time for edge in proof_by_time) <= sum(
                edge.time for edge in proof_by_steps
            )
        elif self.status == Status.FAILED:
            assert self.out_edges is not None
            assert all(edge.dst.status == Status.FAILED for edge in self.out_edges)
            assert self.distance_to_proof == math.inf
            assert self.time_to_proof == math.inf
            assert self.extract_proof(False) == None
            assert self.extract_proof(True) == None
        elif self.status == Status.OPEN:
            assert self.out_edges
            assert not any(edge.dst.status == Status.PROVED for edge in self.out_edges)
            assert not all(edge.dst.status == Status.FAILED for edge in self.out_edges)
            assert self.distance_to_proof == math.inf
            assert self.time_to_proof == math.inf
            assert self.extract_proof(False) == None
            assert self.extract_proof(True) == None

    #################
    # Visualization #
    #################

    @property
    def _graphviz_label(self) -> str:
        label = self.state.tactic_state.replace("\t", "\n")

        if self.status == Status.PROVED:
            label += f"\n\n[Proved]\n{self.distance_to_proof=}\n{self.time_to_proof=}"
        elif self.status == Status.FAILED:
            label += "\n\n[Failed]"
        elif self.is_explored:
            label += "\n\n[Open]"
        else:
            label += "\n\n[Unexplored]"

        label += f"\n{self.cumulative_logprob=:.3f}"

        if self.critic_score is not None:
            label += f"\n{self.critic_score=:.3f}"

        return label

    @property
    def _graphviz_color(self) -> str:
        return "white" if self.critic_score is None else f"0.33 {self.critic_score} 1.0"

    @property
    def _graphviz_style(self) -> str:
        return "filled,diagonals" if self.status == Status.FAILED else "filled"

    def unique_out_edges_and_counts(self) -> List[Tuple["Edge", int]]:
        edges_and_counts = {}

        for edge in self.out_edges or []:
            if edge.tactic in edges_and_counts:
                edges_and_counts[edge.tactic][1] += 1
            else:
                edges_and_counts[edge.tactic] = [edge, 1]

        return [(edge, count) for edge, count in edges_and_counts.values()]


@dataclass
class Edge:
    """An edge in the search tree, representing a tactic."""

    tactic: str
    src: InternalNode = field(repr=False)
    dst: Node = field(repr=False)
    logprob: float = field(repr=False, compare=False)
    time: float = field(repr=False, compare=False)

    def distance_to_proof(self) -> float:
        return 1 + self.dst.distance_to_proof

    def time_to_proof(self) -> float:
        return self.time + self.dst.time_to_proof

    #################
    # Visualization #
    #################

    def add_to_graph(self, graph: graphviz.Digraph, count: Optional[int] = None):
        label = (
            self.tactic.replace("\t", "\n")
            + f"\n\n{self.logprob=:.3f}\n{self.time=:.5f}"
        )

        if count:
            label += f"\n{count=}"

        if self.dst.status == Status.PROVED:
            weight = 50
            width = 3
        elif isinstance(self.dst, ErrorNode):
            weight = -100
            width = 1
        else:
            assert isinstance(self.dst, InternalNode)
            weight = self.dst.priority
            width = 1

        graph.edge(
            str(id(self.src)),
            str(id(self.dst)),
            label=label,
            shape="box",
            weight=str(weight),
            penwidth=str(width),
        )


@dataclass
class SearchResult:
    """The results of attempting to prove a theorem."""

    name: str
    status: Status

    shortest_proof: Optional[List[str]]
    fastest_proof: Optional[List[str]]

    actor_time: float
    environment_time: float
    total_time: float

    num_total_nodes: int
    num_searched_nodes: int

    tree: Node


class BestFirstSearchProver:
    def __init__(
        self,
        tac_gen: TacticGenerator,
        timeout: int,
        max_num_expansions: int,
        num_sampled_tactics: int,
        debug: bool,
        gpu_tac_gen,
        try_cpu_first,
        input_queue: Optional[Queue],
        output_queue: Optional[Queue],
    ) -> None:
        self.tac_gen = tac_gen
        self.timeout = timeout
        self.max_num_expansions = max_num_expansions
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug
        self.gpu_tac_gen = gpu_tac_gen
        self.try_cpu_first = try_cpu_first
        self.input_queue = input_queue
        self.output_queue = output_queue
        assert output_queue is None or output_queue.empty()

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

    def search(self, thm: Theorem, pos: Pos) -> Optional[SearchResult]:
        if self.try_cpu_first:
            result = self._search(thm, pos, True)
            if result is None:
                return None
            elif result.total_time > self.timeout and self.gpu_tac_gen is not None:
                logger.info("Retry with GPU")
                return self._search(thm, pos, False)
        else:
            return self._search(thm, pos, False)

    def _search(self, thm: Theorem, pos: Pos, cpu_only: bool) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.theorem = thm
        self.posision = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        # Cache (state, action) -> (response, time) to avoid duplicate environment steps
        self._transition_cache = {}

        try:
            with Dojo(thm) as (dojo, init_state):
                self.dojo = dojo
                self.root = InternalNode(
                    state=init_state,
                    cumulative_logprob=0.0,
                    # critic_score=self._get_critic_score(init_state.tactic_state),
                    critic_score=None,
                )
                self.nodes = {init_state: self.root}
                self.priority_queue = [self.root]

                with torch.no_grad():
                    try:
                        self._best_first_search(cpu_only)
                    except DojoCrashError:
                        logger.warning(f"Dojo crashed when proving {thm}")
                        pass

            if self.root.status == Status.PROVED:
                shortest_proof = [e.tactic for e in self.root.extract_proof(False)]  # type: ignore
                fastest_proof = [e.tactic for e in self.root.extract_proof(True)]  # type: ignore
            else:
                shortest_proof = fastest_proof = None

            result = SearchResult(
                name=thm.full_name,
                status=self.root.status,
                shortest_proof=shortest_proof,
                fastest_proof=fastest_proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
                tree=self.root,
                # graph=self.to_graphviz(),
            )
            logger.info(result)
            return result

        except DojoInitError as ex:
            return None

    def _best_first_search(self, cpu_only: bool) -> None:
        time_start = time.monotonic()

        while True:
            if len(self.priority_queue) == 0:
                logger.info("Ran out of nodes to search.")
                break

            self._step(cpu_only)

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                self.root.status = Status.OPEN
                logger.info("Search timed out.")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break

            if (self.max_num_expansions is not None) and (
                self.num_expansions >= self.max_num_expansions
            ):
                logger.info("Max expansions reached.")
                break

    def _step(self, cpu_only: bool):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """
        # Search the node with highest priority (lowest negative priority).
        search_node = heapq.heappop(self.priority_queue)
        logger.debug(f"Expanding node: {search_node}")

        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in self.priority_queue
            )

        # Query the model for tactics
        if isinstance(search_node.state, TacticState):
            ts = search_node.state.value
        else:
            ts = search_node.state.unsolved_tactic_state
        suggestions = self._generate_tactics(ts, cpu_only)

        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        results = [
            self._run_tactic(search_node, tactic, logprob)
            for tactic, logprob in suggestions
        ]

        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1

        # If we're running in debug mode, run a full test suite each step
        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )
            self.check_invariants()

    def _batch_generate_gpu(
        self,
        state: List[str],
        file_path: List[Path],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        req = {
            "state": state,
            "file_path": file_path,
            "theorem_full_name": theorem_full_name,
            "theorem_pos": theorem_pos,
            "num_samples": num_samples,
            "output_queue": self.output_queue,
        }
        logger.debug(f"Sending request to GPU: {req}")
        self.input_queue.put(req)
        logger.debug("Waiting for response from GPU...")
        res = self.output_queue.get()
        logger.debug(f"Got a response from GPU: {res}")
        return res

    def _generate_tactics(self, ts: str, cpu_only: bool) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        num_goals = ts.count("⊢")
        assert num_goals >= 1
        if num_goals == 1:
            if not cpu_only and self.gpu_tac_gen is not None:
                suggestions = self._batch_generate_gpu(
                    state=[ts],
                    file_path=[Path(self.theorem.repo.name) / self.theorem.file_path],
                    theorem_full_name=[self.theorem.full_name],
                    theorem_pos=[self.posision],
                    num_samples=self.num_sampled_tactics,
                )[0]
            else:
                suggestions = self.tac_gen.generate(
                    state=ts,
                    file_path=Path(self.theorem.repo.name) / self.theorem.file_path,
                    theorem_full_name=self.theorem.full_name,
                    theorem_pos=self.posision,
                    num_samples=self.num_sampled_tactics,
                )
        else:
            first_goal = ts.split("\n\n")[0]
            path = Path(self.theorem.repo.name) / self.theorem.file_path
            if not cpu_only and self.gpu_tac_gen is not None:
                all_suggestions = self._batch_generate_gpu(
                    state=[ts, first_goal],
                    file_path=[path, path],
                    theorem_full_name=[self.theorem.full_name, self.theorem.full_name],
                    theorem_pos=[self.posision, self.posision],
                    num_samples=self.num_sampled_tactics,
                )
            else:
                all_suggestions = self.tac_gen.batch_generate(
                    state=[ts, first_goal],
                    file_path=[path, path],
                    theorem_full_name=[self.theorem.full_name, self.theorem.full_name],
                    theorem_pos=[self.posision, self.posision],
                    num_samples=self.num_sampled_tactics,
                )
            suggestions = {}
            for t, s in itertools.chain.from_iterable(all_suggestions):
                if t not in suggestions or suggestions[t] < s:
                    suggestions[t] = s
            suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)[
                : self.num_sampled_tactics
            ]

        elapsed = time.monotonic() - t0
        self.actor_time += elapsed

        logger.debug(f"Tactic suggestions: {suggestions}")
        # TODO: Too many repetitions.
        return suggestions

    def _run_tactic(self, node: InternalNode, tactic: str, logprob: float) -> Edge:
        # Must separately record time here, because caching might return a higher time
        logger.debug(f"Trying a tactic: {tactic}")
        # if tactic.startswith("{"):
        #    assert node.state.num_goals > 1
        t0 = time.monotonic()
        response = self.dojo.run_tac(node.state, tactic)

        # If the gym crashed, unwind the stack up to `search()` and give up.
        # if result_type == environment.ResultType.ERROR:
        #    raise GymCrashedError()

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        try:
            # If we've seen this response before, use the existing node
            result_node = self.nodes[response]
        except KeyError:
            # Build a new node
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
            elif type(response) in (TacticError, IncompleteSolve1, ProofGivenUp):
                result_node = ErrorNode(response)
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                    critic_score=None,
                )

            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                # logger.debug(
                #    f"Enqueuing the resulting node with priority {result_node.cumulative_logprob}"
                # )
                heapq.heappush(self.priority_queue, result_node)  # type: ignore

        # Record the new node and add it to the search queue
        self.nodes[response] = result_node

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(
            tactic=tactic,
            src=node,
            dst=result_node,
            logprob=logprob,
            time=elapsed,
        )

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge

    #########
    # DEBUG #
    #########

    def check_invariants(self):
        """
        Perform some sanity checks.
        """
        for node in self.priority_queue:
            assert node in self.nodes.values()
            assert isinstance(node, InternalNode)
            assert not node.is_explored

        for response, node in self.nodes.items():
            if isinstance(response, ProofFinished):
                assert isinstance(node, ProofFinishedNode)
                assert node not in self.priority_queue
                assert self.root.status == Status.PROVED
            elif type(response) in (TacticError, IncompleteSolve1, ProofGivenUp):
                assert isinstance(node, ErrorNode)
                assert node not in self.priority_queue
            else:
                assert isinstance(node, InternalNode)

                if node.is_explored:
                    assert node not in self.priority_queue
                else:
                    assert node in self.priority_queue

                node.check_invariants()

    #################
    # Visualization #
    #################

    def to_graphviz(self, hide_errors: bool = False) -> graphviz.Digraph:
        graph = graphviz.Digraph()

        for node in self.nodes.values():
            if not (hide_errors and isinstance(node, ErrorNode)):
                node.add_to_graph(graph)

            if isinstance(node, InternalNode) and node.out_edges is not None:
                for edge, count in node.unique_out_edges_and_counts():
                    if not (hide_errors and isinstance(edge.result, ErrorNode)):
                        edge.add_to_graph(graph, count)

        return graph


def create_tactic_generator(
    model: str, gen_ckpt_path: Path, ret_ckpt_path: Path, device
):
    if model == "TransformerTacticGenerator":
        return TransformerTacticGenerator.load(gen_ckpt_path, device, freeze=True)
    elif model == "RetrivalAugmentedTacticGenerator":
        return RetrivalAugmentedTacticGenerator(gen_ckpt_path, ret_ckpt_path, device)
    else:
        assert model == "GPT4TacticGenerator"
        return GPT4TacticGenerator()


@ray.remote(num_cpus=1)
class CpuProver(BestFirstSearchProver):
    def __init__(
        self,
        model: str,
        gen_ckpt_path: Path,
        ret_ckpt_path: Path,
        timeout: int,
        max_num_expansions: int,
        num_sampled_tactics: int,
        debug: bool,
        gpu_tac_gen=None,
        try_cpu_first: bool = False,
        # TODO: set try_cpu_first
        input_queue: Optional[Queue] = None,
        output_queue: Optional[Queue] = None,
    ) -> None:
        tac_gen = create_tactic_generator(
            model, gen_ckpt_path, ret_ckpt_path, torch.device("cpu")
        )
        super().__init__(
            tac_gen,
            timeout,
            max_num_expansions,
            num_sampled_tactics,
            debug,
            gpu_tac_gen,
            try_cpu_first,
            input_queue,
            output_queue,
        )


# TODO: Merge with CpuProver
@ray.remote(num_cpus=1, num_gpus=1)
class GpuProver(BestFirstSearchProver):
    def __init__(
        self,
        model: str,
        gen_ckpt_path: Path,
        ret_ckpt_path: Path,
        timeout: int,
        max_num_expansions: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        tac_gen = create_tactic_generator(
            model, gen_ckpt_path, ret_ckpt_path, torch.device("cuda")
        )
        super().__init__(
            tac_gen,
            timeout,
            max_num_expansions,
            num_sampled_tactics,
            debug,
            None,
            False,
            None,
            None,
        )


@ray.remote(num_cpus=1, num_gpus=1)
class GpuTacticGenerator:
    def __init__(
        self, model: str, gen_ckpt_path: Path, ret_ckpt_path: Path, input_queue: Queue
    ) -> None:
        self.input_queue = input_queue
        self.tac_gen = create_tactic_generator(
            model, gen_ckpt_path, ret_ckpt_path, torch.device("cuda")
        )

    def run(self) -> None:
        # TODO: Add some statistics, e.g., idle time, number of requests, etc.
        logger.debug("GPU Tactic Generator waiting for jobs...")
        # TODO: It's also possible to improve the batching of RetrievalAugmentedTacticGenerator
        while True:
            n = self.input_queue.size()
            if n == 0:
                time.sleep(0.1)  # TODO: Make this configurable.
                continue
            reqs = self.input_queue.get_nowait_batch(n)
            logger.debug(f"GPU Tactic Generator got {n} requests {reqs}")

            state = []
            file_path = []
            theorem_full_name = []
            theorem_pos = []
            num_samples = []

            for req in reqs:
                state.extend(req["state"])
                file_path.extend(req["file_path"])
                theorem_full_name.extend(req["theorem_full_name"])
                theorem_pos.extend(req["theorem_pos"])
                num_samples.append(req["num_samples"])

            assert all(num_samples[0] for _ in num_samples)
            num_samples = num_samples[0]

            all_suggestions = self.tac_gen.batch_generate(
                state, file_path, theorem_full_name, theorem_pos, num_samples
            )

            base_idx = 0
            for req in reqs:
                suggestions = all_suggestions[base_idx : base_idx + len(req["state"])]
                base_idx += len(req["state"])
                req["output_queue"].put_nowait(suggestions)

            logger.debug(f"Requests finished. {self.input_queue.size()} requests left.")


class DistributedProver:
    def __init__(
        self,
        model: str,
        gen_ckpt_path: Union[str, Path],
        ret_ckpt_path: Union[str, Path],
        num_cpus: int,
        num_gpus: int,
        timeout: int,
        max_num_expansions: int,
        num_sampled_tactics: int,
        debug: Optional[bool] = False,
    ) -> None:
        gen_ckpt_path = to_path(gen_ckpt_path)
        ret_ckpt_path = to_path(ret_ckpt_path)
        ray.init()

        assert num_gpus <= num_cpus
        if num_gpus == num_cpus:
            # Simply give each CPU worker its own GPU.
            logger.warning(
                f"Launching {num_gpus} workers, each with its own GPU. This may lead to low GPU utilization."
            )
            provers = [
                GpuProver.remote(
                    model,
                    gen_ckpt_path,
                    ret_ckpt_path,
                    timeout=timeout,
                    max_num_expansions=max_num_expansions,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                )
                for _ in range(num_cpus)
            ]
        elif num_gpus == 0:
            logger.info(f"Launching {num_cpus} CPU workers.")
            # CPUs only.
            provers = [
                CpuProver.remote(
                    model,
                    gen_ckpt_path,
                    ret_ckpt_path,
                    timeout,
                    max_num_expansions,
                    num_sampled_tactics,
                    debug,
                )
                for _ in range(num_cpus)
            ]
        else:
            if num_gpus > 1:
                raise NotImplementedError
            # N CPU workers sharing M GPUs (possibly M << N).
            logger.info(
                f"Launching {num_cpus - num_gpus} CPU workers, sharing {num_gpus} GPUs."
            )
            logger.warning("try_cpu_first == False")
            input_queue = Queue()
            gpu_tac_gen = GpuTacticGenerator.remote(
                model, gen_ckpt_path, ret_ckpt_path, input_queue
            )
            gpu_tac_gen.run.remote()
            output_queues = [Queue() for _ in range(num_cpus - num_gpus)]
            provers = [
                CpuProver.remote(
                    model,
                    gen_ckpt_path,
                    ret_ckpt_path,
                    timeout,
                    max_num_expansions,
                    num_sampled_tactics,
                    debug,
                    gpu_tac_gen,
                    False,
                    input_queue,
                    output_queues[i],
                )
                for i in range(num_cpus - num_gpus)
            ]

        self.prover_pool = ActorPool(provers)

    """
    def _exit_gracefully(self, signum, frame):
        ray.shutdown()
        if signum == signal.SIGINT:
            self.old_sigint(signum, frame)
        elif signum == signal.SIGTERM:
            self.old_sigterm(signum, frame)
    """

    def search_unordered(
        self, theorems: List[Theorem], positions: List[Pos]
    ) -> List[SearchResult]:
        # self.old_sigint = signal.signal(signal.SIGINT, self._exit_gracefully)
        # self.old_sigterm = signal.signal(signal.SIGTERM, self._exit_gracefully)
        results = list(
            self.prover_pool.map_unordered(
                lambda p, x: p.search.remote(x[0], x[1]),
                zip_strict(theorems, positions),
            )
        )
        # signal.signal(signal.SIGINT, self.old_sigint)
        # signal.signal(signal.SIGTERM, self.old_sigterm)
        # TODO: https://docs.ray.io/en/latest/ray-core/actors/terminating-actors.html
        return results
