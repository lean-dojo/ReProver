import sys
import pdb
import ray
import time
import heapq
import torch
import itertools
from pathlib import Path
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
from dataclasses import dataclass
from ray.util.actor_pool import ActorPool
from ray.util.queue import Queue, Empty
from typing import List, Optional, Tuple, Union
from lean_dojo import ProofFinished, TacticState, TacticError


from common import zip_strict
from generator.model import (
    TacticGenerator,
    TransformerTacticGenerator,
    RetrivalAugmentedTacticGenerator,
    GPT4TacticGenerator,
)
from prover.search_tree import *


@dataclass(frozen=True)
class SearchResult:
    """The results of attempting to prove a theorem."""

    name: str
    status: Status
    shortest_proof: Optional[List[str]]
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int
    tree: Node


@dataclass(frozen=True)
class ExternalTacticGeneratorConfig:
    requests_queue: Queue
    responses_queue: Queue
    try_internal_first: bool

    def __post_init__(self):
        assert isinstance(self.requests_queue, Queue) and self.requests_queue.empty()
        assert isinstance(self.responses_queue, Queue) and self.responses_queue.empty()
        assert isinstance(self.try_internal_first, bool)


@dataclass(frozen=True)
class TacticGeneratorConfig:
    tac_gen: Optional[TacticGenerator]
    external: Optional[ExternalTacticGeneratorConfig]

    def __post_init__(self):
        if self.tac_gen is None:
            assert (
                self.external is not None
                and not self.external_tac_gen.try_internal_first
            )
        else:
            assert isinstance(self.tac_gen, TacticGenerator)
            assert self.tac_gen.device.type == "cpu" or self.external is None


@dataclass(eq=False)
class InferenceRequest:
    state: List[str]
    file_path: List[Path]
    theorem_full_name: List[str]
    theorem_pos: List[Pos]
    num_samples: int
    responses_queue: Queue

    time_creation: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        assert isinstance(self.responses_queue, Queue) and self.responses_queue.empty()


@dataclass(eq=False)
class InferenceResponse:
    tactic_suggestions: List[List[Tuple[str, float]]]
    time_dequeue: float
    time_completion: float


class BestFirstSearchProver:
    def __init__(
        self,
        tac_gen_config: TacticGeneratorConfig,
        timeout: int,
        max_num_expansions: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        self.tac_gen_config = tac_gen_config
        self.timeout = timeout
        self.max_num_expansions = max_num_expansions
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

    @property
    def tac_gen(self) -> TacticGenerator:
        assert self.tac_gen_config.tac_gen is not None
        return self.tac_gen_config.tac_gen

    @property
    def requests_queue(self) -> Queue:
        assert self.tac_gen_config.external is not None
        return self.tac_gen_config.external.requests_queue

    @property
    def responses_queue(self) -> Queue:
        assert self.tac_gen_config.external is not None
        return self.tac_gen_config.external.responses_queue

    def search(self, thm: Theorem, pos: Pos) -> Optional[SearchResult]:
        if self.tac_gen_config.external is None:
            result = self._search(thm, pos, use_external=False)
        elif self.tac_gen_config.external.try_internal_first:
            result = self._search(thm, pos, use_external=False)
            if result is not None and result.total_time > self.timeout:
                logger.info("Retrying with GPU...")
                result = self._search(thm, pos, use_external=True)
        else:
            result = self._search(thm, pos, use_external=True)
        logger.info(result)
        return result

    def _search(
        self, thm: Theorem, pos: Pos, use_external: bool
    ) -> Optional[SearchResult]:
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
                )
                self.nodes = {init_state: self.root}
                self.priority_queue = [self.root]

                with torch.no_grad():
                    try:
                        self._best_first_search(use_external)
                    except DojoCrashError:
                        logger.warning(f"Dojo crashed when proving {thm}")
                        pass

            if self.root.status == Status.PROVED:
                shortest_proof = [e.tactic for e in self.root.extract_proof()]
            else:
                shortest_proof = None

            result = SearchResult(
                name=thm.full_name,
                status=self.root.status,
                shortest_proof=shortest_proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
                tree=self.root,
            )
            return result

        except DojoInitError as ex:
            return None

    def _best_first_search(self, use_external: bool) -> None:
        time_start = time.monotonic()

        while True:
            if len(self.priority_queue) == 0:
                logger.info("Ran out of nodes to search.")
                break

            self._step(use_external)

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

    def _step(self, use_external: bool):
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
        suggestions = self._generate_tactics(ts, use_external)

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

    def _external_batch_generate(
        self,
        state: List[str],
        file_path: List[Path],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        req = InferenceRequest(
            state,
            file_path,
            theorem_full_name,
            theorem_pos,
            num_samples,
            self.responses_queue,
        )
        self.requests_queue.put_nowait(req)
        res = self.responses_queue.get(block=True)
        return res.tactic_suggestions

    def _generate_tactics(self, ts: str, use_external: bool) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        num_goals = ts.count("⊢")
        assert num_goals >= 1
        if num_goals == 1:
            if use_external:
                suggestions = self._external_batch_generate(
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
            if use_external:
                all_suggestions = self._external_batch_generate(
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
        # logger.debug(f"Trying a tactic: {tactic}")
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


def create_tactic_generator(
    model: str,
    gen_ckpt_path: Path,
    ret_ckpt_path: Path,
    device,
    length_penalty: Optional[float] = None,
    temperature: Optional[float] = None,
    retrieval_weight: Optional[float] = None,
):
    if model == "TransformerTacticGenerator":
        return TransformerTacticGenerator.load(gen_ckpt_path, device, freeze=True)
    elif model == "RetrivalAugmentedTacticGenerator":
        return RetrivalAugmentedTacticGenerator(
            gen_ckpt_path,
            ret_ckpt_path,
            device,
            length_penalty,
            temperature,
            retrieval_weight,
        )
    else:
        assert model == "GPT4TacticGenerator"
        return GPT4TacticGenerator()


@ray.remote
class CpuProver(BestFirstSearchProver):
    def __init__(
        self,
        model: str,
        gen_ckpt_path: Path,
        ret_ckpt_path: Path,
        external_config: Optional[ExternalTacticGeneratorConfig],
        timeout: int,
        max_num_expansions: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        if external_config is not None and not external_config.try_internal_first:
            tac_gen = None
        else:
            tac_gen = create_tactic_generator(
                model, gen_ckpt_path, ret_ckpt_path, torch.device("cpu")
            )
        tac_gen_config = TacticGeneratorConfig(tac_gen, external_config)
        super().__init__(
            tac_gen_config,
            timeout,
            max_num_expansions,
            num_sampled_tactics,
            debug,
        )


@ray.remote(num_gpus=1)
class GpuProver(BestFirstSearchProver):
    def __init__(
        self,
        model: str,
        gen_ckpt_path: Path,
        ret_ckpt_path: Path,
        external_config: Optional[ExternalTacticGeneratorConfig],
        timeout: int,
        max_num_expansions: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        if external_config is not None and not external_config.try_internal_first:
            tac_gen = None
        else:
            tac_gen = create_tactic_generator(
                model, gen_ckpt_path, ret_ckpt_path, torch.device("cuda")
            )
        tac_gen_config = TacticGeneratorConfig(tac_gen, external_config)
        super().__init__(
            tac_gen_config,
            timeout,
            max_num_expansions,
            num_sampled_tactics,
            debug,
        )


@ray.remote(num_gpus=1)
class GpuTacticGenerator:
    def __init__(
        self,
        model: str,
        gen_ckpt_path: Path,
        ret_ckpt_path: Path,
        requests_queue: Queue,
        batch_size: int,
    ) -> None:
        self.model = model
        self.gen_ckpt_path = gen_ckpt_path
        self.ret_ckpt_path = ret_ckpt_path
        self.requests_queue = requests_queue
        self.batch_size = batch_size

    def initialize(self):
        self.tac_gen = create_tactic_generator(
            self.model, self.gen_ckpt_path, self.ret_ckpt_path, torch.device("cuda")
        )

    def run(self) -> None:
        # TODO: Add some statistics, e.g., idle time, number of requests, etc.

        # TODO: It's also possible to improve the batching of RetrievalAugmentedTacticGenerator
        while True:
            n = self.requests_queue.size()
            if n >= 1000:
                logger.error("Hard limit of 1000 reached for requests_queue.")
                ray.shutdown()
                sys.exit(1)
            elif n > 800:
                logger.warning("Soft limit of 800 reached for requests_queue.")
            reqs = [self.requests_queue.get(block=True)]
            for _ in range(self.batch_size - 1):
                try:
                    reqs.append(self.requests_queue.get_nowait())
                except Empty:
                    break

            time_dequeue = time.monotonic()
            logger.info(f"GPU Tactic Generator retrieved {len(reqs)} requests {reqs}")

            state = []
            file_path = []
            theorem_full_name = []
            theorem_pos = []
            num_samples = reqs[0].num_samples

            for req in reqs:
                state.extend(req.state)
                file_path.extend(req.file_path)
                theorem_full_name.extend(req.theorem_full_name)
                theorem_pos.extend(req.theorem_pos)
                assert num_samples == req.num_samples

            all_suggestions = self.tac_gen.batch_generate(
                state, file_path, theorem_full_name, theorem_pos, num_samples
            )
            time_completion = time.monotonic()

            base_idx = 0
            for req in reqs:
                suggestions = all_suggestions[base_idx : base_idx + len(req.state)]
                base_idx += len(req.state)
                res = InferenceResponse(suggestions, time_dequeue, time_completion)
                req.responses_queue.put_nowait(res)
                time_wait = time_dequeue - req.time_creation
                time_inference = time_completion - time_dequeue
                logger.info(
                    f"Request finished. Waiting time: {time_wait}. Inference time {time_inference}"
                )
                if time_wait >= time_inference:
                    logger.warning(
                        "Excessive waiting time. Consider having fewer CPUs per GPU."
                    )

            logger.info(
                f"A batch finished. {self.requests_queue.size()} requests left."
            )


class DistributedProver:
    def __init__(
        self,
        model: str,
        gen_ckpt_path: Union[str, Path],
        ret_ckpt_path: Union[str, Path],
        length_penalty: float,
        temperature: float,
        retrieval_weight: float,
        num_cpus: int,
        num_gpus: int,
        timeout: int,
        max_num_expansions: int,
        num_sampled_tactics: int,
        debug: Optional[bool] = False,
    ) -> None:
        gen_ckpt_path = Path(gen_ckpt_path)
        ret_ckpt_path = Path(ret_ckpt_path)

        self.distributed = num_cpus > 1
        if not self.distributed:
            tac_gen = create_tactic_generator(
                model,
                gen_ckpt_path,
                ret_ckpt_path,
                torch.device("cuda" if num_gpus > 0 else "cpu"),
                length_penalty,
                temperature,
                retrieval_weight,
            )
            tac_gen_config = TacticGeneratorConfig(tac_gen, None)
            self.prover = BestFirstSearchProver(
                tac_gen_config, timeout, max_num_expansions, num_sampled_tactics, debug
            )
            return

        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

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
                    external_config=None,
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
                    external_config=None,
                    timeout=timeout,
                    max_num_expansions=max_num_expansions,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                )
                for _ in range(num_cpus)
            ]
        else:
            # N CPU workers sharing M GPUs (possibly M << N).
            logger.info(f"Launching {num_cpus} CPU workers, sharing {num_gpus} GPUs.")

            requests_queue = Queue(maxsize=1000)
            responses_queues = [Queue(maxsize=1000) for _ in range(num_cpus)]

            gpu_tac_gens = [
                GpuTacticGenerator.remote(
                    model, gen_ckpt_path, ret_ckpt_path, requests_queue, batch_size=1
                )
                for _ in range(num_gpus)
            ]
            ray.get([g.initialize.remote() for g in gpu_tac_gens])
            for g in gpu_tac_gens:
                g.run.remote()

            provers = [
                CpuProver.remote(
                    model,
                    gen_ckpt_path,
                    ret_ckpt_path,
                    external_config=ExternalTacticGeneratorConfig(
                        requests_queue, responses_queues[i], try_internal_first=True
                    ),
                    timeout=timeout,
                    max_num_expansions=max_num_expansions,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                )
                for i in range(num_cpus)
            ]

        # TODO: We can make the prover more scalable if ActorPool works with async workers.
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
        if not self.distributed:
            return [
                self.prover.search(thm, pos)
                for thm, pos in zip_strict(theorems, positions)
            ]
        # self.old_sigint = signal.signal(signal.SIGINT, self._exit_gracefully)
        # self.old_sigterm = signal.signal(signal.SIGTERM, self._exit_gracefully)
        try:
            results = list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(x[0], x[1]),
                    zip_strict(theorems, positions),
                )
            )
        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)
        # signal.signal(signal.SIGINT, self.old_sigint)
        # signal.signal(signal.SIGTERM, self.old_sigterm)
        # TODO: https://docs.ray.io/en/latest/ray-core/actors/terminating-actors.html
        return results
