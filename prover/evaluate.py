import os
import pdb
import json
import torch
import random
import pickle
import argparse
from loguru import logger
from pathlib import Path
from typing import List
from lean_dojo import LeanGitRepo, Theorem
from prover.search import Status, DistributedProver
from lean_dojo import Theorem
from generator.model import TransformerTacticGenerator, RetrivalAugmentedTacticGenerator


def get_theorems(args) -> List[Theorem]:
    data_path = Path(args.data_path)
    data = json.load((data_path / f"{args.split}.json").open())
    theorems = [
        Theorem(LeanGitRepo(t["url"], t["commit"]), t["file_path"], t["full_name"])
        for t in data
    ]
    random.shuffle(theorems)
    logger.info(f"{len(theorems)} theorems loaded from {data_path}")
    return theorems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="default")
    parser.add_argument("--output-dir", type=str, default="evals/")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Split of dataset to evaluate on.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/lean_bench/random",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="data/lean_bench/corpus.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["TransformerTacticGenerator", "RetrivalAugmentedTacticGenerator"],
        default="TransformerTacticGenerator",
    )
    parser.add_argument(
        "--generator-ckpt-path",
        type=str,
        required=True,
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--retriever-ckpt-path",
        type=str,
        required=True,
        help="Checkpoint of the premise retriever.",
    )
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=32,
        help="Number of tactics to sample at each node during proof search (Default: 5).",
    )
    # Follow the setup in PACT.
    # Add 300 seconds fow now since we use CPUs.
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Maximum number of seconds the proof search can take (Default: 900).",
    )
    parser.add_argument(
        "--max-num-expansions",
        type=int,
        default=512,
        help="Maximum number of expansions in Best First Search (Default: 512).",
    )
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    args = parser.parse_args()

    logger.info(args)

    theorems = get_theorems(args)
    device = torch.device("cpu")

    if args.model == "TransformerTacticGenerator":
        tac_gen = TransformerTacticGenerator.load(
            args.generator_ckpt_path, device, freeze=True
        )
    else:
        assert args.model == "RetrivalAugmentedTacticGenerator"
        tac_gen = RetrivalAugmentedTacticGenerator(
            args.generator_ckpt_path, args.retriever_ckpt_path, args.corpus_path, device
        )

    prover = DistributedProver(
        num_cpus=args.num_cpus,
        tac_gen=tac_gen,
        timeout=args.timeout,
        max_num_expansions=args.max_num_expansions,
        num_sampled_tactics=args.num_sampled_tactics,
        debug=args.verbose,
        distributed=(args.num_cpus > 1),
    )
    results = prover.search_unordered(theorems)

    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if args.exp_id is not None:
        pickle_path = f"{args.exp_id}_results.pickle"
        pickle.dump(results, open(pickle_path, "wb"))
        logger.info(f"Results saved to {pickle_path}")


if __name__ == "__main__":
    main()
