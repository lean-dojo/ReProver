"""Script for preprocess state-tactic pairs into the format required by [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)."""

import json
import random
import argparse
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/leandojo_benchmark_4/random/train.json",
    )
    parser.add_argument("--dst-path", type=str, default="state_tactic_pairs.json")
    args = parser.parse_args()
    logger.info(args)

    pairs = []
    for thm in json.load(open(args.data_path)):
        for tac in thm["traced_tactics"]:
            pairs.append({"state": tac["state_before"], "output": tac["tactic"]})
    logger.info(f"Read {len(pairs)} state-tactic paris from {args.data_path}")

    random.shuffle(pairs)
    data = [
        {
            "instruction": f"[GOAL]\n{pair['state']}\n[PROOFSTEP]\n",
            "input": "",
            "output": pair["output"],
        }
        for pair in pairs
    ]
    logger.info(data[0])
    json.dump(data, open(args.dst_path, "wt"))
    logger.info(f"Preprocessed data saved to {args.dst_path}")


if __name__ == "__main__":
    main()
