"""Scripts for computing some simple statistics about the data."""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from lean_dojo import Pos
from lean_dojo.constants import LEAN3_DEPS_DIR

from common import Corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default="data/leandojo_benchmark/")
    args = parser.parse_args()
    logger.info(args)

    corpus = Corpus(args.data_path / "corpus.jsonl")

    logger.info(f"Number of files: {corpus.num_files}")
    logger.info(f"Number of premises: {len(corpus)}")

    data_train = json.load(open(args.data_path / "random/train.json"))
    data_val = json.load(open(args.data_path / "random/val.json"))
    data_test = json.load(open(args.data_path / "random/test.json"))

    logger.info(f"Number of training theorems: {len(data_train)}")
    logger.info(f"Number of validation theorems: {len(data_val)}")
    logger.info(f"Number of test theorems: {len(data_test)}")

    tactics = []
    num_accessed_premises = []

    for data in (data_train, data_val, data_test):
        for ex in tqdm(data):
            if ex["file_path"] in corpus:
                file_path = ex["file_path"]
            else:
                _, repo_name = os.path.split(ex["url"])
                file_path = os.path.join(LEAN3_DEPS_DIR, repo_name, ex["file_path"])
            premises = corpus.get_accessible_premises(file_path, Pos(*ex["start"]))
            num_accessed_premises.append(len(premises))
            for t in ex["traced_tactics"]:
                tactics.append(t["annotated_tactic"][0])
                # tactics.append(t["tactic"])

    logger.info(f"Number of tactics: {len(tactics)}")

    tactics_with_premises = [t for t in tactics if "</a>" in t]
    logger.info(f"Number of tactics with premises: {len(tactics_with_premises)}")

    avg_premises_per_tactic = np.mean([t.count("</a>") for t in tactics_with_premises])
    logger.info(
        f"Average number of premises per tactic (among those with premises): {avg_premises_per_tactic}"
    )

    logger.info(
        f"Average number of accessed premises per theorem: {np.mean(num_accessed_premises)}"
    )


if __name__ == "__main__":
    main()
