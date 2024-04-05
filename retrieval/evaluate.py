"""Script for evaluating the premise retriever."""

import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple
from loguru import logger


def _eval(data, preds_map) -> Tuple[float, float, float]:
    R1 = []
    R10 = []
    MRR = []

    for thm in tqdm(data):
        for i, _ in enumerate(thm["traced_tactics"]):
            pred = preds_map[
                (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
            ]
            all_pos_premises = set(pred["all_pos_premises"])
            if len(all_pos_premises) == 0:
                continue

            retrieved_premises = pred["retrieved_premises"]
            TP1 = retrieved_premises[0] in all_pos_premises
            R1.append(float(TP1) / len(all_pos_premises))
            TP10 = len(all_pos_premises.intersection(retrieved_premises[:10]))
            R10.append(float(TP10) / len(all_pos_premises))

            for j, p in enumerate(retrieved_premises):
                if p in all_pos_premises:
                    MRR.append(1.0 / (j + 1))
                    break
            else:
                MRR.append(0.0)

    R1 = 100 * np.mean(R1)
    R10 = 100 * np.mean(R10)
    MRR = np.mean(MRR)
    return R1, R10, MRR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the premise retriever."
    )
    parser.add_argument(
        "--preds-file",
        type=str,
        required=True,
        help="Path to the retriever's predictions file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the directory containing the train/val/test splits.",
    )
    args = parser.parse_args()
    logger.info(args)

    logger.info(f"Loading predictions from {args.preds_file}")
    preds = pickle.load(open(args.preds_file, "rb"))
    preds_map = {
        (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"]): p
        for p in preds
    }
    assert len(preds) == len(preds_map), "Duplicate predictions found!"

    for split in ("train", "val", "test"):
        data_path = os.path.join(args.data_path, f"{split}.json")
        data = json.load(open(data_path))
        logger.info(f"Evaluating on {data_path}")
        R1, R10, MRR = _eval(data, preds_map)
        logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")


if __name__ == "__main__":
    main()
