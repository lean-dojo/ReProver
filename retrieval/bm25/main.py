"""Script for training the BM25 premise retriever."""

import os
import ray
import json
import pickle
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import multiprocessing
from loguru import logger
from common import Corpus
from lean_dojo import Pos
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer
from typing import List, Dict, Any
from ray.util.actor_pool import ActorPool


from common import Context, get_all_pos_premises


def _process_theorem(
    thm: Dict[str, Any],
    corpus: Corpus,
    tokenizer,
    bm25,
    num_retrieved: int,
    use_all_premises: bool,
) -> List[Dict[str, Any]]:
    preds = []
    file_path = thm["file_path"]

    if use_all_premises:
        accessible_premise_idxs = list(range(len(corpus)))
    else:
        accessible_premise_idxs = corpus.get_accessible_premise_indexes(
            file_path, Pos(*thm["start"])
        )

    for i, tac in enumerate(thm["traced_tactics"]):
        ctx = Context(
            file_path, thm["full_name"], Pos(*thm["start"]), tac["state_before"]
        )
        tokenized_ctx = tokenizer.encode(ctx.serialize()).tokens

        scores = np.array(bm25.get_batch_scores(tokenized_ctx, accessible_premise_idxs))
        scores_idxs = np.argsort(scores)[::-1][:num_retrieved]
        retrieved_idxs = [accessible_premise_idxs[i] for i in scores_idxs]
        retrieved_premises = [corpus[i] for i in retrieved_idxs]
        retrieved_scores = scores[scores_idxs].tolist()

        all_pos_premises = get_all_pos_premises(tac["annotated_tactic"], corpus)
        preds.append(
            {
                "url": thm["url"],
                "commit": thm["commit"],
                "file_path": thm["file_path"],
                "full_name": thm["full_name"],
                "start": thm["start"],
                "tactic_idx": i,
                "context": ctx,
                "all_pos_premises": all_pos_premises,
                "retrieved_premises": retrieved_premises,
                "scores": retrieved_scores,
            }
        )

    return preds


@ray.remote(num_cpus=1)
class TheoremProcessor:
    def __init__(
        self,
        tokenizer_path: str,
        data_path: str,
        num_retrieved: int,
        use_all_premises: bool,
    ) -> None:
        self.num_retrieved = num_retrieved
        self.use_all_premises = use_all_premises

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.corpus = Corpus(os.path.join(data_path, "../corpus.jsonl"))
        premises = [premise.serialize() for premise in self.corpus.all_premises]
        tokenized_premises = [self.tokenizer.encode(p).tokens for p in premises]
        self.bm25 = BM25Okapi(tokenized_premises)

    def process_theorem(self, thm: Dict[str, Any]):
        return _process_theorem(
            thm,
            self.corpus,
            self.tokenizer,
            self.bm25,
            self.num_retrieved,
            self.use_all_premises,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for training the BM25 premise retriever."
    )
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument("--num-retrieved", type=int, default=100)
    parser.add_argument("--use-all-premises", action="store_true")
    parser.add_argument("--num-cpus", type=int, default=32)
    args = parser.parse_args()
    logger.info(args)

    if multiprocessing.cpu_count() < args.num_cpus:
        logger.warning(
            f"Number of cpus requested ({args.num_cpus}) is greater than the number of cpus available ({multiprocessing.cpu_count()})"
        )

    theorems = list(
        itertools.chain.from_iterable(
            json.load(open(os.path.join(args.data_path, f"{split}.json")))
            for split in ("train", "val", "test")
        )
    )

    if args.num_cpus > 1:
        pool = ActorPool(
            [
                TheoremProcessor.remote(
                    args.tokenizer_path,
                    args.data_path,
                    args.num_retrieved,
                    args.use_all_premises,
                )
                for _ in range(args.num_cpus)
            ]
        )
        preds = list(
            itertools.chain.from_iterable(
                tqdm(
                    pool.map_unordered(
                        lambda a, thm: a.process_theorem.remote(thm), theorems
                    ),
                    total=len(theorems),
                )
            )
        )
    else:
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        corpus = Corpus(os.path.join(args.data_path, "../corpus.jsonl"))
        premises = [premise.serialize() for premise in corpus.all_premises]
        tokenized_premises = [tokenizer.encode(p).tokens for p in premises]
        bm25 = BM25Okapi(tokenized_premises)

        preds = list(
            itertools.chain.from_iterable(
                [
                    _process_theorem(
                        thm,
                        corpus,
                        tokenizer,
                        bm25,
                        args.num_retrieved,
                        args.use_all_premises,
                    )
                    for thm in tqdm(theorems)
                ]
            )
        )

    with open(args.output_path, "wb") as oup:
        pickle.dump(preds, oup)
    logger.info(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
