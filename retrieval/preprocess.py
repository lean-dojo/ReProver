import re
import os
from pathlib import Path
import argparse
import pdb
import random
import tempfile
import json
import networkx as nx
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
from typing import List


def le(x: List[int], y: List[int]) -> bool:
    x_line, x_col = x
    y_line, y_col = y
    return x_line < y_line or (x_line == y_line and x_col <= y_col)


def between(x: List[int], y: List[int], z: List[int]) -> bool:
    return le(x, y) and le(y, z)


class Corpus:
    dep_graph: nx.DiGraph

    def __init__(self, jsonl_path: Path):
        self.dep_graph = nx.DiGraph()
        for line in jsonl_path.open():
            data = json.loads(line)
            assert not self.dep_graph.has_node(data["path"])
            self.dep_graph.add_node(data["path"], premises=data["premises"])
            for p in data["imports"]:
                assert self.dep_graph.has_node(p)
                self.dep_graph.add_edge(data["path"], p)
            nx.is_directed_acyclic_graph(self.dep_graph)

    def get_premises(self, path):
        return self.dep_graph.nodes[path]["premises"]

    def num_premises(self, path):
        return len(self.get_premises(path))

    def get_positive_premise(self, full_name: str, def_path: Path, def_pos: List[int]):
        potential_premises = self.get_premises(def_path)
        for p in potential_premises:
            if between(p["start"], def_pos, p["end"]):
                return {
                    "file": def_path,
                    "full_name": p["full_name"],
                    "code": p["code"],
                }
        logger.warning(full_name, def_path, def_pos)
        logger.warning(list(open(def_path))[def_pos[0] - 1])
        return None

    def sample_negative_premises(
        self, num_negatives: int, full_name: str, def_path: Path, def_pos: List[int]
    ):
        # TODO: Alternative strategies include hard negatives, sample premises in the current file, etc.
        accessible_files = nx.descendants(self.dep_graph, def_path)
        accessible_files.add(def_path)
        accessible_files = list(accessible_files)
        nums = [self.num_premises(p) for p in accessible_files]
        # Sample with replacement.
        sampled_files = random.choices(accessible_files, weights=nums, k=num_negatives)
        negs = [random.choice(self.get_premises(p)) for p in sampled_files]
        return negs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess the data into a Hugging Face Dataset."
    )
    parser.add_argument("--data-path", type=str, default="data/lean_bench")
    parser.add_argument(
        "--dst-path",
        type=str,
        default="data/processed/retrieval",
    )
    parser.add_argument("--num-negatives", type=int, default=10)
    args = parser.parse_args()
    logger.info(args)

    # Write everything to a jsonl file, which is then loaded
    # as a Hugging Face Dataset.
    data_files = {}
    with tempfile.TemporaryDirectory() as dirname:
        corpus = Corpus(Path(args.data_path) / "corpus.jsonl")

        for strategy in ("random", "premise"):
            for split in ("train", "val", "test"):
                data_path = Path(args.data_path) / f"{strategy}/{split}.json"
                jsonl_path = Path(dirname) / f"{split}.jsonl"

                with open(jsonl_path, "wt") as oup:
                    for thm in json.load(data_path.open()):
                        repo_name = thm["url"].split("/")[-1]

                        for tac in thm["traced_tactics"]:
                            annot_tac, attribs = tac["annotated_tactic"]
                            matches = list(
                                re.finditer(r"(?<=<a>).+?(?=</a>)", annot_tac)
                            )
                            for m, a in zip(matches, attribs):
                                ex = {
                                    "file": os.path.join(repo_name, thm["file_path"]),
                                    "theorem": thm["full_name"],
                                    "state": tac["state_before"],
                                    "tactic_prefix": annot_tac[: m.start()],
                                    "tactic_arg": m.group(),
                                    "positive_premise": corpus.get_positive_premise(
                                        **a
                                    ),
                                    "negative_premises": corpus.sample_negative_premises(
                                        args.num_negatives, **a
                                    ),
                                }
                                if ex is not None:
                                    oup.write(json.dumps(ex))

                data_files[split] = str(jsonl_path)

            ds = load_dataset("json", data_files=data_files)
            ds = ds.shuffle()
            dst_path = args.dst_path + "_" + strategy
            ds.save_to_disk(dst_path)
            logger.info(f"Dataset saved to {dst_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
