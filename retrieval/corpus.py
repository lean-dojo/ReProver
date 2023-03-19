import random
import json
from loguru import logger
import networkx as nx
from datasets import Dataset
from pathlib import Path


class Corpus:
    dep_graph: nx.DiGraph
    all_premises: Dataset

    def __init__(self, jsonl_path: Path):
        self.dep_graph = nx.DiGraph()
        all_premises = []

        for line in jsonl_path.open():
            data = json.loads(line)
            path = data["path"]
            assert not self.dep_graph.has_node(path)
            premises = data["premises"]
            for prem in premises:
                all_premises.append(
                    _format_doc(
                        {
                            "file": path,
                            "full_name": prem["full_name"],
                            "code": prem["code"],
                        }
                    )
                )
            self.dep_graph.add_node(path, premises=premises)
            for p in data["imports"]:
                assert self.dep_graph.has_node(p)
                self.dep_graph.add_edge(path, p)
            nx.is_directed_acyclic_graph(self.dep_graph)

        self.all_premises = datasets.Dataset.from_dict({"doc": all_premises})

    def get_premises(self, path):
        return [
            p
            for p in self.dep_graph.nodes[path]["premises"]
            if not p["full_name"].startswith("user__.")
        ]

    def num_premises(self, path):
        return len(self.get_premises(path))

    def get_positive_premise(self, full_name: str, def_path: Path, def_pos: List[int]):
        potential_premises = self.get_premises(def_path)
        for p in potential_premises:
            if _between(p["start"], def_pos, p["end"]):
                return {
                    "file": def_path,
                    "full_name": p["full_name"],
                    "code": p["code"],
                }
        logger.warning(f"Unable to find {full_name} in {def_path} at {def_pos}")
        return None

    def sample_negative_premises(
        self, num_negatives: int, def_path: Path, def_pos: List[int]
    ):
        accessible_files = list(self.dep_graph.nodes)

        # TODO: Alternative strategies include hard negatives, sample premises in the current file, etc.
        # accessible_files = nx.descendants(self.dep_graph, def_path)
        # same_file_premises = [p for p in self.get_premises(def_path) if _lt(p["end"], def_pos)]
        # if same_file_premises != []:
        #    accessible_files.add(def_path)
        # accessible_files = list(accessible_files)
        nums = [self.num_premises(p) for p in accessible_files]
        # Sample with replacement.
        sampled_files = random.choices(accessible_files, weights=nums, k=num_negatives)
        negs = []
        for path in sampled_files:
            premises = self.get_premises(path)
            # if path == def_path:
            #    premises = same_file_premises
            p = random.choice(premises)
            negs.append(
                {
                    "file": path,
                    "full_name": p["full_name"],
                    "code": p["code"],
                }
            )

        return negs
