import pdb
import csv
import json
import random

from common import format_state, format_tactic


def main() -> None:
    pairs = []
    data_path = "../data/leandojo_benchmark_4/random/train.json"

    for thm in json.load(open(data_path)):
        for tac in thm["traced_tactics"]:
            if "annotated_tactic" in tac:
                tactic = format_tactic(*tac["annotated_tactic"], normalize=True)
            else:
                tactic = format_tactic(tac["tactic"], [], normalize=True)
            pairs.append({"state": format_state(tac["state_before"]), "output": tactic})

    random.shuffle(pairs)

    with open("state_tactic_pairs.csv", "wt") as oup:
        wt = csv.DictWriter(oup, fieldnames=["state", "output"])
        wt.writeheader()
        for st in pairs:
            wt.writerow(st)


if __name__ == "__main__":
    main()
