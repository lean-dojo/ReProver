import json
import argparse
from glob import glob
from loguru import logger
from lean_dojo import LeanGitRepo, trace, is_available_in_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()
    logger.info(args)

    repos = set()

    for path in glob(f"{args.data_path}/*/*/*.json"):
        data = json.load(open(path))
        for ex in data:
            repo = LeanGitRepo(ex["url"], ex["commit"])
            if not is_available_in_cache(repo):
                repos.add(repo)

    logger.info(f"Repos to trace: {repos}")

    for repo in repos:
        trace(repo)


if __name__ == "__main__":
    main()
