"""Script to download LeanDojo Benchmark and LeanDojo Benchmark 4 into `./data`."""
import os
import argparse
from hashlib import md5
from loguru import logger


LEANDOJO_BENCHMARK_URL = (
    "https://zenodo.org/records/10114157/files/leandojo_benchmark_v5.tar.gz"
)
LEANDOJO_BENCHMARK_4_URL = (
    "https://zenodo.org/records/10114185/files/leandojo_benchmark_4_v5.tar.gz"
)
DOWNLOADS = {
    LEANDOJO_BENCHMARK_URL: "4b256200618d4668b12a9cfe8c4df4d3",
    LEANDOJO_BENCHMARK_4_URL: "5c1bb0ce1fd1572e8c3d0c3a642e356f",
}


def check_md5(filename: str, gt_hashcode: str) -> bool:
    """
    Check the MD5 of a file against the ground truth.
    """
    if not os.path.exists(filename):
        return False
    # The file could be large.
    # See https://stackoverflow.com/questions/48122798/oserror-errno-22-invalid-argument-when-reading-a-huge-file.
    inp = open(filename, "rb")
    hasher = md5()
    while True:
        block = inp.read(64 * (1 << 20))
        if not block:
            break
        hasher.update(block)
    return hasher.hexdigest() == gt_hashcode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()
    logger.info(args)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    for url, hashcode in DOWNLOADS.items():
        logger.info(f"Downloading {url}")
        path = f"{args.data_path}/{os.path.basename(url)}"
        os.system(f"wget {url} -O {path}")
        if not check_md5(path, hashcode):
            raise RuntimeError(f"MD5 of {path} does not match the ground truth.")

        logger.info(f"Extracting {path}")
        os.system(f"tar -xf {path} -C {args.data_path}")

        logger.info(f"Removing {path}")
        os.remove(path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
