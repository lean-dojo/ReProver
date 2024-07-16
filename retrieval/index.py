"""Script for indexing the corpus using the retriever.
"""

import torch
import pickle
import argparse
from loguru import logger

from common import IndexedCorpus
from retrieval.model import PremiseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for training the BM25 premise retriever."
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logger.info(args)

    if not torch.cuda.is_available():
        logger.warning("Indexing the corpus using CPU can be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    model = PremiseRetriever.load_hf(args.ckpt_path, 2048, device)
    model.load_corpus(args.corpus_path)
    model.reindex_corpus(batch_size=args.batch_size)

    pickle.dump(
        IndexedCorpus(model.corpus, model.corpus_embeddings.to(torch.float32).cpu()),
        open(args.output_path, "wb"),
    )
    logger.info(f"Indexed corpus saved to {args.output_path}")


if __name__ == "__main__":
    main()
