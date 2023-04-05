import os
from loguru import logger
from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule
from common import CLI


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(PremiseRetriever, RetrievalDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
