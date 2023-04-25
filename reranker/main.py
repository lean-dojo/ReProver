import os
from loguru import logger
from reranker.model import PremiseReranker
from pytorch_lightning.cli import LightningCLI
from reranker.datamodule import RerankerDataModule


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(PremiseRetriever, RetrievalDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()