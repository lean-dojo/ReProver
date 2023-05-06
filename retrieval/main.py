import os
from loguru import logger
from retrieval.model import PremiseRetriever
from pytorch_lightning.cli import LightningCLI
from retrieval.datamodule import RetrievalDataModule


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")
        parser.link_arguments(
            "data.accessible_premises_only", "model.accessible_premises_only"
        )
        parser.link_arguments("data.corpus_path", "model.corpus_path")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(PremiseRetriever, RetrievalDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
