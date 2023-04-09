import os
from loguru import logger
from generator.model import TransformerTacticGenerator
from generator.datamodule import GeneratorDataModule
from common import CLI


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(TransformerTacticGenerator, GeneratorDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
