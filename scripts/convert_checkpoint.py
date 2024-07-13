import torch
import argparse
from loguru import logger

from generation.model import RetrievalAugmentedGenerator
from retrieval.model import PremiseRetriever


def convert(model_type: str, src: str, dst: str) -> None:
    device = torch.device("cpu")
    if model_type == "generator":
        model = RetrievalAugmentedGenerator.load(src, device, freeze=True)
        model.generator.save_pretrained(dst)
    else:
        assert model_type == "retriever"
        model = PremiseRetriever.load(src, device, freeze=True)
        model.encoder.save_pretrained(dst)
    model.tokenizer.save_pretrained(dst)
    

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, choices=["generator", "retriever"])
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()
    logger.info(args)

    logger.info(f"Loading the model from {args.src}")
    convert(args.model_type, args.src, args.dst)
    logger.info(f"The model saved in Hugging Face format to {args.dst}")


if __name__ == "__main__":
    main()