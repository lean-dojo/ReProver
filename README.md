# retrieval-augmented-prover


## Requirements

* Python dependencies in [retrieval-augmented-prover.yaml](./retrieval-augmented-prover.yaml)
* Append the root of this repo to the `PYTHONPATH` environment variable.
* Download [the dataset](https://drive.google.com/file/d/1ogklwRbaVdXaD9asigc3qh6eD8kfF5S6/view?usp=share_link) and unzip it as `./data/lean_bench`

## Training and Validation

We use [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html) to handle the flags of the training script. For details, run:
```bash
python retrieval/main.py --help
python retrieval/main.py fit --help
```

The default training command uses [this config file](retrieval/confs/cli_default.yaml). It takes ~2 days on one A100 GPU of 80 GB memory. 
```bash
python retrieval/main.py fit --config retrieval/confs/cli_default.yaml
```

Some CLI flags that may be useful:
* `--trainer.devices`: You can use multiple GPUs.
* `--trainer.accumulate_grad_batches` and `--data.batch_size`: If you have smaller memory, use a small `batch_size` and a large `accumulate_grad_batches` to keep the same effective batch size.


Caveats:
* You may use a different pretrained model via `--model.model_name`. Currently, we use `google/by-t5` because they are tokenization-free. For other models, the gap in tokenization may become a problem.
* We use `--trainer.precision: bf16` because [fp16 doesn't work well with pretrained T5](https://github.com/huggingface/transformers/issues/10830).