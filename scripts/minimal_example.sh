#!/usr/bin/sh
git clone git@github.com:lean-dojo/ReProver.git
cd ReProver
export PYTHONPATH=$(pwd):$PYTHONPATH

conda create --yes --name tmp python=3.11 ipython
conda activate tmp
pip install torch deepspeed "pytorch-lightning[extra]" transformers wandb tqdm loguru openai rank_bm25 vllm lean-dojo

python scripts/download_data.py
python scripts/trace_repos.py

python prover/evaluate.py --data-path data/leandojo_benchmark_4/random/ --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small --split test --num-workers 1 --num-gpus 1 --timeout 60 --num-theorems 1 --verbose
# python prover/evaluate.py --data-path data/leandojo_benchmark_4/random/ --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small --split test --num-workers 40 --num-gpus 8