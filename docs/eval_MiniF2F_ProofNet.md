Evaluation on MiniF2F and ProofNet
----------------------------------

Here we evaluate models trained on LeanDojo Benchmark on MiniF2F or ProofNet. We use MiniF2F as an example, but the same procedure applies to ProofNet.

First, use LeanDojo to extract data from MiniF2F. See the end of [this Jupyter notebook](https://github.com/lean-dojo/LeanDojo/blob/main/scripts/generate-benchmark-lean3.ipynb). Save the dataset to `data/leandojo_minif2f`.


For models without retrieval, run:
```bash
python prover/evaluate.py --data-path data/leandojo_minif2f/default/  --ckpt_path PATH_TO_MODEL_CHECKPOINT --split test --num-cpus 8 --with-gpus
```

For models with retrieval, first use the retriever to index the corpus (pre-computing the embeddings of all premises). 
```bash
python retrieval/index.py --ckpt_path PATH_TO_RETRIEVER_CHECKPOINT --corpus-path data/leandojo_minif2f/corpus.jsonl --output-path PATH_TO_INDEXED_CORPUS
```

Then, run:
```bash
python prover/evaluate.py --data-path data/leandojo_minif2f/default/  --ckpt_path PATH_TO_REPROVER_CHECKPOINT --indexed-corpus-path PATH_TO_INDEXED_CORPUS --split test --num-cpus 8 --with-gpus
```
:warning: `PATH_TO_RETRIEVER_CHECKPOINT` must be the same as the `--model.ret_ckpt_path` argument when training `PATH_TO_REPROVER_CHECKPOINT`.
