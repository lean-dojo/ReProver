# BM25 Baseline

First, `cd` into the root of this repo and train BPE tokenizers, one for each data split.
```bash
python retrieval/bm25/train_tokenizer.py --data-path data/leandojo_benchmark_4/random --output-path retrieval/bm25/bpe_tokenizer_lean4_random.json
python retrieval/bm25/train_tokenizer.py --data-path data/leandojo_benchmark_4/novel_premises --output-path retrieval/bm25/bpe_tokenizer_lean4_novel_premises.json
```
Then, perform retrieval using BM25. Each experiment takes several hours, using 32 CPUs by default (configurable via `--num-cpus`). 
```bash
python retrieval/bm25/main.py --tokenizer-path retrieval/bm25/bpe_tokenizer_lean4_random.json --data-path data/leandojo_benchmark_4/random --output-path retrieval/bm25/predictions_lean4_random.pickle
python retrieval/bm25/main.py --tokenizer-path retrieval/bm25/bpe_tokenizer_lean4_novel_premises.json --data-path data/leandojo_benchmark_4/novel_premises --output-path retrieval/bm25/predictions_lean4_novel_premises.pickle
```
Finally, evaluate the predictions using `retrieval/evaluate.py`.
```bash
python retrieval/evaluate.py --data-path data/leandojo_benchmark_4/random --preds-file retrieval/bm25/predictions_lean4_random.pickle
python retrieval/evaluate.py --data-path data/leandojo_benchmark_4/novel_premises --preds-file retrieval/bm25/predictions_lean4_novel_premises.pickle
```
