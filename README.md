# Retrieval-Augmented Prover (ReProver)

![Model](images/ReProver.jpg)

Code for the paper:  

[LeanDojo: Theorem Proving with Retrieval-Augmented Language Models](https://leandojo.org/)      
NeurIPS (Datasets and Benchmarks Track), 2023, Oral presentation  
[Kaiyu Yang](https://yangky11.github.io/), [Aidan Swope](https://aidanswope.com/about), [Alex Gu](https://minimario.github.io/), [Rahul Chalamala](https://rchalamala.github.io/),  
[Peiyang Song](https://peiyang-song.github.io/), [Shixing Yu](https://billysx.github.io/), [Saad Godil](https://www.linkedin.com/in/saad-godil-9728353/), [Ryan Prenger](https://www.linkedin.com/in/ryan-prenger-18797ba1/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/)

```bibtex
@inproceedings{yang2023leandojo,
  title={{LeanDojo}: Theorem Proving with Retrieval-Augmented Language Models},
  author={Yang, Kaiyu and Swope, Aidan and Gu, Alex and Chalamala, Rahul and Song, Peiyang and Yu, Shixing and Godil, Saad and Prenger, Ryan and Anandkumar, Anima},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

Lean 3 is deprecated. The `main` branch only supports Lean 4. You may use the [`legacy`](https://github.com/lean-dojo/ReProver/tree/legacy) branch if you want to work with Lean 3.

[![GitHub license](https://img.shields.io/github/license/MineDojo/MineDojo)](https://github.com/MineDojo/MineDojo/blob/main/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Quick Links

  - [LeanDojo Website](https://leandojo.org/)
  - [Using Trained Models on Hugging Face](#using-trained-models-on-hugging-face)
  - [Using the Model Directly in Lean](#using-the-model-directly-in-lean)
  - [Requirements](#requirements)
  - [Premise Selection](#premise-selection)
  - [Theorem Proving](#theorem-proving)
  - [Questions and Bugs](#questions-and-bugs)


## Using Trained Models on Hugging Face

| Model name | Model architecture | Input | Output |
| ---------- | ------------------ | ----- | ------ |
| [kaiyuy/leandojo-lean4-tacgen-byt5-small](https://huggingface.co/kaiyuy/leandojo-lean4-tacgen-byt5-small) | ByT5 (encoder-decoder) | Proof state | Tactic |
| [kaiyuy/leandojo-lean4-retriever-byt5-small](https://huggingface.co/kaiyuy/leandojo-lean4-retriever-byt5-small) | ByT5 (encoder-only) | Proof state | Embedding |
| [kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small](https://huggingface.co/kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small) | ByT5 (encoder-decoder) | Retrieved premises + proof state | Tactic |

Our trained models are available on HuggingFace Hub. With minimum dependencies (only [PyTorch](https://pytorch.org/) and [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)), you can use our models to perform inference, finetune them on your own data, or plug them into your customized theorem proving pipeline. Below are some examples.


### Tactic Generator

Our tactic generator is a [ByT5](https://huggingface.co/docs/transformers/model_doc/byt5) model finetuned to generate tactics given a proof state.
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-tacgen-byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean4-tacgen-byt5-small")

state = "n : ℕ\n⊢ gcd n n = n"
tokenized_state = tokenizer(state, return_tensors="pt")

# Generate a single tactic.
tactic_ids = model.generate(tokenized_state.input_ids, max_length=1024)
tactic = tokenizer.decode(tactic_ids[0], skip_special_tokens=True)
print(tactic, end="\n\n")

# Generate multiple tactics via beam search.
tactic_candidates_ids = model.generate(
    tokenized_state.input_ids,
    max_length=1024,
    num_beams=4,
    length_penalty=0.0,
    do_sample=False,
    num_return_sequences=4,
    early_stopping=False,
)
tactic_candidates = tokenizer.batch_decode(
    tactic_candidates_ids, skip_special_tokens=True
)
for tac in tactic_candidates:
    print(tac)
```

The expected output is shown below. `<a>` and `</a>` are markers of premises in generated tactics. You should remove them when using the tactics.
```lean
rw [gcd_comm, gcd_rec n n]

simp [gcd]
apply Nat.dvd_antisymm
induction' n with n n_ih
induction' n with n hn
```


### Premise Retriever

At the core of our premise retriever is a ByT5 encoder that embeds states and premises into vectors. You can 
use the vectors to perform retrieval by maximizing cosine similarity.
```python
import torch
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForTextEncoding

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
model = AutoModelForTextEncoding.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")

state = "n : ℕ\n⊢ gcd n n = n"
premises = [
  "<a>vsub_eq_zero_iff_eq</a> @[simp] lemma vsub_eq_zero_iff_eq {p1 p2 : P} : p1 -ᵥ p2 = (0 : G) ↔ p1 = p2",
  "<a>is_scalar_tower.coe_to_alg_hom'</a> @[simp] lemma coe_to_alg_hom' : (to_alg_hom R S A : S → A) = algebra_map S A",
  "<a>polynomial.X_sub_C_ne_zero</a> theorem X_sub_C_ne_zero (r : R) : X - C r ≠ 0",
  "<a>forall_true_iff</a> theorem forall_true_iff : (α → true) ↔ true",
  "def <a>Nat.gcd</a> : Nat → Nat → Nat\n| 0        y := y\n| (succ x) y := have y % succ x < succ x, from mod_lt _ $ succ_pos _,\n                gcd (y % succ x) (succ x)",
  "@[simp] theorem <a>Nat.gcd_zero_left</a> (x : Nat) : gcd 0 x = x",
  "@[simp] theorem <a>Nat.gcd_succ</a> (x y : Nat) : gcd (succ x) y = gcd (y % succ x) (succ x)",
  "@[simp] theorem <a>Nat.mod_self</a> (n : Nat) : n % n = 0",
]  # A corpus of premises to retrieve from.

@torch.no_grad()
def encode(s: Union[str, List[str]]) -> torch.Tensor:
    """Encode texts into feature vectors."""
    if isinstance(s, str):
        s = [s]
        should_squeeze = True
    else:
        should_squeeze = False
    tokenized_s = tokenizer(s, return_tensors="pt", padding=True)
    hidden_state = model(tokenized_s.input_ids).last_hidden_state
    lens = tokenized_s.attention_mask.sum(dim=1)
    features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
    if should_squeeze:
      features = features.squeeze()
    return features

@torch.no_grad()
def retrieve(state: str, premises: List[str], k: int) -> List[str]:
    """Retrieve the top-k premises given a state."""
    state_emb = encode(state)
    premise_embs = encode(premises)
    scores = (state_emb @ premise_embs.T)
    topk = scores.topk(k).indices.tolist()
    return [premises[i] for i in topk]

for p in retrieve(state, premises, k=4):
    print(p, end="\n\n")
```

Expected output:
```lean
def <a>Nat.gcd</a> : Nat → Nat → Nat
| 0        y := y
| (succ x) y := have y % succ x < succ x, from mod_lt _ $ succ_pos _,
                gcd (y % succ x) (succ x)

@[simp] theorem <a>Nat.gcd_zero_left</a> (x : Nat) : gcd 0 x = x

@[simp] theorem <a>Nat.gcd_succ</a> (x y : Nat) : gcd (succ x) y = gcd (y % succ x) (succ x)

@[simp] theorem <a>Nat.mod_self</a> (n : Nat) : n % n = 0
```


### Retrieval-Augmented Tactic Generator

ReProver's tactic generator takes as input the concatenation of retrieved premises and the state.
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small")

state = "n : ℕ\n⊢ gcd n n = n"
retrieved_premises = [
  "def <a>Nat.gcd</a> : Nat → Nat → Nat\n| 0        y := y\n| (succ x) y := have y % succ x < succ x, from mod_lt _ $ succ_pos _,\n                gcd (y % succ x) (succ x)",
  "@[simp] theorem <a>Nat.mod_self</a> (n : Nat) : n % n = 0",
]
input = "\n\n".join(retrieved_premises + [state])
print("------ INPUT ------\n", input)
tokenized_input = tokenizer(input, return_tensors="pt", max_length=2300, truncation=True)

# Generate a single tactic.
tactic_ids = model.generate(tokenized_input.input_ids, max_length=1024)
tactic = tokenizer.decode(tactic_ids[0], skip_special_tokens=True)
print("\n------ OUTPUT ------")
print(tactic, end="\n\n")

# Generate multiple tactics via beam search.
tactic_candidates_ids = model.generate(
    tokenized_input.input_ids,
    max_length=1024,
    num_beams=4,
    length_penalty=0.0,
    do_sample=False,
    num_return_sequences=4,
    early_stopping=False,
)
tactic_candidates = tokenizer.batch_decode(
    tactic_candidates_ids, skip_special_tokens=True
)
for tac in tactic_candidates:
    print(tac)
```

Expected output:
```
------ INPUT ------
 def <a>Nat.gcd</a> : Nat → Nat → Nat
| 0        y := y
| (succ x) y := have y % succ x < succ x, from mod_lt _ $ succ_pos _,
                gcd (y % succ x) (succ x)

@[simp] theorem <a>Nat.mod_self</a> (n : Nat) : n % n = 0

n : ℕ
⊢ gcd n n = n

------ OUTPUT ------
rw [gcd_def, ← gcd_def, ← gcd_def, ← gcd_def]

simp [gcd]
rw [gcd]
rw [gcd_def]
rw [← Nat.mod_self n, ← Nat.mod_self n]
```

**The rest of this document describes our system for training and evaluating LLM-based provers.**


## Using the Model Directly in Lean

Check out [Lean Copilot](https://github.com/lean-dojo/LeanCopilot) if you want to run ReProver's tactic generator directly in Lean's VSCode workflow. 


## Requirements

1. Download and install [Miniconda Python 3](https://docs.conda.io/en/latest/miniconda.html) (Anaconda should also work).
2. Create the conda environment and install Python dependencies:
```bash
conda create --yes --name ReProver python=3.11 ipython
conda activate ReProver
pip install torch  # Depending on your CUDA version; see https://pytorch.org/.
pip install tqdm loguru deepspeed "pytorch-lightning[extra]" transformers wandb openai rank_bm25 lean-dojo vllm
```
3. Prepend the repo's root to the `PYTHONPATH` environment variable.
4. Make sure `wget` and `tar` are available. Then, run `python scripts/download_data.py` to download [LeanDojo Benchmark 4](https://zenodo.org/doi/10.5281/zenodo.8040109). They will be saved to `./data`.
5. Satisfy the requirements of [LeanDojo](https://github.com/lean-dojo/LeanDojo#requirements).
6. Use [LeanDojo](https://github.com/lean-dojo/LeanDojo) to trace all repos in the datasets: `python scripts/trace_repos.py`. This step may take some time. Please refer to [LeanDojo's documentation](https://leandojo.readthedocs.io/en/latest/) if you encounter any issues.
7. Run `wandb login` to log in Weights & Biases.



## Premise Selection

We use [Lightning CLI](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html) to create [retrieval/main.py](retrieval/main.py) for training, validation, and testing the premise retrieval. It takes command line arguments as well as YAML config files. Please run `python retrieval/main.py --help` or refer to the documentation of [Lightning CLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) for details. 

The config files for our experiments are in [./retrieval/confs](./retrieval/confs). We train all models on a single NVIDIA A100 GPU with 80GB memory. When using GPUs with smaller memory, you can change `batch_size`, `accumulate_grad_batches`, and `num_negatives`. However, it may impact the performance due to in-batch negatives in DPR.


### Training the Premise Retriever

Run `python retrieval/main.py fit --help` to see how to use the training script. For example:
```bash
mkdir logs  # Create the directory for training logs.
python retrieval/main.py fit --config retrieval/confs/cli_lean4_random.yaml --trainer.logger.name train_retriever_random --trainer.logger.save_dir logs/train_retriever_random         # Train the retriever on the `random` split of LeanDojo Benchmark 4.
python retrieval/main.py fit --config retrieval/confs/cli_lean4_novel_premises.yaml --trainer.logger.name train_retriever_novel_premises --trainer.logger.save_dir logs/train_retriever_novel_premises  # Train the retriever on the `novel_premises` split of LeanDojo Benchmark 4.
```
Hyperparameters and model checkpoints are saved in `./logs/train_retriever_*`, and you can monitor the training process on Weights & Biases.


### Retrieving Premises for All Proof States

After the models are trained, run the following commands to retrieve premises for all proof states in the dataset.
```bash
python retrieval/main.py predict --config retrieval/confs/cli_lean4_random.yaml --ckpt_path $PATH_TO_RETRIEVER_CHECKPOINT --trainer.logger.name predict_retriever_random --trainer.logger.save_dir logs/predict_retriever_random 
python retrieval/main.py predict --config retrieval/confs/cli_lean4_novel_premises.yaml --ckpt_path $PATH_TO_RETRIEVER_CHECKPOINT --trainer.logger.name predict_retriever_novel_premises --trainer.logger.save_dir logs/predict_retriever_novel_premises
```
, where `PATH_TO_RETRIEVER_CHECKPOINT` is the model checkpoint produced in the previous step. Retrieved premises are saved to `./logs/predict_retriever_*/predictions.pickle`.


### Evaluating the Retrieved Premises

After predictions are saved, evaluate them using metrics such as R@1, R@10, and MRR.
```bash
python retrieval/evaluate.py --data-path data/leandojo_benchmark_4/random --preds-file logs/predict_retriever_random/predictions.pickle
python retrieval/evaluate.py --data-path data/leandojo_benchmark_4/novel_premises --preds-file logs/predict_retriever_novel_premises/predictions.pickle
```


## Theorem Proving


### Training the Tactic Generator

Similar to premise selection, you can run `python generation/main.py --help` and  `python generation/main.py fit --help` to check the command line options.

To train tactic generators without retrieval:
```bash
python generation/main.py fit --config generation/confs/cli_lean4_random.yaml --trainer.logger.name train_generator_random --trainer.logger.save_dir logs/train_generator_random            # LeanDojo Benchmark 4, `random` split
python generation/main.py fit --config generation/confs/cli_lean4_novel_premises.yaml --trainer.logger.name train_generator_novel_premises --trainer.logger.save_dir logs/train_generator_novel_premises    # LeanDojo Benchmark 4, `novel_premises` split
```
Hyperparameters and model checkpoints are saved in `./logs/train_generator_*`, and you can monitor the training process on Weights & Biases.

To train models augmented by retrieval, we need to provide a retriever checkpoint and its predictions on all proof states in the dataset:
```bash 
python generation/main.py fit --config generation/confs/cli_lean4_random.yaml --model.ret_ckpt_path $PATH_TO_RETRIEVER_CHECKPOINT --data.preds_path logs/predict_retriever_random/predictions.pickle --trainer.logger.name train_reprover_random --trainer.logger.save_dir logs/train_reprover_random
python generation/main.py fit --config generation/confs/cli_lean4_novel_premises.yaml --model.ret_ckpt_path $PATH_TO_RETRIEVER_CHECKPOINT --data.preds_path logs/predict_retriever_novel_premises/predictions.pickle --trainer.logger.name train_reprover_novel_premises --trainer.logger.save_dir logs/train_reprover_novel_premises
```


### Theorem Proving Evaluation on LeanDojo Benchmark

After the tactic generator is trained, we combine it with best-first search to prove theorems by interacting with Lean. The evaluation script takes Hugging Face model checkpoints (either local or remote) as input. For remote models, you can simply use their names, e.g., [kaiyuy/leandojo-lean4-tacgen-byt5-small](https://huggingface.co/kaiyuy/leandojo-lean4-tacgen-byt5-small). For locally trained models, you first need to convert them from PyTorch Ligthning checkpoints to Hugging Face checkpoints:
```bash
python scripts/convert_checkpoint.py generator --src $PATH_TO_GENERATOR_CHECKPOINT --dst ./leandojo-lean4-tacgen-byt5-small
python scripts/convert_checkpoint.py retriever --src $PATH_TO_RETRIEVER_CHECKPOINT --dst ./leandojo-lean4-retriever-byt5-small
python scripts/convert_checkpoint.py generator --src $PATH_TO_REPROVER_CHECKPOINT --dst ./leandojo-lean4-retriever-tacgen-byt5-small
```
, where `PATH_TO_GENERATOR_CHECKPOINT` and `PATH_TO_RETRIEVER_CHECKPOINT` are PyTorch Ligthning checkpoints produced by the training script.


To evaluate the model without retrieval, run (using the `random` data split as example):
```bash
python prover/evaluate.py --data-path data/leandojo_benchmark_4/random/ --gen_ckpt_path ./leandojo-lean4-tacgen-byt5-small --split test --num-workers 5 --num-gpus 1
```
You may tweak `--num-workers` and `--num-gpus` to fit your hardware.


For the model with retrieval, first use the retriever to index the corpus (pre-computing the embeddings of all premises):
```bash
python retrieval/index.py --ckpt_path ./leandojo-lean4-retriever-byt5-small --corpus-path data/leandojo_benchmark_4/corpus.jsonl --output-path $PATH_TO_INDEXED_CORPUS
```
It saves the indexed corpus as a pickle file to `PATH_TO_INDEXED_CORPUS`.

Then, run:
```bash
python prover/evaluate.py --data-path data/leandojo_benchmark_4/random/  --gen_ckpt_path ./leandojo-lean4-retriever-tacgen-byt5-small --ret_ckpt_path ./leandojo-lean4-retriever-byt5-small --indexed-corpus-path $PATH_TO_INDEXED_CORPUS --split test --num-workers 5 --num-gpus 1
```


## Questions and Bugs

* For general questions and discussions, please use [GitHub Discussions](https://github.com/lean-dojo/ReProver/discussions).  
* To report a potential bug, please open an issue.
