from tqdm import tqdm
from pathlib import Path
import pytorch_lightning as pl
from transformers import ByT5Tokenizer
from common import *
from torch.utils.data import DataLoader
from typing import Optional, Union


class RetrievalDataset(Dataset):  # type: ignore
    def __init__(
        self,
        data_path: Path,
        corpus: Corpus,
        num_negatives: int,
        max_seq_len: int,
        tokenizer: str,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.num_negatives = num_negatives
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.is_train = is_train

        self.data = []
        num_discarded = 0
        logger.info(f"Loading data from {data_path}")

        for thm in tqdm(json.load(data_path.open())):
            repo_name = thm["url"].split("/")[-1]
            file_path = Path(repo_name) / thm["file_path"]
            deps = self.corpus.get_dependencies(file_path)

            for tac in thm["traced_tactics"]:
                annot_tac, provenances = tac["annotated_tactic"]
                marks = list(re.finditer(r"(?<=<a>).+?(?=</a>)", annot_tac))

                for m, prov in zip(marks, provenances):
                    def_path = Path(prov["def_path"])
                    assert def_path == file_path or def_path in deps
                    pos_premise = self.corpus.locate_premise(
                        def_path, Pos(*prov["def_pos"])
                    )
                    if pos_premise is None:
                        num_discarded += 1
                    else:
                        tactic_prefix = annot_tac[: m.start()]
                        context = Context(
                            file_path,
                            thm["full_name"],
                            Pos(*thm["start"]),
                            tactic_prefix,
                            tac["state_before"],
                        )
                        self.data.append(
                            {
                                "context": context,
                                "tactic_arg": m.group(),
                                "pos_premise": pos_premise,
                            }
                        )

        logger.info(
            f"{len(self.data)} examples remain after discarding {num_discarded} examples"
        )
        assert num_discarded / len(self.data) < 0.01

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        ex = copy(self.data[idx])

        if self.is_train:
            premises = [
                p
                for p in self.corpus.iter_accessible_premises(
                    ex["context"].path, ex["context"].theorem_pos
                )
                if p != ex["pos_premise"]
            ]
            neg_premises = random.sample(premises, self.num_negatives)
            ex["neg_premises"] = [p for p in neg_premises]

        return ex

    def collate(self, examples):
        batch = {}

        context = [ex["context"] for ex in examples]
        tokenized_context = self.tokenizer(
            [c.serialize() for c in context],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        pos_premise = [ex["pos_premise"] for ex in examples]
        tokenized_pos_premise = self.tokenizer(
            [p.serialize() for p in pos_premise],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        batch["context"] = context
        batch["context_ids"] = tokenized_context.input_ids
        batch["context_mask"] = tokenized_context.attention_mask
        batch["pos_premise"] = pos_premise
        batch["pos_premise_ids"] = tokenized_pos_premise.input_ids
        batch["pos_premise_mask"] = tokenized_pos_premise.attention_mask

        if self.is_train:
            batch_size = len(examples)
            label = torch.zeros(batch_size, batch_size * (1 + self.num_negatives))

            # Check if one's negative is another's positive
            for j in range(batch_size):
                pos_premise = examples[j]["pos_premise"]
                for k in range(batch_size * (1 + self.num_negatives)):
                    if k < batch_size:
                        label[j, k] = float(pos_premise == examples[k]["pos_premise"])
                    else:
                        label[j, k] = float(
                            pos_premise
                            == examples[k % batch_size]["neg_premises"][
                                k // batch_size - 1
                            ]
                        )

            batch["label"] = label
            batch["neg_premises"] = []
            batch["neg_premises_ids"] = []
            batch["neg_premises_mask"] = []

            for i in range(self.num_negatives):
                neg_premise = [ex["neg_premises"][i] for ex in examples]
                tokenized_neg_premise = self.tokenizer(
                    [p.serialize() for p in neg_premise],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                batch["neg_premises"].append(neg_premise)
                batch["neg_premises_ids"].append(tokenized_neg_premise.input_ids)
                batch["neg_premises_mask"].append(tokenized_neg_premise.attention_mask)

        for k in examples[0].keys():
            if k not in ("context", "pos_premise", "neg_premises"):
                batch[k] = [ex[k] for ex in examples]

        return batch


class RetrievalDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        data_path: Union[str, Path],
        corpus_path: Union[str, Path],
        num_negatives: int,
        model_name: str,
        batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if isinstance(corpus_path, str):
            corpus_path = Path(corpus_path)

        self.data_path = data_path
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
        self.corpus = Corpus(corpus_path)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RetrievalDataset(
                self.data_path / "train.json",
                self.corpus,
                self.num_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = RetrievalDataset(
                self.data_path / "val.json",
                self.corpus,
                self.num_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

        if stage in (None, "test"):
            self.ds_test = RetrievalDataset(
                self.data_path / "test.json",
                self.corpus,
                self.num_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

    def train_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_train,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_val,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.ds_test,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_test.collate,
            pin_memory=True,
            drop_last=False,
        )


if __name__ == "__main__":
    dm = RetrievalDataModule(
        data_path="data/lean_bench/random/",
        corpus_path="data/lean_bench/corpus.jsonl",
        num_negatives=3,
        model_name="google/byt5-small",
        batch_size=8,
        max_seq_len=1024,
        num_workers=8,
    )
    dm.prepare_data()
    dm.setup("fit")

    # max_len = 0
    for i, data_batch in tqdm(
        enumerate(dm.train_dataloader()), total=len(dm.train_dataloader())
    ):
        if i == 0:
            print(data_batch)
        # max_len = max(max_len, data_batch["pos_premise_ids"].size(1))
        print("pos: ", data_batch["pos_premise_ids"].size())
        # for ids in data_batch["neg_premises_ids"]:
        #    max_len = max(max_len, ids.size(1))
        #    print("neg:", ids.size())

    for i, data_batch in tqdm(enumerate(dm.val_dataloader())):
        if i == 0:
            print(data_batch)

    for i, data_batch in tqdm(enumerate(dm.test_dataloader())):
        if i == 0:
            print(data_batch)

    # print(max_len)
