import torch
from torch.utils.data import Dataset


class InstanceDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length,
    ):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return item


class PairDataset(Dataset):
    def __init__(
        self,
        pair_dataframe,
        tokenizer,
        max_length,
    ):
        self.h_texts = (
            pair_dataframe["h_text"]
            .astype(str)
            .tolist()
        )

        self.n_texts = (
            pair_dataframe["n_text"]
            .astype(str)
            .tolist()
        )

        self.h_labels = (
            pair_dataframe["h_label"]
            .astype(int)
            .tolist()
        )

        self.n_labels = (
            pair_dataframe["n_label"]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.h_texts)

    def _encode(self, text):
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

    def __getitem__(self, index):
        h_encoding = self._encode(
            self.h_texts[index]
        )

        n_encoding = self._encode(
            self.n_texts[index]
        )

        item = {}

        for key, value in h_encoding.items():
            item[f"h_{key}"] = value

        for key, value in n_encoding.items():
            item[f"n_{key}"] = value

        item["h_labels"] = torch.tensor(
            self.h_labels[index],
            dtype=torch.long,
        )

        item["n_labels"] = torch.tensor(
            self.n_labels[index],
            dtype=torch.long,
        )

        return item