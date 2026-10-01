import unittest

import pandas as pd
import torch

from src.data.datasets import (
    InstanceDataset,
    PairDataset,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        text,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        self.calls.append(
            {
                "text": text,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )

        base = len(text)

        return {
            "input_ids": torch.tensor(
                [
                    [
                        base,
                        base + 1,
                        base + 2,
                    ]
                ]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1]]
            ),
        }


class DatasetTest(unittest.TestCase):
    def test_instance_dataset_returns_encoded_item_and_label(self):
        dataframe = pd.DataFrame(
            [
                {
                    "text": "pun text",
                    "label": 1,
                },
                {
                    "text": "non pun text",
                    "label": 0,
                },
            ]
        )

        tokenizer = FakeTokenizer()

        dataset = InstanceDataset(
            dataframe,
            tokenizer,
            max_length=128,
        )

        item = dataset[0]

        self.assertEqual(
            len(dataset),
            2,
        )

        self.assertEqual(
            set(item),
            {
                "input_ids",
                "attention_mask",
                "labels",
            },
        )

        self.assertEqual(
            tuple(
                item[
                    "input_ids"
                ].shape
            ),
            (3,),
        )

        self.assertEqual(
            tuple(
                item[
                    "attention_mask"
                ].shape
            ),
            (3,),
        )

        self.assertEqual(
            item[
                "labels"
            ].item(),
            1,
        )

        self.assertEqual(
            item[
                "labels"
            ].dtype,
            torch.long,
        )

        self.assertEqual(
            tokenizer.calls[0],
            {
                "text": "pun text",
                "padding": "max_length",
                "truncation": True,
                "max_length": 128,
                "return_tensors": "pt",
            },
        )

    def test_pair_dataset_returns_prefixed_encodings_and_labels(self):
        pair_dataframe = pd.DataFrame(
            [
                {
                    "h_text": "pun text",
                    "n_text": "non pun text",
                    "h_label": 1,
                    "n_label": 0,
                }
            ]
        )

        tokenizer = FakeTokenizer()

        dataset = PairDataset(
            pair_dataframe,
            tokenizer,
            max_length=64,
        )

        item = dataset[0]

        self.assertEqual(
            len(dataset),
            1,
        )

        self.assertEqual(
            set(item),
            {
                "h_input_ids",
                "h_attention_mask",
                "n_input_ids",
                "n_attention_mask",
                "h_labels",
                "n_labels",
            },
        )

        self.assertEqual(
            tuple(
                item[
                    "h_input_ids"
                ].shape
            ),
            (3,),
        )

        self.assertEqual(
            tuple(
                item[
                    "n_input_ids"
                ].shape
            ),
            (3,),
        )

        self.assertEqual(
            item[
                "h_labels"
            ].item(),
            1,
        )

        self.assertEqual(
            item[
                "n_labels"
            ].item(),
            0,
        )

        self.assertEqual(
            item[
                "h_labels"
            ].dtype,
            torch.long,
        )

        self.assertEqual(
            item[
                "n_labels"
            ].dtype,
            torch.long,
        )

        self.assertEqual(
            [
                call["text"]
                for call in tokenizer.calls
            ],
            [
                "pun text",
                "non pun text",
            ],
        )

        self.assertTrue(
            all(
                call["padding"]
                == "max_length"
                and call["truncation"]
                is True
                and call["max_length"]
                == 64
                and call["return_tensors"]
                == "pt"
                for call
                in tokenizer.calls
            )
        )


if __name__ == "__main__":
    unittest.main()