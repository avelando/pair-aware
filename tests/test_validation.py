import unittest
from unittest.mock import patch

import pandas as pd

from src.data.validation import (
    validate_input_splits,
    validate_split,
)


EXPECTED_SPLIT_COUNTS = {
    "train": 4,
    "validation": 4,
    "test": 4,
}

EXPECTED_PAIR_COUNTS = {
    "train": 2,
    "validation": 2,
    "test": 2,
}

EXPECTED_CLASS_COUNTS = {
    "train": {
        0: 2,
        1: 2,
    },
    "validation": {
        0: 2,
        1: 2,
    },
    "test": {
        0: 2,
        1: 2,
    },
}


def build_split(prefix):
    rows = []

    for pair_id in range(1, 3):
        rows.extend(
            [
                {
                    "id": f"{prefix}-{pair_id}.H",
                    "text": f"pun {prefix} {pair_id}",
                    "label": 1,
                    "tokens": ["pun", str(pair_id)],
                    "labels": [1, 0],
                },
                {
                    "id": f"{prefix}-{pair_id}.N",
                    "text": f"non pun {prefix} {pair_id}",
                    "label": 0,
                    "tokens": ["non", "pun", str(pair_id)],
                    "labels": [0, 0, 0],
                },
            ]
        )

    return pd.DataFrame(rows)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.config_patcher = patch.multiple(
            "src.data.validation",
            EXPECTED_SPLIT_COUNTS=EXPECTED_SPLIT_COUNTS,
            EXPECTED_PAIR_COUNTS=EXPECTED_PAIR_COUNTS,
            EXPECTED_CLASS_COUNTS=EXPECTED_CLASS_COUNTS,
        )
        self.config_patcher.start()
        self.addCleanup(self.config_patcher.stop)
        self.data = build_split("train")

    def test_valid_split_returns_expected_summary(self):
        result = validate_split(
            self.data,
            "train",
            "test-run",
        )

        self.assertEqual(result["example_count"], 4)
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["class_counts"], {0: 2, 1: 2})
        self.assertEqual(len(result["ids"]), 4)
        self.assertEqual(len(result["pair_ids"]), 2)

    def test_missing_required_column_is_rejected(self):
        data = self.data.drop(columns=["tokens"])

        with self.assertRaisesRegex(
            ValueError,
            "missing columns",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_wrong_split_size_is_rejected(self):
        data = self.data.iloc[:-1].copy()

        with self.assertRaisesRegex(
            ValueError,
            "expected 4 examples",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_duplicated_id_is_rejected(self):
        data = self.data.copy()
        data.loc[1, "id"] = data.loc[0, "id"]

        with self.assertRaisesRegex(
            ValueError,
            "duplicated IDs",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_non_string_id_is_rejected(self):
        data = self.data.copy()
        data.loc[0, "id"] = 1

        with self.assertRaisesRegex(
            ValueError,
            "all IDs must be strings",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_invalid_record_id_is_rejected(self):
        data = self.data.copy()
        data.loc[0, "id"] = "invalid"

        with self.assertRaisesRegex(
            ValueError,
            "Invalid record ID format",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_non_binary_class_label_is_rejected(self):
        data = self.data.copy()
        data.loc[0, "label"] = 2

        with self.assertRaisesRegex(
            ValueError,
            "labels must be binary",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_wrong_class_distribution_is_rejected(self):
        data = self.data.copy()
        data.loc[1, "label"] = 1

        with self.assertRaisesRegex(
            ValueError,
            "expected class distribution",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_empty_text_is_rejected(self):
        data = self.data.copy()
        data.loc[0, "text"] = "   "

        with self.assertRaisesRegex(
            ValueError,
            "invalid text",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_suffix_label_mismatch_is_rejected(self):
        data = self.data.copy()
        data.loc[0, "label"] = 0
        data.loc[1, "label"] = 1

        with self.assertRaisesRegex(
            ValueError,
            "suffix and label are inconsistent",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_invalid_tokens_type_is_rejected(self):
        data = self.data.copy()
        data.at[0, "tokens"] = "pun"

        with self.assertRaisesRegex(
            ValueError,
            "tokens must be a list",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_invalid_token_labels_type_is_rejected(self):
        data = self.data.copy()
        data.at[0, "labels"] = "1 0"

        with self.assertRaisesRegex(
            ValueError,
            "token labels must be a list",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_token_length_mismatch_is_rejected(self):
        data = self.data.copy()
        data.at[0, "labels"] = [1]

        with self.assertRaisesRegex(
            ValueError,
            "token and token-label lengths differ",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_non_binary_token_label_is_rejected(self):
        data = self.data.copy()
        data.at[0, "labels"] = [2, 0]

        with self.assertRaisesRegex(
            ValueError,
            "token labels must be binary",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_pun_without_positive_token_is_rejected(self):
        data = self.data.copy()
        data.at[0, "labels"] = [0, 0]

        with self.assertRaisesRegex(
            ValueError,
            "pun instance has no positive token label",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_non_pun_with_positive_token_is_rejected(self):
        data = self.data.copy()
        data.at[1, "labels"] = [1, 0, 0]

        with self.assertRaisesRegex(
            ValueError,
            "non-pun instance has positive token labels",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_incomplete_pair_is_rejected(self):
        data = self.data.copy()
        data.loc[1, "id"] = "3.N"

        with self.assertRaisesRegex(
            ValueError,
            "invalid H/N pairs found",
        ):
            validate_split(
                data,
                "train",
                "test-run",
            )

    def test_missing_split_is_rejected(self):
        split_data = {
            "train": build_split("train"),
            "validation": build_split("validation"),
        }

        with self.assertRaisesRegex(
            ValueError,
            "missing splits",
        ):
            validate_input_splits(
                split_data,
                "test-run",
            )

    def test_record_overlap_across_splits_is_rejected(self):
        train = build_split("shared")
        validation = build_split("shared")
        test = build_split("test")

        with self.assertRaisesRegex(
            ValueError,
            "record IDs overlap across splits",
        ):
            validate_input_splits(
                {
                    "train": train,
                    "validation": validation,
                    "test": test,
                },
                "test-run",
            )

    def test_pair_overlap_across_splits_is_rejected(self):
        validated = [
            {
                "ids": {"train-1.H", "train-1.N"},
                "pair_ids": {"shared"},
                "class_counts": {0: 1, 1: 1},
                "example_count": 2,
                "pair_count": 1,
            },
            {
                "ids": {"validation-1.H", "validation-1.N"},
                "pair_ids": {"shared"},
                "class_counts": {0: 1, 1: 1},
                "example_count": 2,
                "pair_count": 1,
            },
            {
                "ids": {"test-1.H", "test-1.N"},
                "pair_ids": {"test"},
                "class_counts": {0: 1, 1: 1},
                "example_count": 2,
                "pair_count": 1,
            },
        ]

        with patch(
            "src.data.validation.validate_split",
            side_effect=validated,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "pair IDs overlap across splits",
            ):
                validate_input_splits(
                    {
                        "train": self.data,
                        "validation": self.data,
                        "test": self.data,
                    },
                    "test-run",
                )

    def test_valid_input_splits_return_all_summaries(self):
        result = validate_input_splits(
            {
                "train": build_split("train"),
                "validation": build_split("validation"),
                "test": build_split("test"),
            },
            "test-run",
        )

        self.assertEqual(
            set(result),
            {"train", "validation", "test"},
        )
        self.assertEqual(result["train"]["pair_count"], 2)
        self.assertEqual(result["validation"]["pair_count"], 2)
        self.assertEqual(result["test"]["pair_count"], 2)


if __name__ == "__main__":
    unittest.main()