import unittest

import pandas as pd

from src.data.pairing import (
    _random_derangement,
    build_pairs,
    validate_pair_table,
)


class PairingTest(unittest.TestCase):
    def setUp(self):
        rows = []

        for pair_id in range(1, 9):
            rows.extend(
                [
                    {
                        "id": f"{pair_id}.H",
                        "text": f"pun {pair_id}",
                        "label": 1,
                    },
                    {
                        "id": f"{pair_id}.N",
                        "text": f"non pun {pair_id}",
                        "label": 0,
                    },
                ]
            )

        self.data = pd.DataFrame(rows)

    def test_true_pairs(self):
        pairs = build_pairs(
            self.data,
            "true_pair",
        )

        result = validate_pair_table(
            pairs,
            expected_true_pairs=True,
        )

        self.assertEqual(result["pair_count"], 8)
        self.assertEqual(result["true_matches"], 8)
        self.assertEqual(result["unique_h"], 8)
        self.assertEqual(result["unique_n"], 8)
        self.assertEqual(
            set(pairs["h_id"]),
            set(self.data.loc[self.data["label"] == 1, "id"]),
        )
        self.assertEqual(
            set(pairs["n_id"]),
            set(self.data.loc[self.data["label"] == 0, "id"]),
        )

    def test_shuffled_pairs_are_deterministic_derangements(self):
        first = build_pairs(
            self.data,
            "shuffled_pair",
            seed=13,
        )

        second = build_pairs(
            self.data,
            "shuffled_pair",
            seed=13,
        )

        self.assertTrue(first.equals(second))
        self.assertEqual(
            int(
                (
                    first["h_pair_id"]
                    == first["n_pair_id"]
                ).sum()
            ),
            0,
        )
        self.assertEqual(first["h_id"].nunique(), 8)
        self.assertEqual(first["n_id"].nunique(), 8)
        self.assertEqual(
            set(first["h_id"]),
            set(self.data.loc[self.data["label"] == 1, "id"]),
        )
        self.assertEqual(
            set(first["n_id"]),
            set(self.data.loc[self.data["label"] == 0, "id"]),
        )

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown pairing strategy",
        ):
            build_pairs(
                self.data,
                "invalid",
            )

    def test_shuffled_pair_requires_seed(self):
        with self.assertRaisesRegex(
            ValueError,
            "A seed is required",
        ):
            build_pairs(
                self.data,
                "shuffled_pair",
            )

    def test_single_pair_cannot_be_deranged(self):
        data = self.data.iloc[:2].copy()

        with self.assertRaisesRegex(
            ValueError,
            "At least two pairs are required",
        ):
            build_pairs(
                data,
                "shuffled_pair",
                seed=13,
            )

    def test_duplicate_derangement_values_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Derangement values must be unique",
        ):
            _random_derangement(
                ["1", "1", "2"],
                seed=13,
            )

    def test_mismatched_pair_ids_are_rejected(self):
        data = self.data.copy()
        data.loc[data["id"] == "8.N", "id"] = "9.N"

        with self.assertRaisesRegex(
            ValueError,
            "do not define the same pair IDs",
        ):
            build_pairs(
                data,
                "true_pair",
            )

    def test_missing_pair_column_is_rejected(self):
        pairs = build_pairs(
            self.data,
            "true_pair",
        ).drop(columns=["h_text"])

        with self.assertRaisesRegex(
            ValueError,
            "Missing pair columns",
        ):
            validate_pair_table(
                pairs,
                expected_true_pairs=True,
            )

    def test_duplicated_pun_instance_is_rejected(self):
        pairs = build_pairs(
            self.data,
            "true_pair",
        )
        pairs.loc[1, "h_id"] = pairs.loc[0, "h_id"]

        with self.assertRaisesRegex(
            ValueError,
            "Duplicated pun instances",
        ):
            validate_pair_table(
                pairs,
                expected_true_pairs=True,
            )

    def test_duplicated_non_pun_instance_is_rejected(self):
        pairs = build_pairs(
            self.data,
            "true_pair",
        )
        pairs.loc[1, "n_id"] = pairs.loc[0, "n_id"]

        with self.assertRaisesRegex(
            ValueError,
            "Duplicated non-pun instances",
        ):
            validate_pair_table(
                pairs,
                expected_true_pairs=True,
            )

    def test_invalid_pun_label_is_rejected(self):
        pairs = build_pairs(
            self.data,
            "true_pair",
        )
        pairs.loc[0, "h_label"] = 0

        with self.assertRaisesRegex(
            ValueError,
            "All H instances must have label 1",
        ):
            validate_pair_table(
                pairs,
                expected_true_pairs=True,
            )

    def test_invalid_non_pun_label_is_rejected(self):
        pairs = build_pairs(
            self.data,
            "true_pair",
        )
        pairs.loc[0, "n_label"] = 1

        with self.assertRaisesRegex(
            ValueError,
            "All N instances must have label 0",
        ):
            validate_pair_table(
                pairs,
                expected_true_pairs=True,
            )

    def test_true_pair_mismatch_is_rejected(self):
        pairs = build_pairs(
            self.data,
            "true_pair",
        )
        pairs.loc[0, "n_pair_id"] = "2"

        with self.assertRaisesRegex(
            ValueError,
            "True-pair table contains mismatched pair IDs",
        ):
            validate_pair_table(
                pairs,
                expected_true_pairs=True,
            )

    def test_shuffled_pair_true_match_is_rejected(self):
        pairs = build_pairs(
            self.data,
            "shuffled_pair",
            seed=13,
        )
        pairs.loc[0, "n_pair_id"] = pairs.loc[0, "h_pair_id"]

        with self.assertRaisesRegex(
            ValueError,
            "Shuffled-pair table contains true H/N matches",
        ):
            validate_pair_table(
                pairs,
                expected_true_pairs=False,
            )

    def test_invalid_true_pair_flag_is_rejected(self):
        pairs = build_pairs(
            self.data,
            "true_pair",
        )
        pairs.loc[0, "is_true_pair"] = False

        with self.assertRaisesRegex(
            ValueError,
            "True-pair table contains invalid pair flags",
        ):
            validate_pair_table(
                pairs,
                expected_true_pairs=True,
            )

    def test_invalid_shuffled_pair_flag_is_rejected(self):
        pairs = build_pairs(
            self.data,
            "shuffled_pair",
            seed=13,
        )
        pairs.loc[0, "is_true_pair"] = True

        with self.assertRaisesRegex(
            ValueError,
            "Shuffled-pair table contains invalid pair flags",
        ):
            validate_pair_table(
                pairs,
                expected_true_pairs=False,
            )


if __name__ == "__main__":
    unittest.main()