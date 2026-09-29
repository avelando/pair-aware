import unittest

import pandas as pd

from src.data.pairing import (
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
        self.assertEqual(
            first["n_pair_id"].nunique(),
            8,
        )


if __name__ == "__main__":
    unittest.main()
