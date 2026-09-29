import unittest

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_pair_predictions
from src.evaluation.predictions import (
    build_instance_predictions,
    build_pair_predictions,
)


class PredictionsTest(unittest.TestCase):
    def test_pair_metrics_from_instance_logits(self):
        dataframe = pd.DataFrame(
            [
                {
                    "id": "1.H",
                    "text": "h1",
                    "label": 1,
                },
                {
                    "id": "1.N",
                    "text": "n1",
                    "label": 0,
                },
                {
                    "id": "2.H",
                    "text": "h2",
                    "label": 1,
                },
                {
                    "id": "2.N",
                    "text": "n2",
                    "label": 0,
                },
            ]
        )

        logits = np.array(
            [
                [0.0, 3.0],
                [3.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float16,
        )

        instance_predictions = (
            build_instance_predictions(
                dataframe,
                logits,
            )
        )

        pair_predictions = (
            build_pair_predictions(
                instance_predictions
            )
        )

        metrics = evaluate_pair_predictions(
            pair_predictions
        )

        self.assertEqual(
            metrics["pair_count"],
            2,
        )
        self.assertEqual(
            metrics[
                "pair_ranking_accuracy"
            ],
            0.5,
        )
        self.assertEqual(
            metrics["pair_exact_match"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
