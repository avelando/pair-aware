import unittest

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_pair_predictions
from src.evaluation.predictions import (
    INSTANCE_PREDICTION_COLUMNS,
    PAIR_PREDICTION_COLUMNS,
    build_instance_predictions,
    build_pair_predictions,
)


class PredictionsTest(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
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

    def test_instance_predictions_are_built_correctly(self):
        logits = np.array(
            [
                [0.0, 2.0],
                [3.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        predictions = build_instance_predictions(
            self.dataframe,
            logits,
        )

        self.assertEqual(
            tuple(predictions.columns),
            INSTANCE_PREDICTION_COLUMNS,
        )

        self.assertEqual(
            predictions["pair_id"].tolist(),
            ["1", "1", "2", "2"],
        )

        self.assertEqual(
            predictions["suffix"].tolist(),
            ["H", "N", "H", "N"],
        )

        self.assertEqual(
            predictions["true_label"].tolist(),
            [1, 0, 1, 0],
        )

        self.assertEqual(
            predictions["predicted_label"].tolist(),
            [1, 0, 0, 1],
        )

        self.assertEqual(
            predictions["correct"].tolist(),
            [True, True, False, False],
        )

        np.testing.assert_allclose(
            predictions[
                "pun_score"
            ].to_numpy(),
            np.array(
                [2.0, -3.0, -1.0, 1.0]
            ),
        )

        np.testing.assert_allclose(
            predictions[
                "probability_non_pun"
            ].to_numpy()
            + predictions[
                "probability_pun"
            ].to_numpy(),
            np.ones(4),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_invalid_instance_logits_shape_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Expected logits with shape",
        ):
            build_instance_predictions(
                self.dataframe,
                np.zeros((4, 3)),
            )

    def test_dataframe_and_logits_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Dataframe and logits must have the same length",
        ):
            build_instance_predictions(
                self.dataframe,
                np.zeros((3, 2)),
            )

    def test_suffix_and_label_mismatch_is_rejected(self):
        dataframe = self.dataframe.copy()

        dataframe.loc[
            0,
            "label",
        ] = 0

        with self.assertRaisesRegex(
            ValueError,
            "Suffix and label are inconsistent",
        ):
            build_instance_predictions(
                dataframe,
                np.zeros((4, 2)),
            )

    def test_pair_predictions_cover_positive_tie_and_negative_margins(self):
        instance_predictions = pd.DataFrame(
            [
                {
                    "id": "1.H",
                    "pair_id": "1",
                    "suffix": "H",
                    "predicted_label": 1,
                    "pun_score": 2.0,
                    "probability_pun": 0.9,
                },
                {
                    "id": "1.N",
                    "pair_id": "1",
                    "suffix": "N",
                    "predicted_label": 0,
                    "pun_score": -1.0,
                    "probability_pun": 0.2,
                },
                {
                    "id": "2.H",
                    "pair_id": "2",
                    "suffix": "H",
                    "predicted_label": 1,
                    "pun_score": 0.5,
                    "probability_pun": 0.7,
                },
                {
                    "id": "2.N",
                    "pair_id": "2",
                    "suffix": "N",
                    "predicted_label": 0,
                    "pun_score": 0.5,
                    "probability_pun": 0.3,
                },
                {
                    "id": "3.H",
                    "pair_id": "3",
                    "suffix": "H",
                    "predicted_label": 0,
                    "pun_score": -2.0,
                    "probability_pun": 0.1,
                },
                {
                    "id": "3.N",
                    "pair_id": "3",
                    "suffix": "N",
                    "predicted_label": 1,
                    "pun_score": 1.0,
                    "probability_pun": 0.8,
                },
            ]
        )

        pairs = build_pair_predictions(
            instance_predictions
        )

        self.assertEqual(
            tuple(pairs.columns),
            PAIR_PREDICTION_COLUMNS,
        )

        self.assertEqual(
            pairs["pair_id"].tolist(),
            ["1", "2", "3"],
        )

        np.testing.assert_allclose(
            pairs[
                "pair_margin"
            ].to_numpy(),
            np.array(
                [3.0, 0.0, -3.0]
            ),
        )

        self.assertEqual(
            pairs[
                "ranking_correct"
            ].tolist(),
            [True, False, False],
        )

        self.assertEqual(
            pairs[
                "ranking_tie"
            ].tolist(),
            [False, True, False],
        )

        self.assertEqual(
            pairs[
                "exact_match"
            ].tolist(),
            [True, True, False],
        )

        self.assertEqual(
            pairs["pun_id"].tolist(),
            [
                "1.H",
                "2.H",
                "3.H",
            ],
        )

        self.assertEqual(
            pairs["non_pun_id"].tolist(),
            [
                "1.N",
                "2.N",
                "3.N",
            ],
        )

    def test_missing_instance_prediction_column_is_rejected(self):
        instance_predictions = pd.DataFrame(
            [
                {
                    "id": "1.H",
                    "pair_id": "1",
                    "suffix": "H",
                    "predicted_label": 1,
                    "pun_score": 1.0,
                }
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "Missing instance prediction columns",
        ):
            build_pair_predictions(
                instance_predictions
            )

    def test_pair_with_one_instance_is_rejected(self):
        instance_predictions = pd.DataFrame(
            [
                {
                    "id": "1.H",
                    "pair_id": "1",
                    "suffix": "H",
                    "predicted_label": 1,
                    "pun_score": 1.0,
                    "probability_pun": 0.8,
                }
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not contain exactly two instances",
        ):
            build_pair_predictions(
                instance_predictions
            )

    def test_pair_with_more_than_two_instances_is_rejected(self):
        instance_predictions = pd.DataFrame(
            [
                {
                    "id": "1.H",
                    "pair_id": "1",
                    "suffix": "H",
                    "predicted_label": 1,
                    "pun_score": 1.0,
                    "probability_pun": 0.8,
                },
                {
                    "id": "1.N",
                    "pair_id": "1",
                    "suffix": "N",
                    "predicted_label": 0,
                    "pun_score": -1.0,
                    "probability_pun": 0.2,
                },
                {
                    "id": "1.X",
                    "pair_id": "1",
                    "suffix": "N",
                    "predicted_label": 0,
                    "pun_score": -0.5,
                    "probability_pun": 0.3,
                },
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not contain exactly two instances",
        ):
            build_pair_predictions(
                instance_predictions
            )

    def test_pair_without_one_h_and_one_n_is_rejected(self):
        instance_predictions = pd.DataFrame(
            [
                {
                    "id": "1.H",
                    "pair_id": "1",
                    "suffix": "H",
                    "predicted_label": 1,
                    "pun_score": 1.0,
                    "probability_pun": 0.8,
                },
                {
                    "id": "2.H",
                    "pair_id": "1",
                    "suffix": "H",
                    "predicted_label": 1,
                    "pun_score": 0.5,
                    "probability_pun": 0.7,
                },
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not contain one H and one N instance",
        ):
            build_pair_predictions(
                instance_predictions
            )

    def test_pair_metrics_from_instance_logits(self):
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
                self.dataframe,
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
            metrics[
                "pair_exact_match"
            ],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()