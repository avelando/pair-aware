import unittest

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import (
    evaluate_instance_logits,
    evaluate_instance_predictions,
    evaluate_pair_predictions,
    predictions_from_logits,
)


class MetricsTest(unittest.TestCase):
    def test_predictions_from_logits(self):
        logits = np.array(
            [
                [3.0, 1.0],
                [0.0, 2.0],
                [1.0, 1.0],
            ]
        )

        predictions = predictions_from_logits(logits)

        np.testing.assert_array_equal(
            predictions,
            np.array([0, 1, 0]),
        )
        self.assertEqual(predictions.dtype.kind, "i")

    def test_predictions_from_torch_logits(self):
        logits = torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 0.0],
            ]
        )

        predictions = predictions_from_logits(logits)

        np.testing.assert_array_equal(
            predictions,
            np.array([1, 0]),
        )

    def test_invalid_logits_shape_is_rejected(self):
        invalid_logits = (
            np.array([0.0, 1.0]),
            np.zeros((2, 3)),
            np.zeros((2, 2, 1)),
        )

        for logits in invalid_logits:
            with self.subTest(shape=logits.shape):
                with self.assertRaisesRegex(
                    ValueError,
                    "Expected logits with shape",
                ):
                    predictions_from_logits(logits)

    def test_instance_metrics_are_computed_correctly(self):
        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 0])

        metrics, report, matrix = evaluate_instance_predictions(
            labels,
            predictions,
        )

        expected_half_metrics = (
            "accuracy",
            "precision_non_pun",
            "recall_non_pun",
            "f1_non_pun",
            "precision_pun",
            "recall_pun",
            "f1_pun",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        )

        for key in expected_half_metrics:
            self.assertAlmostEqual(metrics[key], 0.5)

        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertAlmostEqual(report["non_pun"]["f1-score"], 0.5)
        self.assertAlmostEqual(report["pun"]["f1-score"], 0.5)

        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [1, 1],
                    [1, 1],
                ]
            ),
        )

    def test_instance_metrics_handle_single_present_class(self):
        labels = np.array([0, 0])
        predictions = np.array([0, 0])

        metrics, _, matrix = evaluate_instance_predictions(
            labels,
            predictions,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_non_pun"], 1.0)
        self.assertEqual(metrics["f1_pun"], 0.0)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [2, 0],
                    [0, 0],
                ]
            ),
        )

    def test_multidimensional_labels_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Labels must be one-dimensional",
        ):
            evaluate_instance_predictions(
                np.array([[0], [1]]),
                np.array([0, 1]),
            )

    def test_multidimensional_predictions_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Predictions must be one-dimensional",
        ):
            evaluate_instance_predictions(
                np.array([0, 1]),
                np.array([[0], [1]]),
            )

    def test_different_instance_lengths_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Labels and predictions must have the same length",
        ):
            evaluate_instance_predictions(
                np.array([0, 1]),
                np.array([0]),
            )

    def test_non_binary_labels_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Labels must be binary",
        ):
            evaluate_instance_predictions(
                np.array([0, 2]),
                np.array([0, 1]),
            )

    def test_non_binary_predictions_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Predictions must be binary",
        ):
            evaluate_instance_predictions(
                np.array([0, 1]),
                np.array([0, 2]),
            )

    def test_instance_logits_return_predictions_and_metrics(self):
        labels = np.array([0, 1, 1])

        logits = np.array(
            [
                [3.0, 0.0],
                [0.0, 2.0],
                [4.0, 1.0],
            ]
        )

        metrics, report, matrix, predictions = evaluate_instance_logits(
            labels,
            logits,
        )

        np.testing.assert_array_equal(
            predictions,
            np.array([0, 1, 0]),
        )

        self.assertAlmostEqual(
            metrics["accuracy"],
            2.0 / 3.0,
        )

        self.assertIn("non_pun", report)
        self.assertIn("pun", report)

        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [1, 0],
                    [1, 1],
                ]
            ),
        )

    def test_pair_metrics_are_computed_correctly(self):
        pair_predictions = pd.DataFrame(
            [
                {
                    "ranking_correct": True,
                    "ranking_tie": False,
                    "exact_match": True,
                    "pair_margin": 2.0,
                },
                {
                    "ranking_correct": False,
                    "ranking_tie": True,
                    "exact_match": False,
                    "pair_margin": 0.0,
                },
                {
                    "ranking_correct": False,
                    "ranking_tie": False,
                    "exact_match": True,
                    "pair_margin": -1.0,
                },
            ]
        )

        metrics = evaluate_pair_predictions(
            pair_predictions
        )

        self.assertEqual(
            metrics["pair_count"],
            3,
        )

        self.assertAlmostEqual(
            metrics["pair_ranking_accuracy"],
            1.0 / 3.0,
        )

        self.assertAlmostEqual(
            metrics["pair_exact_match"],
            2.0 / 3.0,
        )

        self.assertEqual(
            metrics["pair_ties"],
            1,
        )

        self.assertAlmostEqual(
            metrics["mean_pair_margin"],
            1.0 / 3.0,
        )

        self.assertAlmostEqual(
            metrics["median_pair_margin"],
            0.0,
        )

        self.assertAlmostEqual(
            metrics["std_pair_margin"],
            np.sqrt(7.0 / 3.0),
        )

        self.assertEqual(
            metrics["min_pair_margin"],
            -1.0,
        )

        self.assertEqual(
            metrics["max_pair_margin"],
            2.0,
        )

    def test_single_pair_has_zero_margin_standard_deviation(self):
        pair_predictions = pd.DataFrame(
            [
                {
                    "ranking_correct": True,
                    "ranking_tie": False,
                    "exact_match": True,
                    "pair_margin": 1.5,
                }
            ]
        )

        metrics = evaluate_pair_predictions(
            pair_predictions
        )

        self.assertEqual(
            metrics["pair_count"],
            1,
        )

        self.assertEqual(
            metrics["std_pair_margin"],
            0.0,
        )

        self.assertEqual(
            metrics["mean_pair_margin"],
            1.5,
        )

        self.assertEqual(
            metrics["median_pair_margin"],
            1.5,
        )

        self.assertEqual(
            metrics["min_pair_margin"],
            1.5,
        )

        self.assertEqual(
            metrics["max_pair_margin"],
            1.5,
        )

    def test_missing_pair_metric_column_is_rejected(self):
        pair_predictions = pd.DataFrame(
            [
                {
                    "ranking_correct": True,
                    "ranking_tie": False,
                    "exact_match": True,
                }
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "Missing pair prediction columns",
        ):
            evaluate_pair_predictions(
                pair_predictions
            )

    def test_empty_pair_predictions_are_rejected(self):
        pair_predictions = pd.DataFrame(
            columns=[
                "ranking_correct",
                "ranking_tie",
                "exact_match",
                "pair_margin",
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "Pair predictions cannot be empty",
        ):
            evaluate_pair_predictions(
                pair_predictions
            )


if __name__ == "__main__":
    unittest.main()