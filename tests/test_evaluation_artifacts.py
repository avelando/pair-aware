import unittest

import numpy as np
import pandas as pd

from src.evaluation.artifacts import evaluate_split_predictions
from src.evaluation.predictions import COMPLETE_PAIR_PREDICTION_COLUMNS, INSTANCE_PREDICTION_COLUMNS


class EvaluationArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame({
            "id": ["1.H", "1.N", "2.H", "2.N"],
            "text": ["h1", "n1", "h2", "n2"],
            "label": [1, 0, 1, 0],
        })
        self.labels = self.dataframe["label"].to_numpy()
        self.logits = np.array([[0.0, 3.0], [3.0, 0.0], [2.0, 1.0], [0.0, 1.0]])

    def test_predictions_and_metrics_describe_the_same_instances(self):
        result = evaluate_split_predictions(self.dataframe, self.logits, self.labels, "validation")
        self.assertEqual(tuple(result["predictions"].columns), INSTANCE_PREDICTION_COLUMNS)
        self.assertEqual(tuple(result["pair_predictions"].columns), COMPLETE_PAIR_PREDICTION_COLUMNS)
        self.assertEqual(result["predictions"]["id"].tolist(), self.dataframe["id"].tolist())
        self.assertEqual(result["instance"]["f1_macro"], 0.5)
        self.assertEqual(result["pair"]["pair_ranking_accuracy"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])

    def test_pair_predictions_preserve_member_labels_logits_and_probabilities(self):
        result = evaluate_split_predictions(self.dataframe, self.logits, self.labels, "validation")
        instances = result["predictions"].set_index("id")
        for pair in result["pair_predictions"].to_dict("records"):
            for prefix in ("pun", "non_pun"):
                member = instances.loc[pair[f"{prefix}_id"]]
                for column in ("true_label", "logit_non_pun", "logit_pun", "probability_non_pun"):
                    self.assertEqual(pair[f"{prefix}_{column}"], member[column])
                self.assertEqual(pair[f"{prefix}_probability"], member["probability_pun"])

    def test_test_split_uses_the_same_artifact_format(self):
        result = evaluate_split_predictions(self.dataframe, self.logits, self.labels, "test")
        self.assertEqual(len(result["pair_predictions"]), 2)

    def test_nonfinite_logits_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                logits = self.logits.copy()
                logits[0, 0] = value
                evaluate_split_predictions(self.dataframe, logits, self.labels, "validation")

    def test_misaligned_labels_are_rejected(self):
        for labels in (self.labels[::-1], self.labels[:3], self.labels[:, None]):
            with self.subTest(shape=labels.shape), self.assertRaises(RuntimeError):
                evaluate_split_predictions(self.dataframe, self.logits, labels, "validation")

    def test_invalid_logit_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_split_predictions(self.dataframe, self.logits[:3], self.labels, "validation")

    def test_empty_dataframe_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_split_predictions(self.dataframe.iloc[:0], self.logits[:0], self.labels[:0], "validation")

    def test_training_split_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_split_predictions(self.dataframe, self.logits, self.labels, "train")


if __name__ == "__main__":
    unittest.main()
