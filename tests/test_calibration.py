import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, log_loss

import src.evaluation.artifacts as artifacts_module
import src.experiments.fingerprint as fingerprint_module
from src.evaluation.calibration import (
    EVALUATION_PROTOCOL,
    evaluate_probability_calibration,
    predictions_at_threshold,
    probabilities_from_scores,
    select_validation_threshold,
)
from src.results.io import (
    is_run_completed,
    mark_run_completed,
    prepare_run_directory,
    write_dataframe,
    write_history,
    write_json,
)


class CalibrationTest(unittest.TestCase):
    def test_probability_metrics_match_reference_definitions(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([-2.0, 0.2, 0.7, 3.0])
        probabilities = probabilities_from_scores(scores)
        result = evaluate_probability_calibration(labels, scores, scores > 0)
        self.assertAlmostEqual(result["brier_score"], brier_score_loss(labels, probabilities))
        self.assertAlmostEqual(result["log_loss"], log_loss(labels, probabilities))
        self.assertEqual(result["predicted_pun_rate"], 0.75)
        self.assertEqual(sum(bin_["count"] for bin_ in result["reliability_bins"]), 4)

    def test_ece_uses_positive_class_frequency_and_bin_weights(self):
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        scores = np.log(probabilities / (1 - probabilities))
        result = evaluate_probability_calibration([0, 1, 1, 1], scores, scores > 0, n_bins=2)
        self.assertAlmostEqual(result["ece_pun"], 0.25)
        self.assertEqual(result["ece_definition"], "positive_class_equal_width")

    def test_extreme_scores_preserve_finite_unclipped_log_loss(self):
        result = evaluate_probability_calibration([1, 0], [-1000.0, 1000.0], [0, 1])
        self.assertEqual(result["log_loss"], 1000.0)
        self.assertEqual(result["brier_score"], 1.0)
        self.assertEqual(result["ece_pun"], 1.0)
        self.assertEqual([bin_["bin_index"] for bin_ in result["reliability_bins"]], [0, 14])

    def test_score_ties_use_non_pun_at_the_default_threshold(self):
        np.testing.assert_array_equal(predictions_at_threshold([-1, 0, 1], 0), [0, 0, 1])

    def test_threshold_recovers_shifted_decision_boundary(self):
        selection = select_validation_threshold([0, 0, 1, 1], [1, 2, 3, 4])
        self.assertEqual(selection["score_threshold"], 2.0)
        self.assertEqual(selection["validation_f1_macro"], 1.0)
        self.assertAlmostEqual(selection["default_validation_f1_macro"], 1 / 3)
        self.assertAlmostEqual(selection["probability_threshold"], 1 / (1 + np.exp(-2.0)))

    def test_threshold_matches_exhaustive_search_with_duplicate_scores(self):
        generator = np.random.default_rng(13)
        for _ in range(30):
            labels = generator.integers(0, 2, size=20)
            labels[:2] = [0, 1]
            scores = generator.integers(-3, 4, size=20).astype(float)
            candidates = np.unique(np.r_[scores, 0.0, np.nextafter(scores.min(), -np.inf)])
            values = [f1_score(labels, scores > threshold, average="macro") for threshold in candidates]
            expected = min(
                (index for index, value in enumerate(values) if value == max(values)),
                key=lambda index: (abs(candidates[index]), candidates[index]),
            )
            selection = select_validation_threshold(labels, scores)
            self.assertAlmostEqual(selection["validation_f1_macro"], max(values))
            self.assertEqual(selection["score_threshold"], candidates[expected])

    def test_equal_optima_prefer_default_then_lower_threshold(self):
        default = select_validation_threshold([0, 1], [-1, 1])
        self.assertEqual(default["score_threshold"], 0.0)
        lower = select_validation_threshold([1, 0, 1, 0, 1], [-2, -1, 0, 1, 2])
        self.assertEqual(lower["score_threshold"], -1.0)

    def test_selection_is_invariant_to_instance_order(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(
            select_validation_threshold(labels, scores),
            select_validation_threshold(labels[::-1], scores[::-1]),
        )

    def test_invalid_inputs_are_rejected(self):
        invalid = (
            ([], []), ([0, 1], [1]), ([[0, 1]], [[1, 2]]),
            ([0.5, 1], [1, 2]), ([0, 1], [np.nan, 2]), ([0, 1], [1, np.inf]),
        )
        for labels, scores in invalid:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                select_validation_threshold(labels, scores)
        with self.assertRaises(ValueError):
            select_validation_threshold([1, 1], [1, 2])
        for bins in (0, -1, 1.5, True):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                evaluate_probability_calibration([0, 1], [0, 1], [0, 1], n_bins=bins)
        with self.assertRaises(ValueError):
            evaluate_probability_calibration([0, 1], [0, 1], [0, 2])


class CalibrationIntegrationTest(unittest.TestCase):
    def evaluate(self, test_labels=None, evaluation_scope="full"):
        validation = pd.DataFrame({
            "id": ["1.H", "1.N", "2.H", "2.N"],
            "label": [1, 0, 1, 0], "text": ["h1", "n1", "h2", "n2"],
        })
        test = validation.copy()
        if test_labels is not None:
            test["label"] = test_labels
            test["id"] = [f"{index // 2 + 3}.{'H' if label else 'N'}" for index, label in enumerate(test_labels)]
        data = {"validation": validation, "test": test}
        calls = []

        def predict(split_name, frame):
            calls.append(split_name)
            scores = np.array([3.0, 1.0, 4.0, 2.0])
            return np.column_stack((np.zeros(4), scores)), frame["label"].to_numpy()

        with patch.object(
            artifacts_module, "select_validation_threshold", wraps=select_validation_threshold,
        ) as fit:
            results = artifacts_module.evaluate_run_splits(data, predict, evaluation_scope)
        self.assertEqual(fit.call_count, 1)
        return results, calls

    def test_test_labels_cannot_change_selected_threshold(self):
        first, _ = self.evaluate()
        second, _ = self.evaluate(test_labels=[0, 1, 0, 1])
        self.assertEqual(first["validation"]["threshold"], second["validation"]["threshold"])
        self.assertEqual(second["test"]["instance_calibrated"]["f1_macro"], 0.0)

    def test_threshold_improves_classification_without_changing_ranking(self):
        results, calls = self.evaluate()
        self.assertEqual(calls, ["validation", "test"])
        for result in results.values():
            self.assertAlmostEqual(result["instance"]["f1_macro"], 1 / 3)
            self.assertEqual(result["instance_calibrated"]["f1_macro"], 1.0)
            self.assertEqual(result["pair"]["pair_exact_match"], 0.0)
            self.assertEqual(result["pair_calibrated"]["pair_exact_match"], 1.0)
            for metric in ("pair_ranking_accuracy", "mean_pair_margin", "pair_ties"):
                self.assertEqual(result["pair"][metric], result["pair_calibrated"][metric])
            self.assertEqual(result["predictions"]["score_threshold"].unique().tolist(), [2.0])
            self.assertTrue(result["predictions"]["calibrated_correct"].all())
            self.assertTrue(result["pair_predictions"]["calibrated_exact_match"].all())

    def test_validation_only_metrics_and_exports_remain_isolated(self):
        results, calls = self.evaluate(evaluation_scope="validation")
        metrics = artifacts_module.build_evaluation_metrics(results)
        self.assertEqual(calls, ["validation"])
        self.assertTrue(all(name.startswith("validation_") for name in metrics))
        self.assertEqual(metrics["validation_threshold"]["fit_split"], "validation")
        self.assertIn("validation_calibration", metrics)
        self.assertIn("validation_confusion_matrix_calibrated", metrics)

    def test_fingerprint_records_the_evaluation_protocol(self):
        config = fingerprint_module.get_config_snapshot("true_pair")
        self.assertEqual(config["evaluation_protocol"], EVALUATION_PROTOCOL)
        paths = fingerprint_module._source_paths("true_pair")
        self.assertTrue(any(path.name == "calibration.py" for path in paths))


class CalibrationCompletionTest(unittest.TestCase):
    evaluate = CalibrationIntegrationTest.evaluate

    def create_run(self, directory):
        results, _ = self.evaluate(evaluation_scope="validation")
        paths = prepare_run_directory(directory)
        write_json({
            "status": "completed", "evaluation_scope": "validation",
            "fingerprint": {"config": {"evaluation_protocol": EVALUATION_PROTOCOL}},
        }, paths["metadata"])
        write_json(artifacts_module.build_evaluation_metrics(results), paths["metrics"])
        write_history([{"epoch": 1}], paths["history"])
        write_dataframe(results["validation"]["predictions"], paths["validation_predictions"])
        write_dataframe(results["validation"]["pair_predictions"], paths["validation_pair_predictions"])
        paths["checkpoint"].write_bytes(b"checkpoint")
        mark_run_completed(directory)
        return paths

    def completed(self, directory):
        with patch.multiple(
            "src.results.io", EXPECTED_SPLIT_COUNTS={"validation": 4}, EXPECTED_PAIR_COUNTS={"validation": 2},
        ):
            return is_run_completed(directory)

    def test_calibration_outputs_are_required_for_new_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_run(Path(directory))
            self.assertTrue(self.completed(directory))
            metrics = json.loads(paths["metrics"].read_text())
            del metrics["validation_calibration"]
            write_json(metrics, paths["metrics"])
            self.assertFalse(self.completed(directory))

    def test_threshold_must_have_been_fitted_on_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_run(Path(directory))
            metrics = json.loads(paths["metrics"].read_text())
            metrics["validation_threshold"]["fit_split"] = "test"
            write_json(metrics, paths["metrics"])
            self.assertFalse(self.completed(directory))

    def test_calibrated_csv_columns_are_required(self):
        for artifact, column in (
            ("validation_predictions", "calibrated_predicted_label"),
            ("validation_pair_predictions", "calibrated_exact_match"),
        ):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                paths = self.create_run(Path(directory))
                frame = pd.read_csv(paths[artifact]).drop(columns=column)
                write_dataframe(frame, paths[artifact])
                self.assertFalse(self.completed(directory))


if __name__ == "__main__":
    unittest.main()
