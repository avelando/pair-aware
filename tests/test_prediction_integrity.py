import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.results.io as results_io
from src.evaluation.artifacts import build_evaluation_metrics, evaluate_run_splits
from src.evaluation.calibration import EVALUATION_PROTOCOL, probabilities_from_scores
from src.experiments.fingerprint import _source_paths


class PredictionIntegrityTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.multiple(
            results_io, EXPECTED_SPLIT_COUNTS={"validation": 4, "test": 4},
            EXPECTED_PAIR_COUNTS={"validation": 2, "test": 2},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def create_run(self, directory, scope="full", validation_scores=None):
        validation = pd.DataFrame({
            "id": ["001.H", "001.N", "002.H", "002.N"],
            "text": ["NA", "null", "h2", "n2"], "label": [1, 0, 1, 0],
        })
        test = validation.copy()
        test["id"] = ["003.H", "003.N", "004.H", "004.N"]
        data = {"validation": validation} if scope == "validation" else {"validation": validation, "test": test}
        scores = {"validation": [3.0, 1.0, 4.0, 2.0], "test": [-1.0, -3.0, -2.0, -4.0]}
        if validation_scores is not None:
            scores["validation"] = validation_scores
        evaluated = evaluate_run_splits(
            data, lambda split, frame: (np.column_stack((np.zeros(4), scores[split])), frame["label"].to_numpy()), scope,
        )
        paths = results_io.prepare_run_directory(directory)
        results_io.write_json({
            "status": "completed", "evaluation_scope": scope, "experiment_id": "integrity-test",
            "fingerprint": {"config": {"evaluation_protocol": EVALUATION_PROTOCOL}},
        }, paths["metadata"])
        results_io.write_json(build_evaluation_metrics(evaluated), paths["metrics"])
        results_io.write_history([{"epoch": 1}], paths["history"])
        for split, result in evaluated.items():
            instance_key = "validation_predictions" if split == "validation" else "predictions"
            pair_key = "validation_pair_predictions" if split == "validation" else "pair_predictions"
            results_io.write_dataframe(result["predictions"], paths[instance_key])
            results_io.write_dataframe(result["pair_predictions"], paths[pair_key])
        paths["checkpoint"].write_bytes(b"checkpoint")
        results_io.mark_run_completed(directory)
        return paths

    def change_csv(self, path, column, value):
        frame = pd.read_csv(
            path, dtype={"id": str, "pair_id": str, "pun_id": str, "non_pun_id": str, "text": str},
            keep_default_na=False, float_precision="round_trip",
        )
        frame[column] = frame[column].astype(object)
        frame.loc[0, column] = value
        results_io.write_dataframe(frame, path)

    def test_complete_exports_preserve_leading_zero_ids_and_literal_texts(self):
        for scope in ("validation", "full"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as directory:
                self.create_run(directory, scope)
                self.assertTrue(results_io.is_run_completed(directory, expected_experiment_id="integrity-test"))

    def test_test_predictions_use_the_validation_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_run(directory)
            test = pd.read_csv(paths["predictions"])
            self.assertEqual(test["score_threshold"].unique().tolist(), [2.0])
            self.assertEqual(test["calibrated_predicted_label"].tolist(), [0, 0, 0, 0])
            self.assertTrue(results_io.is_run_completed(directory))

    def test_row_order_does_not_change_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_run(directory)
            for key in ("validation_predictions", "validation_pair_predictions", "predictions", "pair_predictions"):
                frame = pd.read_csv(paths[key], dtype={"id": str, "pair_id": str, "pun_id": str, "non_pun_id": str, "text": str}, keep_default_na=False)
                results_io.write_dataframe(frame.iloc[::-1], paths[key])
            self.assertTrue(results_io.is_run_completed(directory))

    def test_both_splits_require_complete_instance_and_pair_schemas(self):
        for key, column in (
            ("validation_predictions", "logit_pun"), ("predictions", "probability_pun"),
            ("validation_pair_predictions", "pun_true_label"), ("pair_predictions", "non_pun_logit_pun"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                paths = self.create_run(directory)
                results_io.write_dataframe(pd.read_csv(paths[key]).drop(columns=[column]), paths[key])
                self.assertFalse(results_io.is_run_completed(directory))

    def test_duplicate_test_ids_and_pair_ids_are_rejected(self):
        for key, column in (("predictions", "id"), ("pair_predictions", "pair_id")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                paths = self.create_run(directory)
                frame = pd.read_csv(paths[key], dtype={column: str})
                frame.loc[1, column] = frame.loc[0, column]
                results_io.write_dataframe(frame, paths[key])
                self.assertFalse(results_io.is_run_completed(directory))

    def test_inconsistent_instance_fields_are_rejected_in_both_splits(self):
        changes = (
            ("pair_id", "incorrect"), ("suffix", "N"), ("true_label", 0.5),
            ("predicted_label", 0.5), ("pun_score", 99.0), ("probability_pun", 0.5),
            ("probability_non_pun", -0.1), ("correct", False), ("logit_pun", float("inf")),
            ("score_threshold", 2.5), ("calibrated_predicted_label", 0.5), ("calibrated_correct", False),
        )
        for key in ("validation_predictions", "predictions"):
            for column, value in changes:
                if key == "predictions" and column in {"correct", "calibrated_correct"}:
                    value = True
                with self.subTest(key=key, column=column), tempfile.TemporaryDirectory() as directory:
                    paths = self.create_run(directory)
                    self.change_csv(paths[key], column, value)
                    self.assertFalse(results_io.is_run_completed(directory))

    def test_inconsistent_pair_fields_are_rejected_in_both_splits(self):
        for key in ("validation_pair_predictions", "pair_predictions"):
            for column, value in (
                ("pun_id", "999.H"), ("non_pun_id", "999.N"), ("pun_true_label", 0),
                ("pun_score", 99.0), ("pun_logit_pun", 99.0), ("non_pun_probability", 0.7),
                ("pair_margin", 99.0), ("ranking_correct", False), ("ranking_tie", True),
                ("score_threshold", 2.5), ("pun_calibrated_predicted_label", 0.5),
                ("calibrated_exact_match", key == "pair_predictions"),
            ):
                with self.subTest(key=key, column=column), tempfile.TemporaryDirectory() as directory:
                    paths = self.create_run(directory)
                    self.change_csv(paths[key], column, value)
                    self.assertFalse(results_io.is_run_completed(directory))

    def test_saved_metrics_must_match_their_prediction_exports(self):
        for section, name in (
            ("validation_instance", "f1_macro"), ("test_pair", "mean_pair_margin"),
            ("test_instance_calibrated", "predicted_pun_rate"), ("validation_calibration", "brier_score"),
        ):
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                paths = self.create_run(directory)
                metrics = json.loads(paths["metrics"].read_text())
                metrics[section][name] = 99.0
                results_io.write_json(metrics, paths["metrics"])
                self.assertFalse(results_io.is_run_completed(directory))

    def test_equivalent_decisions_do_not_allow_changing_the_selected_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_run(directory)
            for key in ("validation_predictions", "validation_pair_predictions", "predictions", "pair_predictions"):
                frame = pd.read_csv(paths[key], dtype={"id": str, "pair_id": str, "pun_id": str, "non_pun_id": str, "text": str}, keep_default_na=False)
                frame["score_threshold"] = 2.5
                results_io.write_dataframe(frame, paths[key])
            metrics = json.loads(paths["metrics"].read_text())
            metrics["validation_threshold"]["score_threshold"] = 2.5
            metrics["validation_threshold"]["probability_threshold"] = float(probabilities_from_scores(2.5))
            results_io.write_json(metrics, paths["metrics"])
            self.assertFalse(results_io.is_run_completed(directory))

    def test_validation_only_integrity_never_reads_test_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_run(directory, "validation")
            original = pd.read_csv
            accessed = []

            def read(path, *arguments, **keywords):
                accessed.append(Path(path).name)
                if Path(path) in {paths["predictions"], paths["pair_predictions"]}:
                    raise AssertionError("Test predictions were accessed during validation-only checking.")
                return original(path, *arguments, **keywords)

            with patch("pandas.read_csv", side_effect=read):
                self.assertTrue(results_io.is_run_completed(directory))
            self.assertNotIn("predictions.csv", accessed)
            self.assertNotIn("pair_predictions.csv", accessed)

    def test_cross_split_pair_overlap_is_rejected_even_with_consistent_pair_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_run(directory)
            for key in ("predictions", "pair_predictions"):
                frame = pd.read_csv(paths[key], dtype={"id": str, "pair_id": str, "pun_id": str, "non_pun_id": str, "text": str}, keep_default_na=False)
                for column in ("id", "pair_id", "pun_id", "non_pun_id"):
                    if column in frame:
                        frame[column] = frame[column].str.replace("003", "001", regex=False).str.replace("004", "002", regex=False)
                results_io.write_dataframe(frame, paths[key])
            self.assertFalse(results_io.is_run_completed(directory))

    def test_smallest_negative_threshold_survives_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_run(directory, scope="validation", validation_scores=[0.0, 0.0, 0.0, 1.0])
            metrics = json.loads(paths["metrics"].read_text())
            self.assertEqual(metrics["validation_threshold"]["score_threshold"], np.nextafter(0.0, -np.inf))
            self.assertTrue(results_io.is_run_completed(directory))

    def test_invalid_json_metric_types_are_rejected(self):
        for value in (True, None, "invalid", 10 ** 1000, float("nan")):
            with self.subTest(value_type=type(value).__name__), tempfile.TemporaryDirectory() as directory:
                paths = self.create_run(directory)
                metrics = json.loads(paths["metrics"].read_text())
                metrics["validation_calibration"]["log_loss"] = value
                results_io.write_json(metrics, paths["metrics"])
                self.assertFalse(results_io.is_run_completed(directory))

    def test_integrity_source_is_included_in_training_fingerprints(self):
        for method in ("instance_level", "true_pair", "shuffled_pair"):
            self.assertIn("integrity.py", [path.name for path in _source_paths(method)])


if __name__ == "__main__":
    unittest.main()
