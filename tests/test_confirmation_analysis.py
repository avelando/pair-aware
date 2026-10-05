import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.results.confirmation_analysis as analysis
from src.evaluation.artifacts import build_evaluation_metrics, evaluate_run_splits
from src.evaluation.calibration import EVALUATION_PROTOCOL
from src.paths import PROJECT_ROOT
from src.results.io import mark_run_completed, prepare_run_directory, write_dataframe, write_history, write_json


class ConfirmationAnalysisTest(unittest.TestCase):
    def records(self, weight=0.25):
        return pd.DataFrame([
            {**task, **dict.fromkeys(analysis.RUN_METRICS, 0.7)}
            for task in analysis.build_confirmation_grid(weight)
        ])

    def test_complete_grid_is_aggregated_without_mixing_methods_or_weights(self):
        tables = analysis.build_confirmation_tables(self.records(), 0.25)
        self.assertEqual({name: len(frame) for name, frame in tables.items()}, {
            "runs": 72, "by_split": 12, "by_method": 2,
            "paired_runs": 36, "paired_by_split": 6, "paired_summary": 1,
        })
        self.assertTrue(tables["by_split"]["run_count"].eq(6).all())
        self.assertTrue(tables["by_method"]["run_count"].eq(36).all())
        self.assertTrue(tables["by_method"]["pair_loss_weight"].eq(0.25).all())

    def test_missing_duplicate_mixed_weights_and_invalid_metrics_are_rejected(self):
        records = self.records()
        invalid = [records.iloc[:-1], pd.concat([records.iloc[:-1], records.iloc[:1]])]
        for column, value in (
            ("pair_loss_weight", 0.5), ("evaluation_scope", "validation"),
            ("test_f1_macro", float("nan")), ("test_f1_macro", 1.1), ("validation_log_loss", -1),
        ):
            changed = records.copy()
            changed.loc[0, column] = value
            invalid.append(changed)
        for frame in invalid:
            with self.assertRaises(ValueError):
                analysis.build_confirmation_tables(frame, 0.25)

    def test_paired_differences_match_split_and_model_seeds(self):
        records = self.records()
        records["test_f1_macro"] = records["model_seed"].map(lambda seed: 0.5 + seed / 1000)
        records.loc[records["method"] == "true_pair", "test_f1_macro"] += 0.02
        tables = analysis.build_confirmation_tables(records.sample(frac=1, random_state=13), 0.25)
        column = "test_f1_macro_true_minus_shuffled"
        self.assertTrue(np.allclose(tables["paired_runs"][column], 0.02))
        self.assertAlmostEqual(tables["paired_summary"].iloc[0][f"{column}_mean"], 0.02)
        self.assertEqual(set(tables["paired_runs"]["model_seed"]), set(analysis.MODEL_SEEDS))

    def test_split_and_model_variability_are_reported_separately(self):
        records = self.records()
        split_order = {seed: index for index, seed in enumerate(analysis.SPLIT_SEEDS)}
        model_order = {seed: index for index, seed in enumerate(analysis.MODEL_SEEDS)}
        records["test_f1_macro"] = 0.7 + records["split_seed"].map(split_order) * 0.01 + records["model_seed"].map(model_order) * 0.001
        table = analysis.build_confirmation_tables(records, 0.25)["by_method"]
        self.assertAlmostEqual(table.iloc[0]["test_f1_macro_mean"], 0.7275)
        self.assertAlmostEqual(table.iloc[0]["test_f1_macro_std_between_splits"], np.std(np.arange(6) * 0.01, ddof=1))
        self.assertAlmostEqual(table.iloc[0]["test_f1_macro_mean_std_model_seeds"], np.std(np.arange(6) * 0.001, ddof=1))

    def test_invalid_selection_stops_before_any_test_data_are_loaded(self):
        with patch.object(analysis, "load_screening_selection", side_effect=RuntimeError("invalid selection")), patch.object(
            analysis, "load_split_directory",
        ) as load, patch.object(analysis, "write_dataframe") as write, self.assertRaises(RuntimeError):
            analysis.generate_confirmation_analysis()
        load.assert_not_called()
        write.assert_not_called()

    def test_export_audit_rebuilds_thresholds_and_detects_test_artifact_changes(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            weight = 0.25
            tasks = analysis.build_confirmation_grid(weight)
            decision = {"selected_pair_loss_weight": weight, "selection_path": "selection.json", "selection_sha256": "frozen"}
            split_data = {
                seed: {
                    split: pd.DataFrame({
                        "id": [f"{offset}.H", f"{offset}.N", f"{offset + 1}.H", f"{offset + 1}.N"],
                        "text": ["h1", "n1", "h2", "n2"], "label": [1, 0, 1, 0],
                    })
                    for split, offset in (("validation", 1), ("test", 3))
                }
                for seed in analysis.SPLIT_SEEDS
            }
            stack.enter_context(patch.object(analysis, "load_screening_selection", return_value=decision))
            stack.enter_context(patch.object(analysis, "load_split_directory", side_effect=lambda seed, split_names: split_data[seed]))
            stack.enter_context(patch.object(
                analysis, "get_run_dir",
                side_effect=lambda **task: root / "runs" / task["method"] / str(task["split_seed"]) / str(task["model_seed"]),
            ))
            stack.enter_context(patch.multiple(
                "src.results.io", EXPECTED_SPLIT_COUNTS={"validation": 4, "test": 4}, EXPECTED_PAIR_COUNTS={"validation": 2, "test": 2},
            ))

            def fingerprint(**task):
                return {
                    "experiment_id": str(tuple(task.values())), "fingerprint_version": 1,
                    "config": {"evaluation_protocol": EVALUATION_PROTOCOL, "pair_loss_weight": weight},
                    "config_hash": "config", "dataset_hash": "dataset", "source_hash": "source",
                }

            stack.enter_context(patch.object(analysis, "build_experiment_fingerprint", side_effect=fingerprint))
            first_paths = None
            for task in tasks:
                paths = prepare_run_directory(analysis.get_run_dir(**task))
                expected = fingerprint(**task)
                write_json({
                    **task, "status": "completed", "experiment_id": expected["experiment_id"],
                    "training": {"pair_loss_weight": weight},
                    "fingerprint": {"version": 1, **{key: expected[key] for key in ("config", "config_hash", "dataset_hash", "source_hash")}},
                }, paths["metadata"])
                logits = np.array([[0, 3], [0, 1], [0, 4], [0, 2]])
                if task["method"] == "shuffled_pair":
                    logits = -logits
                results = evaluate_run_splits(
                    split_data[task["split_seed"]], lambda split, frame: (logits, frame["label"].to_numpy()), "full",
                )
                write_json(build_evaluation_metrics(results), paths["metrics"])
                write_history([{"epoch": 1}], paths["history"])
                for split in ("validation", "test"):
                    prefix = "validation_" if split == "validation" else ""
                    write_dataframe(results[split]["predictions"], paths[f"{prefix}predictions"])
                    write_dataframe(results[split]["pair_predictions"], paths[f"{prefix}pair_predictions"])
                mark_run_completed(analysis.get_run_dir(**task))
                if first_paths is None:
                    first_paths = paths
            phase_root = root / "confirmation" / "lambda_0p25"
            linked = {**decision, "status": "validated"}
            write_json({"phase": "confirmation", "tasks": tasks, "expected_runs": 72, "selection": linked}, phase_root / "confirmation_plan.json")
            write_json({"status": "completed", "artifacts_verified": True, "tasks": tasks, "selection": linked}, phase_root / "confirmation_result.json")
            result = analysis.generate_confirmation_analysis(confirmation_root=root / "confirmation")
            metadata = json.loads(Path(result["metadata_path"]).read_text())
            self.assertEqual(result["completed_runs"], 72)
            self.assertFalse(metadata["calibration_changes_probabilities"])
            self.assertIn("not_independent_holdout", metadata["interpretation"])
            for entry in result["outputs"].values():
                self.assertEqual(entry["sha256"], analysis._hash_file(entry["path"]))
            runs = pd.read_csv(result["outputs"]["runs"]["path"])
            self.assertTrue(runs["test_score_threshold"].equals(runs["validation_score_threshold"]))
            self.assertAlmostEqual(runs.iloc[0]["test_f1_calibration_gain"], 2 / 3)
            for artifact, column, value in (
                ("predictions", "calibrated_predicted_label", 0),
                ("pair_predictions", "pair_margin", 999),
                ("validation_predictions", "score_threshold", 999),
                ("predictions", "true_label", 0),
            ):
                path = first_paths[artifact]
                original = path.read_bytes()
                frame = pd.read_csv(path)
                frame.loc[0, column] = value
                write_dataframe(frame, path)
                try:
                    with self.assertRaises((AssertionError, RuntimeError, ValueError)):
                        analysis.generate_confirmation_analysis(confirmation_root=root / "confirmation")
                finally:
                    path.write_bytes(original)
            path = first_paths["metrics"]
            original = path.read_bytes()
            metrics = json.loads(original)
            metrics["test_instance"]["f1_macro"] = 0.99
            write_json(metrics, path)
            with self.assertRaises(ValueError):
                analysis.generate_confirmation_analysis(confirmation_root=root / "confirmation")
            path.write_bytes(original)
            report_path = phase_root / "confirmation_result.json"
            report = json.loads(report_path.read_text())
            report["selection"]["selection_sha256"] = "different selection"
            write_json(report, report_path)
            with self.assertRaises(RuntimeError):
                analysis.generate_confirmation_analysis(confirmation_root=root / "confirmation")

    def test_cross_split_selection_overlap_is_counted(self):
        data = {
            seed: {"validation": pd.DataFrame({"id": [str(seed)]}), "test": pd.DataFrame({"id": ["13", "21"]})}
            for seed in analysis.SPLIT_SEEDS
        }
        rows = analysis._selection_overlap(data)
        self.assertEqual(rows[0]["test_instances_in_other_validation_splits"], 1)
        self.assertEqual(rows[2]["test_instances_in_other_validation_splits"], 2)

    def test_cli_help_does_not_load_results_or_training_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.results.confirmation_analysis", "--help"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--selection", result.stdout)


if __name__ == "__main__":
    unittest.main()
