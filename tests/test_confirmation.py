import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.experiments.run_confirmation as runner
import src.results.confirmation as confirmation
import src.results.selection as screening
from src.paths import PROJECT_ROOT
from src.results.io import write_json


class ConfirmationTest(unittest.TestCase):
    def records(self):
        return pd.DataFrame([
            {
                **task, **dict.fromkeys(screening.METRIC_FIELDS, 0.7),
                "run_dir": f"/original/{index}", "experiment_id": f"experiment-{index}",
                "config_hash": "config", "dataset_hash": "dataset", "source_hash": "source",
                "metadata_sha256": "metadata", "metrics_sha256": "metrics",
                "validation_predictions_sha256": "instances", "validation_pair_predictions_sha256": "pairs",
            }
            for index, task in enumerate(screening._screening_tasks())
        ])

    def manifest(self, directory, records):
        with patch.object(screening, "collect_screening_results", return_value=records):
            result = screening.generate_screening_selection(directory)
        return Path(result["selection_path"])

    def grid_result(self, weight, failures=0):
        tasks = runner.build_confirmation_grid(weight)
        return {
            "tasks": tasks, "results": [], "planned_runs": 72, "completed_runs": 0,
            "skipped_runs": 72 - failures, "failed_runs": failures,
        }

    def test_confirmation_contains_exactly_36_runs_per_method_with_one_weight(self):
        for weight in screening.PAIR_LOSS_WEIGHTS:
            tasks = runner.build_confirmation_grid(weight)
            self.assertEqual(len(tasks), 72)
            self.assertEqual(Counter(task["method"] for task in tasks), {"true_pair": 36, "shuffled_pair": 36})
            self.assertEqual(len({tuple(task.values()) for task in tasks}), 72)
            self.assertTrue(all(task["pair_loss_weight"] == weight and task["evaluation_scope"] == "full" for task in tasks))

    def test_invalid_weights_are_rejected(self):
        for weight in (True, -1, 0.3, float("nan"), float("inf"), None):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                runner.build_confirmation_grid(weight)

    def test_preview_does_not_read_selection_create_reports_or_start_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "load_screening_selection") as load, patch(
            "src.experiments.run_grid.execute_grid_run",
        ) as train, patch.object(runner, "write_json") as write:
            result = runner.run_confirmation(dry_run=True, preview_weight=0.25, output_root=directory)
        load.assert_not_called()
        train.assert_not_called()
        write.assert_not_called()
        self.assertEqual(result["planned_runs"], 72)
        self.assertEqual(result["selection"]["status"], "preview")

    def test_preview_cannot_be_used_for_training(self):
        with patch.object(runner, "run_grid") as train, self.assertRaises(ValueError):
            runner.run_confirmation(preview_weight=0.25)
        train.assert_not_called()

    def test_valid_manifest_is_audited_without_rewriting_the_decision(self):
        records = self.records()
        with tempfile.TemporaryDirectory() as directory:
            path = self.manifest(directory, records)
            original = path.read_bytes()
            with patch.object(screening, "collect_screening_results", return_value=records):
                result = confirmation.load_screening_selection(path)
            self.assertEqual(result["selected_pair_loss_weight"], 0.0)
            self.assertEqual(result["selection_sha256"], screening._hash_file(path))
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_protocol_decision_and_provenance_are_rejected(self):
        records = self.records()
        mutations = [
            lambda obj: obj.update(completed_runs=107),
            lambda obj: obj["rule"].update(selection_split="test"),
            lambda obj: obj.update(selected_pair_loss_weight=0.25),
            lambda obj: obj.update(selection_source_sha256="stale"),
            lambda obj: obj["inputs"][0].update(metrics_sha256="stale"),
            lambda obj: obj.update(inputs=obj["inputs"][:-1]),
            lambda obj: obj.update(best_validation_f1_macro=1.0),
            lambda obj: obj["outputs"].pop("runs"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.manifest(directory, records)
            original = path.read_bytes()
            with patch.object(screening, "collect_screening_results", return_value=records):
                for mutation in mutations:
                    obj = json.loads(original)
                    mutation(obj)
                    write_json(obj, path)
                    with self.assertRaises((RuntimeError, ValueError)):
                        confirmation.load_screening_selection(path)
            path.write_bytes(original)

    def test_modified_tables_are_rejected_even_if_their_hash_is_updated(self):
        records = self.records()
        with tempfile.TemporaryDirectory() as directory:
            path = self.manifest(directory, records)
            obj = json.loads(path.read_text())
            table_path = Path(directory) / "screening_by_lambda.csv"
            table = pd.read_csv(table_path)
            table.loc[0, "f1_macro_mean"] = 0.95
            table.to_csv(table_path, index=False)
            with patch.object(screening, "collect_screening_results", return_value=records):
                with self.assertRaises(RuntimeError):
                    confirmation.load_screening_selection(path)
                obj["outputs"]["by_lambda"]["sha256"] = screening._hash_file(table_path)
                write_json(obj, path)
                with self.assertRaises(RuntimeError):
                    confirmation.load_screening_selection(path)

    def test_changed_input_artifacts_invalidate_the_saved_selection(self):
        records = self.records()
        with tempfile.TemporaryDirectory() as directory:
            path = self.manifest(directory, records)
            changed = records.copy()
            changed.loc[0, "validation_predictions_sha256"] = "changed"
            with patch.object(screening, "collect_screening_results", return_value=changed), self.assertRaises(RuntimeError):
                confirmation.load_screening_selection(path)

    def test_selection_is_validated_before_dispatch_and_failures_block_training(self):
        with patch.object(runner, "load_screening_selection", side_effect=RuntimeError("invalid selection")), patch.object(
            runner, "run_grid",
        ) as train, patch.object(runner, "write_json") as write, self.assertRaises(RuntimeError):
            runner.run_confirmation()
        train.assert_not_called()
        write.assert_not_called()

    def test_confirmation_records_the_decision_and_audits_all_completed_runs(self):
        decision = {"selected_pair_loss_weight": 0.25, "selection_path": "selection.json", "selection_sha256": "frozen"}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner, "load_screening_selection", return_value=decision,
        ), patch.object(runner, "run_grid", return_value=self.grid_result(0.25)) as grid, patch.object(
            runner, "audit_confirmation_runs",
        ) as audit:
            result = runner.run_confirmation(output_root=directory, max_retries=2)
            plan = json.loads((Path(directory) / "lambda_0p25" / "confirmation_plan.json").read_text())
            report = json.loads(Path(result["confirmation_result_path"]).read_text())
        self.assertEqual(plan["selection"]["selection_sha256"], "frozen")
        self.assertEqual(len(plan["tasks"]), 72)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["artifacts_verified"])
        self.assertEqual(grid.call_args.kwargs["pair_loss_weights"], (0.25,))
        self.assertEqual(grid.call_args.kwargs["max_retries"], 2)
        audit.assert_called_once_with(runner.build_confirmation_grid(0.25))

    def test_training_and_artifact_failures_are_recorded_as_failures(self):
        decision = {"selected_pair_loss_weight": 0.25}
        for result, error in ((self.grid_result(0.25, failures=1), None), (None, RuntimeError("training failed"))):
            with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, "load_screening_selection", return_value=decision,
            ), patch.object(runner, "run_grid", return_value=result, side_effect=error), patch.object(
                runner, "audit_confirmation_runs",
            ) as audit:
                if error is None:
                    runner.run_confirmation(output_root=directory)
                else:
                    with self.assertRaises(RuntimeError):
                        runner.run_confirmation(output_root=directory)
                report = json.loads((Path(directory) / "lambda_0p25" / "confirmation_result.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["artifacts_verified"])
                audit.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner, "load_screening_selection", return_value=decision,
        ), patch.object(runner, "run_grid", return_value=self.grid_result(0.25)), patch.object(
            runner, "audit_confirmation_runs", side_effect=RuntimeError("missing artifacts"),
        ):
            with self.assertRaises(RuntimeError):
                runner.run_confirmation(output_root=directory)
            report = json.loads((Path(directory) / "lambda_0p25" / "confirmation_result.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["artifacts_verified"])

    def test_artifact_audit_checks_current_fingerprints_for_all_runs(self):
        tasks = runner.build_confirmation_grid(0.25)
        with patch.object(runner, "build_experiment_fingerprint", return_value={"experiment_id": "current"}) as fingerprint, patch.object(
            runner, "is_run_completed", return_value=True,
        ) as completed:
            runner.audit_confirmation_runs(tasks)
        self.assertEqual(fingerprint.call_count, 72)
        self.assertEqual(completed.call_count, 72)
        self.assertTrue(all(call.kwargs["expected_experiment_id"] == "current" for call in completed.call_args_list))
        with patch.object(runner, "build_experiment_fingerprint", return_value={"experiment_id": "current"}), patch.object(
            runner, "is_run_completed", return_value=False,
        ), self.assertRaises(RuntimeError):
            runner.audit_confirmation_runs(tasks)

    def test_cli_preview_lists_72_runs_and_rejects_a_training_override(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.experiments.run_confirmation", "--dry-run", "--preview-weight", "0.25"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Selection status: preview", result.stdout)
        self.assertIn("Planned runs: 72", result.stdout)
        self.assertEqual(len([line for line in result.stdout.splitlines() if line.startswith("[")]), 72)
        with patch.object(sys, "argv", ["run_confirmation", "--preview-weight", "0.25"]), redirect_stdout(io.StringIO()), patch(
            "sys.stderr", new_callable=io.StringIO,
        ), self.assertRaises(SystemExit) as error:
            runner.parse_args()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
