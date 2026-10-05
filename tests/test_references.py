import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.experiments.run_references as references
from src.config import MODEL_SEEDS, SPLIT_SEEDS
from src.experiments.lifecycle import RunLockedError
from src.paths import PROJECT_ROOT


class ReferencesTest(unittest.TestCase):
    def result(self, skipped=0):
        tasks = references.build_reference_grid()
        return {
            "tasks": tasks, "planned_runs": len(tasks), "failed_runs": 0,
            "completed_runs": len(tasks) - skipped, "skipped_runs": skipped,
        }

    def test_grid_preserves_all_seeds_and_lambda_one_controls(self):
        tasks = references.build_reference_grid()
        self.assertEqual(len(tasks), 108)
        for method in references.REFERENCE_METHODS:
            subset = [task for task in tasks if task["method"] == method]
            self.assertEqual(len(subset), 36)
            self.assertEqual({(task["split_seed"], task["model_seed"]) for task in subset}, {
                (split_seed, model_seed) for split_seed in SPLIT_SEEDS for model_seed in MODEL_SEEDS
            })
            self.assertEqual({task["evaluation_scope"] for task in subset}, {"full"})
            self.assertEqual({task["pair_loss_weight"] for task in subset}, {None} if method == "instance_level" else {1.0})

    def test_preview_does_not_read_selection_or_write_results(self):
        with patch.object(references, "load_screening_selection") as select, patch.object(
            references, "write_json",
        ) as write, patch.object(references, "run_lock") as lock:
            result = references.run_references(dry_run=True)
        self.assertEqual(result["planned_runs"], 108)
        for mock in (select, write, lock):
            mock.assert_not_called()

    def test_invalid_retries_are_rejected_before_selection_or_execution(self):
        with patch.object(references, "run_grid") as grid, patch.object(references, "load_screening_selection") as select:
            with self.assertRaises(ValueError):
                references.run_references(max_retries=-1)
        grid.assert_not_called()
        select.assert_not_called()

    def execute(self, root, grid_result=None, grid_error=None, audit_error=None):
        selection = root / "selection.json"
        selection.write_bytes(b"frozen decision\n")
        decision = {"selected_pair_loss_weight": 0.25, "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest()}
        with patch.object(references, "load_screening_selection", return_value=decision) as select, patch.object(
            references, "run_grid", return_value=self.result() if grid_result is None else grid_result, side_effect=grid_error,
        ) as grid, patch.object(references, "audit_reference_runs", return_value=[{"experiment_id": "verified"}], side_effect=audit_error) as audit:
            result = references.run_references(selection_path=selection, max_retries=2, output_root=root / "references")
        select.assert_called_once_with(selection)
        self.assertFalse(grid.call_args.kwargs["force"])
        self.assertEqual(grid.call_args.kwargs["max_retries"], 2)
        return result, audit

    def test_results_record_frozen_selection_and_audited_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, audit = self.execute(root, grid_result=self.result(skipped=72))
            audit.assert_called_once_with(references.build_reference_grid())
            saved = json.loads(Path(result["reference_result_path"]).read_text())
            self.assertEqual(saved["status"], "completed")
            self.assertTrue(saved["artifacts_verified"])
            self.assertEqual(saved["skipped_runs"], 72)
            self.assertEqual(saved["completed_runs"], 36)
            self.assertEqual(saved["selection"]["selected_pair_loss_weight"], 0.25)
            self.assertEqual({path.name for path in (root / "references").iterdir()}, {"run.lock", "reference_plan.json", "reference_result.json"})

    def test_missing_selection_blocks_test_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            references, "load_screening_selection", side_effect=FileNotFoundError(),
        ), patch.object(references, "run_grid") as grid:
            with self.assertRaises(FileNotFoundError):
                references.run_references(output_root=Path(directory) / "references")
        grid.assert_not_called()

    def test_incomplete_grid_and_invalid_artifacts_cannot_be_reported_as_completed(self):
        for arguments in ({"grid_result": {**self.result(), "completed_runs": 107}}, {"audit_error": RuntimeError("invalid artifacts")}):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(RuntimeError):
                    self.execute(root, **arguments)
                saved = json.loads((root / "references" / "reference_result.json").read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertFalse(saved["artifacts_verified"])

    def test_interruption_is_recorded_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                self.execute(root, grid_error=KeyboardInterrupt())
            saved = json.loads((root / "references" / "reference_result.json").read_text())
            self.assertEqual(saved["status"], "interrupted")
            result, audit = self.execute(root, grid_result=self.result(skipped=108))
            self.assertEqual(result["skipped_runs"], 108)

    def test_changed_selection_is_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def change_selection(**arguments):
                (root / "selection.json").write_text("changed decision")
                return self.result()

            with self.assertRaises(RuntimeError):
                self.execute(root, grid_error=change_selection)
            self.assertFalse(json.loads((root / "references" / "reference_result.json").read_text())["artifacts_verified"])

    def test_artifact_audit_uses_current_fingerprint_for_every_method(self):
        tasks = [references.build_reference_grid()[index] for index in (0, 36, 72)]
        with patch.object(references, "build_experiment_fingerprint", return_value={"experiment_id": "current"}) as fingerprint, patch.object(
            references, "is_run_completed", return_value=True,
        ) as complete:
            records = references.audit_reference_runs(tasks)
            self.assertEqual([call.kwargs for call in fingerprint.call_args_list], tasks)
            self.assertEqual(len(records), 3)
            self.assertTrue(all(call.kwargs["expected_experiment_id"] == "current" for call in complete.call_args_list))
            complete.return_value = False
            with self.assertRaises(RuntimeError):
                references.audit_reference_runs(tasks)

    def test_reference_lock_prevents_concurrent_collection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(references, "run_grid") as grid:
            root = Path(directory)
            with references.run_lock(root, experiment_id="v3_reference_collection"):
                with self.assertRaises(RunLockedError):
                    references.run_references(output_root=root)
        grid.assert_not_called()

    def test_cli_preview_does_not_require_training_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.experiments.run_references", "--dry-run"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Planned runs: 108", result.stdout)


if __name__ == "__main__":
    unittest.main()
