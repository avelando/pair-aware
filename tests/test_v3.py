import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import src.experiments.run_v3 as pipeline
from src.experiments.lifecycle import RunLockedError
from src.paths import PROJECT_ROOT
from src.results.io import write_json


class V3Test(unittest.TestCase):
    def phase_result(self, tasks, skipped=False):
        return {
            "tasks": tasks, "planned_runs": len(tasks), "failed_runs": 0,
            "completed_runs": 0 if skipped else len(tasks), "skipped_runs": len(tasks) if skipped else 0,
        }

    @contextmanager
    def environment(self, reuse=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            selection_path = root / "selection.json"
            content = b'{"selected_pair_loss_weight": 0.25}\n'
            if reuse:
                selection_path.write_bytes(content)
            decision = {
                "selected_pair_loss_weight": 0.25, "selection_path": str(selection_path),
                "selection_sha256": hashlib.sha256(content).hexdigest(),
            }
            events = []

            def screening(**arguments):
                events.append("screening")
                selection_path.write_bytes(content)
                return self.phase_result(pipeline.build_screening_grid())

            def select(path):
                self.assertEqual(path, selection_path)
                events.append("selection")
                return dict(decision)

            def confirmation(**arguments):
                events.append("confirmation")
                self.assertEqual(arguments["selection_path"], selection_path)
                return {
                    **self.phase_result(pipeline.build_confirmation_grid(0.25)), "selection": dict(decision),
                    "confirmation_result_path": str(root / "confirmation_result.json"),
                }

            stack.enter_context(patch.object(pipeline, "DEFAULT_SELECTION_PATH", selection_path))
            screen = stack.enter_context(patch.object(pipeline, "run_screening", side_effect=screening))
            confirm = stack.enter_context(patch.object(pipeline, "run_confirmation", side_effect=confirmation))
            stack.enter_context(patch.object(pipeline, "load_screening_selection", side_effect=select))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield root, selection_path, decision, events, screen, confirm

    def test_preview_lists_180_runs_without_reading_or_writing_results(self):
        with patch.object(pipeline, "run_screening") as screen, patch.object(pipeline, "run_confirmation") as confirm, patch.object(
            pipeline, "load_screening_selection",
        ) as select, patch.object(pipeline, "write_json") as write, patch.object(pipeline, "run_lock") as lock:
            result = pipeline.run_v3(dry_run=True, preview_weight=0.25)
        for mock in (screen, confirm, select, write, lock):
            mock.assert_not_called()
        self.assertEqual(result["planned_runs"], 180)
        self.assertEqual(result["screening"]["planned_runs"], 108)
        self.assertEqual(result["confirmation"]["planned_runs"], 72)
        self.assertEqual(result["execution_mode"], "sequential")
        self.assertTrue(all(task["evaluation_scope"] == "validation" for task in result["screening"]["tasks"]))
        self.assertTrue(all(task["pair_loss_weight"] == 0.25 for task in result["confirmation"]["tasks"]))

    def test_invalid_options_are_rejected_before_training(self):
        for arguments in ({"preview_weight": 0.25}, {"max_retries": -1}, {"max_retries": True}, {"dry_run": True, "preview_weight": 0.3}):
            with patch.object(pipeline, "run_screening") as screen, self.assertRaises(ValueError):
                pipeline.run_v3(**arguments)
            screen.assert_not_called()

    def test_phases_run_in_order_and_only_execution_status_is_generated(self):
        with self.environment() as (root, selection, decision, events, screen, confirm):
            result = pipeline.run_v3(max_retries=2, output_root=root / "pipeline")
            self.assertEqual(events, ["screening", "selection", "confirmation"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completed_runs"], 180)
            self.assertEqual(result["selection"], decision)
            self.assertEqual({path.name for path in (root / "pipeline").iterdir()}, {"run.lock", "pipeline_status.json"})
            saved = json.loads(Path(result["status_path"]).read_text())
            self.assertEqual(saved["phases"]["confirmation"]["planned_runs"], 72)
            self.assertEqual(saved["phase"], "completed")
            self.assertFalse(screen.call_args.kwargs["force"])
            self.assertTrue(confirm.call_args.kwargs["fail_fast"])
            self.assertEqual(confirm.call_args.kwargs["max_retries"], 2)

    def test_existing_selection_is_validated_and_reused_without_screening(self):
        with self.environment(reuse=True) as (root, selection, decision, events, screen, confirm):
            original = selection.read_bytes()
            result = pipeline.run_v3(output_root=root / "pipeline")
            screen.assert_not_called()
            self.assertEqual(events, ["selection", "confirmation"])
            self.assertEqual(selection.read_bytes(), original)
            self.assertEqual(result["skipped_runs"], 108)
            self.assertEqual(result["completed_runs"], 72)

    def test_missing_frozen_selection_blocks_reselection(self):
        with self.environment() as (root, selection, decision, events, screen, confirm):
            write_json({"pipeline_version": 1, "execution_mode": "sequential", "selection": decision}, root / "pipeline" / "pipeline_status.json")
            with self.assertRaises(RuntimeError):
                pipeline.run_v3(output_root=root / "pipeline")
            screen.assert_not_called()
            confirm.assert_not_called()
            state = json.loads((root / "pipeline" / "pipeline_status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["selection"], decision)

    def test_changed_frozen_selection_blocks_confirmation(self):
        with self.environment(reuse=True) as (root, selection, decision, events, screen, confirm):
            previous = {**decision, "selection_sha256": "previous decision"}
            write_json({"pipeline_version": 1, "execution_mode": "sequential", "selection": previous}, root / "pipeline" / "pipeline_status.json")
            with self.assertRaises(RuntimeError):
                pipeline.run_v3(output_root=root / "pipeline")
            confirm.assert_not_called()
            state = json.loads((root / "pipeline" / "pipeline_status.json").read_text())
            self.assertEqual(state["selection"], previous)

    def test_failed_screening_prevents_confirmation(self):
        with self.environment() as (root, selection, decision, events, screen, confirm):
            screen.side_effect = None
            screen.return_value = {"failed_runs": 1}
            with self.assertRaises(RuntimeError):
                pipeline.run_v3(output_root=root / "pipeline")
            confirm.assert_not_called()
            self.assertEqual(json.loads((root / "pipeline" / "pipeline_status.json").read_text())["phase"], "screening")

    def test_interrupted_confirmation_preserves_the_selection_and_can_resume(self):
        with self.environment() as (root, selection, decision, events, screen, confirm):
            original = confirm.side_effect
            confirm.side_effect = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                pipeline.run_v3(output_root=root / "pipeline")
            state = json.loads((root / "pipeline" / "pipeline_status.json").read_text())
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["selection"], decision)
            confirm.side_effect = original
            result = pipeline.run_v3(output_root=root / "pipeline")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(screen.call_count, 1)

    def test_incomplete_confirmation_and_different_selection_are_rejected(self):
        for mutation in (lambda obj: obj.update(completed_runs=71), lambda obj: obj["selection"].update(selected_pair_loss_weight=0.5)):
            with self.environment(reuse=True) as (root, selection, decision, events, screen, confirm):
                result = {**self.phase_result(pipeline.build_confirmation_grid(0.25)), "selection": dict(decision), "confirmation_result_path": "result.json"}
                mutation(result)
                confirm.side_effect = None
                confirm.return_value = result
                with self.assertRaises(RuntimeError):
                    pipeline.run_v3(output_root=root / "pipeline")
                self.assertEqual(json.loads((root / "pipeline" / "pipeline_status.json").read_text())["status"], "failed")

    def test_selection_file_changed_during_confirmation_is_rejected(self):
        with self.environment(reuse=True) as (root, selection, decision, events, screen, confirm):
            original = confirm.side_effect

            def change_selection(**arguments):
                result = original(**arguments)
                selection.write_text("{}\n")
                return result

            confirm.side_effect = change_selection
            with self.assertRaises(RuntimeError):
                pipeline.run_v3(output_root=root / "pipeline")
            state = json.loads((root / "pipeline" / "pipeline_status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["selection"], decision)

    def test_completed_runs_can_all_be_reused(self):
        with self.environment(reuse=True) as (root, selection, decision, events, screen, confirm):
            confirm.side_effect = None
            confirm.return_value = {
                **self.phase_result(pipeline.build_confirmation_grid(0.25), skipped=True),
                "selection": decision, "confirmation_result_path": "result.json",
            }
            result = pipeline.run_v3(output_root=root / "pipeline")
            self.assertEqual(result["completed_runs"], 0)
            self.assertEqual(result["skipped_runs"], 180)
            self.assertEqual(result["status"], "completed")

    def test_pipeline_lock_prevents_two_active_executions(self):
        with self.environment(reuse=True) as (root, selection, decision, events, screen, confirm):
            with pipeline.run_lock(root / "pipeline", experiment_id="v3_sequential_pipeline"):
                with self.assertRaises(RunLockedError):
                    pipeline.run_v3(output_root=root / "pipeline")
            screen.assert_not_called()
            confirm.assert_not_called()

    def test_unknown_status_format_is_preserved_and_rejected(self):
        with self.environment(reuse=True) as (root, selection, decision, events, screen, confirm):
            path = root / "pipeline" / "pipeline_status.json"
            write_json({"pipeline_version": 999}, path)
            original = path.read_bytes()
            with self.assertRaises(RuntimeError):
                pipeline.run_v3(output_root=root / "pipeline")
            self.assertEqual(path.read_bytes(), original)
            confirm.assert_not_called()

    def test_cli_preview_lists_both_grids_without_training_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.experiments.run_v3", "--dry-run", "--preview-weight", "0.25"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Execution mode: sequential", result.stdout)
        self.assertIn("Total planned runs: 180", result.stdout)
        self.assertEqual(len([line for line in result.stdout.splitlines() if line.startswith("[")]), 180)
        with patch.object(sys, "argv", ["run_v3", "--preview-weight", "0.25"]), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            pipeline.parse_args()


    def test_reference_previews_distinguish_requests_from_unique_runs(self):
        with patch.object(pipeline, "run_references") as run, patch.object(pipeline, "write_json") as write:
            for weight, unique in ((0.25, 288), (1.0, 216)):
                result = pipeline.run_v3(dry_run=True, preview_weight=weight, include_references=True)
                self.assertEqual(result["planned_runs"], 288)
                self.assertEqual(result["unique_planned_runs"], unique)
                self.assertEqual(result["references"]["planned_runs"], 108)
        run.assert_not_called()
        write.assert_not_called()

    def test_reference_phase_runs_after_confirmation_with_frozen_selection(self):
        with self.environment() as (root, selection, decision, events, screen, confirm):
            def collect(**arguments):
                events.append("references")
                self.assertEqual(arguments["selection_path"], selection)
                self.assertTrue(arguments["fail_fast"])
                return {
                    **self.phase_result(pipeline.build_reference_grid()), "selection": decision,
                    "reference_result_path": "reference_result.json",
                }

            with patch.object(pipeline, "run_references", side_effect=collect):
                result = pipeline.run_v3(output_root=root / "pipeline", include_references=True)
            self.assertEqual(events, ["screening", "selection", "confirmation", "references"])
            self.assertEqual(result["planned_runs"], 288)
            self.assertEqual(result["completed_runs"], 288)
            self.assertTrue(result["include_references"])
            self.assertEqual(result["phases"]["references"]["planned_runs"], 108)

    def test_interrupted_reference_phase_preserves_selection_on_resume(self):
        with self.environment(reuse=True) as (root, selection, decision, events, screen, confirm):
            with patch.object(pipeline, "run_references", side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.run_v3(output_root=root / "pipeline", include_references=True)
            saved = json.loads((root / "pipeline" / "pipeline_status.json").read_text())
            self.assertEqual(saved["phase"], "references")
            self.assertEqual(saved["selection"], decision)
            with patch.object(pipeline, "run_references", return_value={
                **self.phase_result(pipeline.build_reference_grid(), skipped=True),
                "selection": decision, "reference_result_path": "reference_result.json",
            }):
                result = pipeline.run_v3(output_root=root / "pipeline", include_references=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["skipped_runs"], 216)
            screen.assert_not_called()

    def test_changed_selection_and_incomplete_references_are_rejected(self):
        for mutation in (lambda obj: obj.update(completed_runs=107), lambda obj: obj["selection"].update(selection_sha256="changed")):
            with self.environment(reuse=True) as (root, selection, decision, events, screen, confirm):
                result = {
                    **self.phase_result(pipeline.build_reference_grid()), "selection": dict(decision),
                    "reference_result_path": "reference_result.json",
                }
                mutation(result)
                with patch.object(pipeline, "run_references", return_value=result):
                    with self.assertRaises(RuntimeError):
                        pipeline.run_v3(output_root=root / "pipeline", include_references=True)

    def test_cli_reference_preview_lists_all_three_grids(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.experiments.run_v3", "--dry-run", "--include-references", "--preview-weight", "1.0"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Total planned runs: 288", result.stdout)
        self.assertIn("Unique planned runs: 216", result.stdout)


if __name__ == "__main__":
    unittest.main()
