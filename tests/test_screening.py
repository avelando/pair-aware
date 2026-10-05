import io
import subprocess
import sys
import unittest
from collections import Counter
from contextlib import redirect_stdout
from unittest.mock import patch

import src.experiments.run_grid as grid_module
import src.experiments.run_screening as screening_module
from src.config import PAIR_LOSS_WEIGHTS, SCREENING_MODEL_SEEDS, SPLIT_SEEDS
from src.paths import PROJECT_ROOT, get_run_dir


class ScreeningTest(unittest.TestCase):
    def test_screening_contains_the_exact_108_protocol_combinations(self):
        tasks = screening_module.build_screening_grid()
        expected = {
            (weight, split_seed, model_seed)
            for weight in PAIR_LOSS_WEIGHTS
            for split_seed in SPLIT_SEEDS
            for model_seed in SCREENING_MODEL_SEEDS
        }
        actual = {
            (task["pair_loss_weight"], task["split_seed"], task["model_seed"])
            for task in tasks
        }
        self.assertEqual(len(tasks), 108)
        self.assertEqual(actual, expected)
        self.assertTrue(all(task["method"] == "true_pair" for task in tasks))
        self.assertTrue(all(task["evaluation_scope"] == "validation" for task in tasks))
        self.assertEqual(Counter(task["pair_loss_weight"] for task in tasks), dict.fromkeys(PAIR_LOSS_WEIGHTS, 18))

    def test_screening_paths_are_unique_and_separate_from_full_evaluation(self):
        tasks = screening_module.build_screening_grid()
        paths = {get_run_dir(**task) for task in tasks}
        self.assertEqual(len(paths), 108)
        for task in tasks:
            full_task = {**task, "evaluation_scope": "full"}
            self.assertNotEqual(get_run_dir(**task), get_run_dir(**full_task))

    def test_dry_run_never_dispatches_training(self):
        with patch.object(grid_module, "execute_grid_run") as execute:
            result = screening_module.run_screening(dry_run=True)
        execute.assert_not_called()
        self.assertEqual(result["tasks"], screening_module.build_screening_grid())
        self.assertEqual(result["planned_runs"], 108)
        self.assertEqual(result["results"], [])

    def test_mixed_grid_does_not_repeat_instance_baseline_for_each_weight(self):
        tasks = grid_module.build_grid(
            split_seeds=(13,), model_seeds=(13,), pair_loss_weights=(0.0, 0.5),
        )
        self.assertEqual(len(tasks), 5)
        self.assertEqual(sum(task["method"] == "instance_level" for task in tasks), 1)
        self.assertIsNone(tasks[0]["pair_loss_weight"])

    def test_invalid_weight_grids_are_rejected_before_execution(self):
        for weights in ((), (0, 0.0), (-1,), (float("nan"),), (float("inf"),), (True,)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                grid_module.build_grid(methods=("true_pair",), pair_loss_weights=weights)
        with self.assertRaises(ValueError):
            grid_module.build_grid(methods=("instance_level",), pair_loss_weights=(0.5,))
        with self.assertRaises(ValueError):
            grid_module.build_grid(evaluation_scope="invalid")

    def test_zero_weight_and_validation_scope_reach_pair_runner(self):
        with patch.object(grid_module, "run_pair_aware", return_value={"status": "completed"}) as execute:
            grid_module.execute_grid_run("true_pair", 13, 40, pair_loss_weight=0.0, evaluation_scope="validation")
        execute.assert_called_once_with(
            method="true_pair", split_seed=13, model_seed=40, force=False,
            pair_loss_weight=0.0, evaluation_scope="validation",
        )

    def test_instance_runner_rejects_pair_loss_weight(self):
        with patch.object(grid_module, "run_instance_level") as execute:
            with self.assertRaises(ValueError):
                grid_module.execute_grid_run("instance_level", 13, 40, pair_loss_weight=0.0)
        execute.assert_not_called()

    def test_retries_preserve_weight_and_validation_scope(self):
        task = screening_module.build_screening_grid()[0]
        with patch.object(
            grid_module, "execute_grid_run",
            side_effect=[RuntimeError("temporary failure"), {"status": "completed"}],
        ) as execute, redirect_stdout(io.StringIO()):
            record, error = grid_module.execute_grid_task_with_retries(task, max_retries=1)
        self.assertIsNone(error)
        self.assertEqual(record["retry_count"], 1)
        self.assertEqual(execute.call_count, 2)
        for call in execute.call_args_list:
            self.assertEqual(call.kwargs, {**task, "force": False})

    def test_screening_dispatches_all_tasks_with_validation_scope(self):
        with patch.object(
            grid_module, "execute_grid_run", return_value={"status": "skipped"},
        ) as execute, patch.object(screening_module, "generate_screening_selection") as select, redirect_stdout(io.StringIO()):
            result = screening_module.run_screening()
        select.assert_called_once_with()
        self.assertEqual(execute.call_count, 108)
        self.assertEqual(result["skipped_runs"], 108)
        self.assertEqual(result["failed_runs"], 0)
        for task, record, call in zip(result["tasks"], result["results"], execute.call_args_list, strict=True):
            self.assertEqual(call.kwargs, {**task, "force": False})
            self.assertEqual(record["pair_loss_weight"], task["pair_loss_weight"])
            self.assertEqual(record["evaluation_scope"], "validation")

    def test_grid_cli_accepts_weights_and_scope(self):
        arguments = [
            "run_grid", "--methods", "true_pair", "--pair-loss-weights", "0", "0.25",
            "--evaluation-scope", "validation", "--dry-run",
        ]
        with patch.object(sys, "argv", arguments):
            args = grid_module.parse_args()
        self.assertEqual(args.pair_loss_weights, [0.0, 0.25])
        self.assertEqual(args.evaluation_scope, "validation")
        self.assertTrue(args.dry_run)

    def test_screening_cli_dry_run_lists_all_tasks_without_loading_training(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.experiments.run_screening", "--dry-run"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Planned runs: 108", result.stdout)
        lines = [line for line in result.stdout.splitlines() if line.startswith("[")]
        self.assertEqual(len(lines), 108)
        self.assertTrue(all("evaluation_scope=validation" in line for line in lines))
        self.assertTrue(any("pair_loss_weight=0.0 " in line for line in lines))


if __name__ == "__main__":
    unittest.main()
