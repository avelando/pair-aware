import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import src.experiments.gpu_scheduler as scheduler_module


class FakeProcess:
    def __init__(
        self,
        returncode=None,
    ):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class GpuSchedulerTest(unittest.TestCase):
    def test_query_gpu_memory_converts_mib_to_gib(self):
        completed = SimpleNamespace(
            stdout="32768, 24576\n"
        )

        with patch.object(
            scheduler_module.subprocess,
            "run",
            return_value=completed,
        ):
            result = (
                scheduler_module.query_gpu_memory()
            )

        self.assertEqual(
            result,
            {
                "total_gb": 32.0,
                "free_gb": 24.0,
            },
        )

    def test_query_gpu_memory_rejects_empty_output(self):
        completed = SimpleNamespace(
            stdout=""
        )

        with patch.object(
            scheduler_module.subprocess,
            "run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "no GPU memory information",
            ):
                scheduler_module.query_gpu_memory()

    def test_parallel_settings_require_vram_reservation(self):
        with self.assertRaisesRegex(
            ValueError,
            "vram_per_run_gb is required",
        ):
            scheduler_module.validate_parallel_settings(
                max_parallel=2,
                vram_per_run_gb=None,
                vram_safety_margin_gb=2.0,
                poll_seconds=5.0,
            )

    def test_vram_capacity_accounts_for_scheduler_reservations(self):
        memory = {
            "total_gb": 32.0,
            "free_gb": 30.0,
        }

        self.assertTrue(
            scheduler_module.has_vram_capacity(
                memory=memory,
                vram_per_run_gb=10.0,
                vram_safety_margin_gb=2.0,
                reserved_by_scheduler_gb=10.0,
            )
        )

        self.assertFalse(
            scheduler_module.has_vram_capacity(
                memory=memory,
                vram_per_run_gb=10.0,
                vram_safety_margin_gb=2.0,
                reserved_by_scheduler_gb=20.0,
            )
        )

    def test_instance_command_uses_current_python(self):
        command = (
            scheduler_module.build_run_command(
                {
                    "method": "instance_level",
                    "split_seed": 13,
                    "model_seed": 21,
                }
            )
        )

        self.assertEqual(
            command,
            [
                sys.executable,
                "-m",
                "src.experiments.run_instance_level",
                "--split-seed",
                "13",
                "--model-seed",
                "21",
            ],
        )

    def test_pair_command_includes_method_and_force(self):
        command = (
            scheduler_module.build_run_command(
                {
                    "method": "true_pair",
                    "split_seed": 21,
                    "model_seed": 40,
                },
                force=True,
            )
        )

        self.assertEqual(
            command,
            [
                sys.executable,
                "-m",
                "src.experiments.run_pair_aware",
                "--method",
                "true_pair",
                "--split-seed",
                "21",
                "--model-seed",
                "40",
                "--force",
            ],
        )

    def test_finalize_success_requires_completed_artifacts(self):
        entry = {
            "process": FakeProcess(
                returncode=0
            ),
            "log_file": io.StringIO(),
            "context": {
                "method": "instance_level",
                "split_seed": 13,
                "model_seed": 13,
                "run_dir": Path(
                    "results/run"
                ),
                "log_path": Path(
                    "logs/run.log"
                ),
                "experiment_id": "experiment-id",
            },
            "started_at": 10.0,
        }

        with patch.object(
            scheduler_module,
            "perf_counter",
            return_value=15.0,
        ), patch.object(
            scheduler_module,
            "is_run_completed",
            return_value=True,
        ):
            result = (
                scheduler_module.finalize_run_process(
                    entry
                )
            )

        self.assertEqual(
            result["status"],
            "completed",
        )

        self.assertEqual(
            result["duration_seconds"],
            5.0,
        )

    def test_finalize_nonzero_exit_is_failed(self):
        entry = {
            "process": FakeProcess(
                returncode=1
            ),
            "log_file": io.StringIO(),
            "context": {
                "method": "true_pair",
                "split_seed": 13,
                "model_seed": 21,
                "run_dir": Path(
                    "results/run"
                ),
                "log_path": Path(
                    "logs/run.log"
                ),
                "experiment_id": "experiment-id",
            },
            "started_at": 10.0,
        }

        with patch.object(
            scheduler_module,
            "perf_counter",
            return_value=12.0,
        ), patch.object(
            scheduler_module,
            "is_run_completed",
        ) as completed_mock:
            result = (
                scheduler_module.finalize_run_process(
                    entry
                )
            )

        self.assertEqual(
            result["status"],
            "failed",
        )

        self.assertEqual(
            result["error_type"],
            "SubprocessError",
        )

        completed_mock.assert_not_called()

    def test_completed_task_is_skipped_without_launching_process(self):
        task = {
            "method": "instance_level",
            "split_seed": 13,
            "model_seed": 13,
        }

        context = {
            **task,
            "run_dir": Path(
                "results/run"
            ),
            "log_path": Path(
                "logs/run.log"
            ),
            "experiment_id": "experiment-id",
        }

        with patch.object(
            scheduler_module,
            "build_task_context",
            return_value=context,
        ), patch.object(
            scheduler_module,
            "is_run_completed",
            return_value=True,
        ), patch.object(
            scheduler_module,
            "query_gpu_memory",
        ) as memory_mock, patch.object(
            scheduler_module,
            "launch_run_process",
        ) as launch_mock, patch.object(
            scheduler_module,
            "perf_counter",
            side_effect=[
                1.0,
                2.0,
            ],
        ):
            result = (
                scheduler_module.run_parallel_grid(
                    tasks=[
                        task
                    ],
                    force=False,
                    fail_fast=False,
                    max_parallel=2,
                    vram_per_run_gb=10.0,
                    poll_seconds=0.01,
                )
            )

        self.assertEqual(
            result["skipped_runs"],
            1,
        )

        memory_mock.assert_not_called()
        launch_mock.assert_not_called()

    def test_vram_capacity_does_not_double_count_materialized_usage(self):
        memory = {
            "total_gb": 32.0,
            "free_gb": 20.0,
        }

        self.assertTrue(
            scheduler_module.has_vram_capacity(
                memory=memory,
                vram_per_run_gb=10.0,
                vram_safety_margin_gb=2.0,
                reserved_by_scheduler_gb=10.0,
                baseline_free_gb=30.0,
            )
        )

        self.assertFalse(
            scheduler_module.has_vram_capacity(
                memory=memory,
                vram_per_run_gb=10.0,
                vram_safety_margin_gb=2.0,
                reserved_by_scheduler_gb=20.0,
                baseline_free_gb=30.0,
            )
        )

    def test_parallel_grid_uses_initial_free_vram_as_reservation_baseline(self):
        task = {
            "method": "instance_level",
            "split_seed": 13,
            "model_seed": 13,
        }

        context = {
            **task,
            "run_dir": Path(
                "results/run"
            ),
            "log_path": Path(
                "logs/run.log"
            ),
            "experiment_id": "experiment-id",
        }

        running_entry = {
            "process": FakeProcess(
                returncode=0
            ),
            "context": context,
            "log_file": io.StringIO(),
            "started_at": 1.0,
        }

        with patch.object(
            scheduler_module,
            "build_task_context",
            return_value=context,
        ), patch.object(
            scheduler_module,
            "is_run_completed",
            return_value=False,
        ), patch.object(
            scheduler_module,
            "query_gpu_memory",
            return_value={
                "total_gb": 32.0,
                "free_gb": 30.0,
            },
        ), patch.object(
            scheduler_module,
            "has_vram_capacity",
            return_value=True,
        ) as capacity_mock, patch.object(
            scheduler_module,
            "launch_run_process",
            return_value=running_entry,
        ), patch.object(
            scheduler_module,
            "finalize_run_process",
            return_value={
                **task,
                "status": "completed",
                "duration_seconds": 1.0,
            },
        ), patch.object(
            scheduler_module,
            "sleep",
        ), patch.object(
            scheduler_module,
            "perf_counter",
            side_effect=[
                1.0,
                2.0,
            ],
        ):
            scheduler_module.run_parallel_grid(
                tasks=[task],
                force=False,
                fail_fast=False,
                max_parallel=2,
                vram_per_run_gb=10.0,
                vram_safety_margin_gb=2.0,
                poll_seconds=0.01,
            )

        capacity_mock.assert_called_once_with(
            memory={
                "total_gb": 32.0,
                "free_gb": 30.0,
            },
            vram_per_run_gb=10.0,
            vram_safety_margin_gb=2.0,
            reserved_by_scheduler_gb=0.0,
            baseline_free_gb=30.0,
        )

    def test_fail_fast_stops_other_running_processes(self):
        first_task = {
            "method": "instance_level",
            "split_seed": 13,
            "model_seed": 13,
        }

        second_task = {
            "method": "instance_level",
            "split_seed": 13,
            "model_seed": 21,
        }

        first_context = {
            **first_task,
            "run_dir": Path(
                "results/run-1"
            ),
            "log_path": Path(
                "logs/run-1.log"
            ),
            "experiment_id": "experiment-1",
        }

        second_context = {
            **second_task,
            "run_dir": Path(
                "results/run-2"
            ),
            "log_path": Path(
                "logs/run-2.log"
            ),
            "experiment_id": "experiment-2",
        }

        first_process = MagicMock()

        first_process.poll.side_effect = [
            None,
            1,
        ]

        second_process = MagicMock()

        second_process.poll.return_value = (
            None
        )

        first_entry = {
            "process": first_process,
            "context": first_context,
            "log_file": io.StringIO(),
            "started_at": 1.0,
        }

        second_entry = {
            "process": second_process,
            "context": second_context,
            "log_file": io.StringIO(),
            "started_at": 1.0,
        }

        with patch.object(
            scheduler_module,
            "build_task_context",
            side_effect=[
                first_context,
                second_context,
            ],
        ), patch.object(
            scheduler_module,
            "is_run_completed",
            return_value=False,
        ), patch.object(
            scheduler_module,
            "query_gpu_memory",
            return_value={
                "total_gb": 32.0,
                "free_gb": 30.0,
            },
        ), patch.object(
            scheduler_module,
            "has_vram_capacity",
            return_value=True,
        ), patch.object(
            scheduler_module,
            "launch_run_process",
            side_effect=[
                first_entry,
                second_entry,
            ],
        ), patch.object(
            scheduler_module,
            "finalize_run_process",
            return_value={
                **first_task,
                "status": "failed",
                "duration_seconds": 1.0,
                "error_message": "training failed",
            },
        ), patch.object(
            scheduler_module,
            "stop_run_process",
        ) as stop_mock, patch.object(
            scheduler_module,
            "sleep",
        ), patch.object(
            scheduler_module,
            "perf_counter",
            return_value=1.0,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Parallel grid stopped after failure",
            ):
                scheduler_module.run_parallel_grid(
                    tasks=[
                        first_task,
                        second_task,
                    ],
                    force=False,
                    fail_fast=True,
                    max_parallel=2,
                    vram_per_run_gb=10.0,
                    vram_safety_margin_gb=2.0,
                    poll_seconds=0.01,
                )

        self.assertEqual(
            stop_mock.call_count,
            2,
        )

        stop_mock.assert_any_call(
            second_entry
        )

if __name__ == "__main__":
    unittest.main()