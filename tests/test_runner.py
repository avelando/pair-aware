import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import src.experiments.runner as runner_module


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(
            "results/instance_level/split_13/model_seed_13"
        )

        self.experiment_fingerprint = {
            "experiment_id": "experiment-id",
            "fingerprint_version": 1,
            "config_hash": "config-hash",
            "dataset_hash": "dataset-hash",
            "source_hash": "source-hash",
            "config": {
                "method": "instance_level",
            },
        }

    def test_completed_run_is_skipped_before_lock(self):
        execute = MagicMock()

        with patch.object(
            runner_module,
            "get_run_dir",
            return_value=self.run_dir,
        ), patch.object(
            runner_module,
            "build_experiment_fingerprint",
            return_value=self.experiment_fingerprint,
        ), patch.object(
            runner_module,
            "is_run_completed",
            return_value=True,
        ) as completed_mock, patch.object(
            runner_module,
            "run_lock",
        ) as lock_mock:
            result = runner_module.run_experiment(
                method="instance_level",
                split_seed=13,
                model_seed=13,
                execute=execute,
            )

        self.assertEqual(
            result,
            {
                "status": "skipped",
                "method": "instance_level",
                "split_seed": 13,
                "model_seed": 13,
                "run_dir": str(self.run_dir),
                "experiment_id": "experiment-id",
                "evaluation_scope": "full",
            },
        )

        completed_mock.assert_called_once_with(
            self.run_dir,
            expected_experiment_id="experiment-id",
        )

        lock_mock.assert_not_called()
        execute.assert_not_called()

    def test_completion_is_rechecked_after_lock(self):
        execute = MagicMock()

        with patch.object(
            runner_module,
            "get_run_dir",
            return_value=self.run_dir,
        ), patch.object(
            runner_module,
            "build_experiment_fingerprint",
            return_value=self.experiment_fingerprint,
        ), patch.object(
            runner_module,
            "is_run_completed",
            side_effect=[
                False,
                True,
            ],
        ) as completed_mock, patch.object(
            runner_module,
            "run_lock",
        ) as lock_mock, patch.object(
            runner_module,
            "start_run_attempt",
        ) as start_mock:
            result = runner_module.run_experiment(
                method="instance_level",
                split_seed=13,
                model_seed=13,
                execute=execute,
            )

        self.assertEqual(
            result["status"],
            "skipped",
        )

        self.assertEqual(
            completed_mock.call_count,
            2,
        )

        lock_mock.assert_called_once_with(
            self.run_dir,
            "experiment-id",
        )

        start_mock.assert_not_called()
        execute.assert_not_called()

    def test_successful_run_records_completed_attempt(self):
        execute = MagicMock(
            return_value={
                "status": "completed",
            }
        )

        with patch.object(
            runner_module,
            "get_run_dir",
            return_value=self.run_dir,
        ), patch.object(
            runner_module,
            "build_experiment_fingerprint",
            return_value=self.experiment_fingerprint,
        ), patch.object(
            runner_module,
            "is_run_completed",
            return_value=False,
        ), patch.object(
            runner_module,
            "run_lock",
        ), patch.object(
            runner_module,
            "start_run_attempt",
            return_value=3,
        ), patch.object(
            runner_module,
            "complete_run_attempt",
        ) as complete_mock, patch.object(
            runner_module,
            "fail_run_attempt",
        ) as fail_mock, patch.object(
            runner_module,
            "perf_counter",
            side_effect=[
                10.0,
                15.0,
            ],
        ):
            result = runner_module.run_experiment(
                method="instance_level",
                split_seed=13,
                model_seed=13,
                execute=execute,
            )

        self.assertEqual(
            result["attempt"],
            3,
        )

        execute.assert_called_once_with(
            split_seed=13,
            model_seed=13,
            run_dir=self.run_dir,
            experiment_fingerprint=(
                self.experiment_fingerprint
            ),
            attempt=3,
        )

        complete_mock.assert_called_once_with(
            run_dir=self.run_dir,
            experiment_id="experiment-id",
            attempt=3,
            duration_seconds=5.0,
        )

        fail_mock.assert_not_called()

    def test_failed_run_records_failed_attempt(self):
        error = RuntimeError(
            "execution failed"
        )

        execute = MagicMock(
            side_effect=error
        )

        with patch.object(
            runner_module,
            "get_run_dir",
            return_value=self.run_dir,
        ), patch.object(
            runner_module,
            "build_experiment_fingerprint",
            return_value=self.experiment_fingerprint,
        ), patch.object(
            runner_module,
            "is_run_completed",
            return_value=False,
        ), patch.object(
            runner_module,
            "run_lock",
        ), patch.object(
            runner_module,
            "start_run_attempt",
            return_value=2,
        ), patch.object(
            runner_module,
            "complete_run_attempt",
        ) as complete_mock, patch.object(
            runner_module,
            "fail_run_attempt",
        ) as fail_mock, patch.object(
            runner_module,
            "perf_counter",
            side_effect=[
                10.0,
                14.0,
            ],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "execution failed",
            ):
                runner_module.run_experiment(
                    method="instance_level",
                    split_seed=13,
                    model_seed=13,
                    execute=execute,
                )

        fail_mock.assert_called_once_with(
            run_dir=self.run_dir,
            experiment_id="experiment-id",
            attempt=2,
            duration_seconds=4.0,
            error=error,
        )

        complete_mock.assert_not_called()

    def test_force_bypasses_completed_run_checks(self):
        execute = MagicMock(
            return_value={
                "status": "completed",
            }
        )

        with patch.object(
            runner_module,
            "get_run_dir",
            return_value=self.run_dir,
        ), patch.object(
            runner_module,
            "build_experiment_fingerprint",
            return_value=self.experiment_fingerprint,
        ), patch.object(
            runner_module,
            "is_run_completed",
            return_value=True,
        ) as completed_mock, patch.object(
            runner_module,
            "run_lock",
        ) as lock_mock, patch.object(
            runner_module,
            "start_run_attempt",
            return_value=1,
        ), patch.object(
            runner_module,
            "complete_run_attempt",
        ), patch.object(
            runner_module,
            "perf_counter",
            side_effect=[
                1.0,
                2.0,
            ],
        ):
            result = runner_module.run_experiment(
                method="instance_level",
                split_seed=13,
                model_seed=13,
                execute=execute,
                force=True,
            )

        completed_mock.assert_not_called()

        lock_mock.assert_called_once_with(
            self.run_dir,
            "experiment-id",
        )

        self.assertEqual(
            result["attempt"],
            1,
        )

    def test_skip_can_be_disabled_globally(self):
        execute = MagicMock(
            return_value={
                "status": "completed",
            }
        )

        with patch.object(
            runner_module,
            "SKIP_COMPLETED_RUNS",
            False,
        ), patch.object(
            runner_module,
            "get_run_dir",
            return_value=self.run_dir,
        ), patch.object(
            runner_module,
            "build_experiment_fingerprint",
            return_value=self.experiment_fingerprint,
        ), patch.object(
            runner_module,
            "is_run_completed",
            return_value=True,
        ) as completed_mock, patch.object(
            runner_module,
            "run_lock",
        ), patch.object(
            runner_module,
            "start_run_attempt",
            return_value=1,
        ), patch.object(
            runner_module,
            "complete_run_attempt",
        ), patch.object(
            runner_module,
            "perf_counter",
            side_effect=[
                1.0,
                2.0,
            ],
        ):
            result = runner_module.run_experiment(
                method="instance_level",
                split_seed=13,
                model_seed=13,
                execute=execute,
            )

        completed_mock.assert_not_called()

        self.assertEqual(
            result["attempt"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
