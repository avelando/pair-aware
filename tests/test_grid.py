import unittest
from unittest.mock import MagicMock, patch

import src.experiments.run_grid as grid_module


class GridTest(unittest.TestCase):
    def test_default_grid_contains_all_108_unique_runs(self):
        grid = grid_module.build_grid()

        self.assertEqual(
            len(grid),
            108,
        )

        combinations = {
            (
                task["method"],
                task["split_seed"],
                task["model_seed"],
            )
            for task in grid
        }

        self.assertEqual(
            len(combinations),
            108,
        )

        self.assertEqual(
            grid[0],
            {
                "method": "instance_level",
                "split_seed": 13,
                "model_seed": 13,
                "pair_loss_weight": None,
                "evaluation_scope": "full",
            },
        )

        self.assertEqual(
            grid[-1],
            {
                "method": "shuffled_pair",
                "split_seed": 101,
                "model_seed": 101,
                "pair_loss_weight": 1.0,
                "evaluation_scope": "full",
            },
        )

    def test_filtered_grid_preserves_requested_order(self):
        grid = grid_module.build_grid(
            methods=(
                "true_pair",
                "shuffled_pair",
            ),
            split_seeds=(
                13,
            ),
            model_seeds=(
                21,
                40,
            ),
        )

        self.assertEqual(
            grid,
            [
                {
                    "method": "true_pair",
                    "split_seed": 13,
                    "model_seed": 21,
                    "pair_loss_weight": 1.0,
                    "evaluation_scope": "full",
                },
                {
                    "method": "true_pair",
                    "split_seed": 13,
                    "model_seed": 40,
                    "pair_loss_weight": 1.0,
                    "evaluation_scope": "full",
                },
                {
                    "method": "shuffled_pair",
                    "split_seed": 13,
                    "model_seed": 21,
                    "pair_loss_weight": 1.0,
                    "evaluation_scope": "full",
                },
                {
                    "method": "shuffled_pair",
                    "split_seed": 13,
                    "model_seed": 40,
                    "pair_loss_weight": 1.0,
                    "evaluation_scope": "full",
                },
            ],
        )

    def test_invalid_method_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid methods",
        ):
            grid_module.build_grid(
                methods=(
                    "invalid",
                )
            )

    def test_invalid_split_seed_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid split seeds",
        ):
            grid_module.build_grid(
                split_seeds=(
                    999,
                )
            )

    def test_invalid_model_seed_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid model seeds",
        ):
            grid_module.build_grid(
                model_seeds=(
                    999,
                )
            )

    def test_duplicate_selection_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "cannot contain duplicates",
        ):
            grid_module.build_grid(
                model_seeds=(
                    13,
                    13,
                )
            )

    def test_instance_run_dispatches_to_instance_runner(self):
        expected = {
            "status": "completed",
        }

        with patch.object(
            grid_module,
            "run_instance_level",
            return_value=expected,
        ) as instance_mock, patch.object(
            grid_module,
            "run_pair_aware",
        ) as pair_mock:
            result = (
                grid_module.execute_grid_run(
                    method="instance_level",
                    split_seed=13,
                    model_seed=21,
                    force=True,
                )
            )

        self.assertIs(
            result,
            expected,
        )

        instance_mock.assert_called_once_with(
            split_seed=13,
            model_seed=21,
            force=True,
            evaluation_scope="full",
        )

        pair_mock.assert_not_called()

    def test_pair_run_dispatches_to_pair_runner(self):
        expected = {
            "status": "completed",
        }

        with patch.object(
            grid_module,
            "run_instance_level",
        ) as instance_mock, patch.object(
            grid_module,
            "run_pair_aware",
            return_value=expected,
        ) as pair_mock:
            result = (
                grid_module.execute_grid_run(
                    method="true_pair",
                    split_seed=21,
                    model_seed=40,
                    force=False,
                )
            )

        self.assertIs(
            result,
            expected,
        )

        pair_mock.assert_called_once_with(
            method="true_pair",
            split_seed=21,
            model_seed=40,
            force=False,
            pair_loss_weight=1.0,
            evaluation_scope="full",
        )

        instance_mock.assert_not_called()

    def test_dry_run_does_not_execute_training(self):
        with patch.object(
            grid_module,
            "execute_grid_run",
        ) as execute_mock:
            result = grid_module.run_grid(
                methods=(
                    "instance_level",
                ),
                split_seeds=(
                    13,
                ),
                model_seeds=(
                    13,
                    21,
                ),
                dry_run=True,
            )

        self.assertEqual(
            result["planned_runs"],
            2,
        )

        self.assertEqual(
            result["completed_runs"],
            0,
        )

        self.assertEqual(
            result["skipped_runs"],
            0,
        )

        self.assertEqual(
            result["failed_runs"],
            0,
        )

        self.assertEqual(
            result["results"],
            [],
        )

        execute_mock.assert_not_called()

    def test_grid_counts_completed_skipped_and_failed_runs(self):
        with patch.object(
            grid_module,
            "execute_grid_run",
            side_effect=[
                {
                    "status": "completed",
                    "run_dir": "run-1",
                    "experiment_id": "id-1",
                    "attempt": 1,
                },
                {
                    "status": "skipped",
                    "run_dir": "run-2",
                    "experiment_id": "id-2",
                },
                RuntimeError(
                    "training failed"
                ),
            ],
        ):
            result = grid_module.run_grid(
                methods=(
                    "instance_level",
                ),
                split_seeds=(
                    13,
                ),
                model_seeds=(
                    13,
                    21,
                    40,
                ),
            )

        self.assertEqual(
            result["planned_runs"],
            3,
        )

        self.assertEqual(
            result["completed_runs"],
            1,
        )

        self.assertEqual(
            result["skipped_runs"],
            1,
        )

        self.assertEqual(
            result["failed_runs"],
            1,
        )

        self.assertEqual(
            len(
                result["results"]
            ),
            3,
        )

        failed = result[
            "results"
        ][2]

        self.assertEqual(
            failed["status"],
            "failed",
        )

        self.assertEqual(
            failed["error_type"],
            "RuntimeError",
        )

        self.assertEqual(
            failed["error_message"],
            "training failed",
        )

    def test_fail_fast_propagates_first_failure(self):
        with patch.object(
            grid_module,
            "execute_grid_run",
            side_effect=RuntimeError(
                "training failed"
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "training failed",
            ):
                grid_module.run_grid(
                    methods=(
                        "instance_level",
                    ),
                    split_seeds=(
                        13,
                    ),
                    model_seeds=(
                        13,
                    ),
                    fail_fast=True,
                )

    def test_sequential_grid_retries_failed_run_then_completes(self):
        with patch.object(
            grid_module,
            "execute_grid_run",
            side_effect=[
                RuntimeError(
                    "transient failure"
                ),
                {
                    "status": "completed",
                    "run_dir": "run-1",
                    "experiment_id": "id-1",
                    "attempt": 2,
                },
            ],
        ) as execute_mock:
            result = grid_module.run_grid(
                methods=(
                    "instance_level",
                ),
                split_seeds=(
                    13,
                ),
                model_seeds=(
                    13,
                ),
                max_retries=1,
            )

        self.assertEqual(
            execute_mock.call_count,
            2,
        )

        self.assertEqual(
            result["completed_runs"],
            1,
        )

        self.assertEqual(
            result["failed_runs"],
            0,
        )

        self.assertEqual(
            result["retried_runs"],
            1,
        )

        self.assertEqual(
            result["retry_attempts"],
            1,
        )

        self.assertEqual(
            result["results"][0][
                "retry_count"
            ],
            1,
        )

    def test_sequential_grid_stops_after_retry_budget(self):
        with patch.object(
            grid_module,
            "execute_grid_run",
            side_effect=[
                RuntimeError(
                    "first failure"
                ),
                RuntimeError(
                    "second failure"
                ),
            ],
        ) as execute_mock:
            result = grid_module.run_grid(
                methods=(
                    "instance_level",
                ),
                split_seeds=(
                    13,
                ),
                model_seeds=(
                    13,
                ),
                max_retries=1,
            )

        self.assertEqual(
            execute_mock.call_count,
            2,
        )

        self.assertEqual(
            result["completed_runs"],
            0,
        )

        self.assertEqual(
            result["failed_runs"],
            1,
        )

        self.assertEqual(
            result["retry_attempts"],
            1,
        )

        self.assertEqual(
            result["results"][0][
                "error_message"
            ],
            "second failure",
        )

    def test_negative_max_retries_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_retries must be a non-negative integer",
        ):
            grid_module.run_grid(
                methods=(
                    "instance_level",
                ),
                split_seeds=(
                    13,
                ),
                model_seeds=(
                    13,
                ),
                max_retries=-1,
            )

if __name__ == "__main__":
    unittest.main()
