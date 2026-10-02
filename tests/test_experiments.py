import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

import src.experiments.run_instance_level as instance_module
import src.experiments.run_pair_aware as pair_module


class ExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.experiment_fingerprint = {
            "experiment_id": "experiment-id",
            "fingerprint_version": 1,
            "config_hash": "config-hash",
            "dataset_hash": "dataset-hash",
            "source_hash": "source-hash",
            "config": {
                "method": "test",
            },
        }

    def test_instance_run_seeds_accept_configured_values(self):
        instance_module.validate_run_seeds(
            13,
            101,
        )

    def test_instance_run_rejects_invalid_split_seed(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid split seed",
        ):
            instance_module.validate_run_seeds(
                999,
                13,
            )

    def test_instance_run_rejects_invalid_model_seed(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid model seed",
        ):
            instance_module.validate_run_seeds(
                13,
                999,
            )

    def test_pair_run_arguments_accept_configured_values(self):
        pair_module.validate_run_arguments(
            "true_pair",
            21,
            40,
        )

        pair_module.validate_run_arguments(
            "shuffled_pair",
            21,
            40,
        )

    def test_pair_run_rejects_invalid_method(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid pair-aware method",
        ):
            pair_module.validate_run_arguments(
                "instance_level",
                13,
                13,
            )

    def test_pair_run_rejects_invalid_split_seed(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid split seed",
        ):
            pair_module.validate_run_arguments(
                "true_pair",
                999,
                13,
            )

    def test_pair_run_rejects_invalid_model_seed(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid model seed",
        ):
            pair_module.validate_run_arguments(
                "true_pair",
                13,
                999,
            )

    def test_true_pair_training_uses_true_counterfactual_pairs(self):
        dataframe = pd.DataFrame(
            {
                "id": ["1.H", "1.N"]
            }
        )

        with patch.object(
            pair_module,
            "build_pairs",
            return_value="pairs",
        ) as build_mock:
            result = pair_module.build_training_pairs(
                train_dataframe=dataframe,
                method="true_pair",
                split_seed=73,
            )

        self.assertEqual(
            result,
            "pairs",
        )

        build_mock.assert_called_once_with(
            dataframe,
            "true_pair",
        )

    def test_shuffled_pair_training_uses_split_seed(self):
        dataframe = pd.DataFrame(
            {
                "id": ["1.H", "1.N"]
            }
        )

        with patch.object(
            pair_module,
            "build_pairs",
            return_value="pairs",
        ) as build_mock:
            result = pair_module.build_training_pairs(
                train_dataframe=dataframe,
                method="shuffled_pair",
                split_seed=73,
            )

        self.assertEqual(
            result,
            "pairs",
        )

        build_mock.assert_called_once_with(
            dataframe,
            "shuffled_pair",
            seed=73,
        )

    def test_instance_metadata_records_run_configuration(self):
        split_metadata = {
            "dataset": "test",
        }

        with patch.object(
            instance_module,
            "utc_now",
            return_value="2026-10-01T00:00:00+00:00",
        ), patch.object(
            instance_module,
            "get_git_provenance",
            return_value={
                "git_commit": "abc",
            },
        ), patch.object(
            instance_module,
            "get_environment_metadata",
            return_value={
                "device": "cuda",
            },
        ):
            metadata = instance_module.create_run_metadata(
                split_seed=13,
                model_seed=40,
                run_dir=Path("results/run"),
                split_metadata=split_metadata,
                experiment_fingerprint=(
                    self.experiment_fingerprint
                ),
                device=torch.device("cuda"),
            )

        self.assertEqual(
            metadata["status"],
            "running",
        )

        self.assertEqual(
            metadata["method"],
            "instance_level",
        )

        self.assertEqual(
            metadata["split_seed"],
            13,
        )

        self.assertEqual(
            metadata["model_seed"],
            40,
        )

        self.assertEqual(
            metadata["run_dir"],
            "results/run",
        )

        self.assertEqual(
            metadata["experiment_id"],
            "experiment-id",
        )

        self.assertEqual(
            metadata["fingerprint"]["config_hash"],
            "config-hash",
        )

        self.assertEqual(
            metadata["fingerprint"]["dataset_hash"],
            "dataset-hash",
        )

        self.assertEqual(
            metadata["fingerprint"]["source_hash"],
            "source-hash",
        )

        self.assertEqual(
            metadata["model"]["name"],
            instance_module.MODEL_NAME,
        )

        self.assertEqual(
            metadata["model"]["revision"],
            instance_module.MODEL_REVISION,
        )

        self.assertEqual(
            metadata["training"]["train_batch_size"],
            instance_module.INSTANCE_TRAIN_BATCH_SIZE,
        )

        self.assertEqual(
            metadata["seed_control"]["seed"],
            40,
        )

        self.assertEqual(
            metadata["split_metadata"],
            split_metadata,
        )

    def test_true_pair_metadata_records_pairing_configuration(self):
        with patch.object(
            pair_module,
            "utc_now",
            return_value="2026-10-01T00:00:00+00:00",
        ), patch.object(
            pair_module,
            "get_git_provenance",
            return_value={
                "git_commit": "abc",
            },
        ), patch.object(
            pair_module,
            "get_environment_metadata",
            return_value={
                "device": "cuda",
            },
        ):
            metadata = pair_module.create_run_metadata(
                method="true_pair",
                split_seed=13,
                model_seed=21,
                run_dir=Path("results/run"),
                split_metadata={
                    "dataset": "test",
                },
                train_pair_count=1995,
                experiment_fingerprint=(
                    self.experiment_fingerprint
                ),
                device=torch.device("cuda"),
            )

        pairing = metadata["pairing"]

        self.assertEqual(
            metadata["method"],
            "true_pair",
        )

        self.assertEqual(
            pairing["strategy"],
            "true_pair",
        )

        self.assertEqual(
            pairing["train_pair_count"],
            1995,
        )

        self.assertIsNone(
            pairing["shuffle_seed"]
        )

        self.assertIsNone(
            pairing["shuffle_seed_source"]
        )

        self.assertEqual(
            pairing["pairing_algorithm"],
            "true_counterfactual_pair",
        )

        self.assertTrue(
            pairing[
                "validation_uses_true_pairs"
            ]
        )

        self.assertTrue(
            pairing[
                "test_uses_true_pairs"
            ]
        )

    def test_shuffled_pair_metadata_records_split_seed_pairing(self):
        with patch.object(
            pair_module,
            "utc_now",
            return_value="2026-10-01T00:00:00+00:00",
        ), patch.object(
            pair_module,
            "get_git_provenance",
            return_value={
                "git_commit": "abc",
            },
        ), patch.object(
            pair_module,
            "get_environment_metadata",
            return_value={
                "device": "cuda",
            },
        ):
            metadata = pair_module.create_run_metadata(
                method="shuffled_pair",
                split_seed=42,
                model_seed=101,
                run_dir=Path("results/run"),
                split_metadata={
                    "dataset": "test",
                },
                train_pair_count=1995,
                experiment_fingerprint=(
                    self.experiment_fingerprint
                ),
                device=torch.device("cuda"),
            )

        pairing = metadata["pairing"]

        self.assertEqual(
            pairing["shuffle_seed"],
            42,
        )

        self.assertEqual(
            pairing["shuffle_seed_source"],
            "split_seed",
        )

        self.assertEqual(
            pairing["pairing_algorithm"],
            "uniform_derangement_by_rejection_sampling",
        )

        self.assertEqual(
            metadata["seed_control"]["seed"],
            101,
        )

    def test_instance_completed_run_is_skipped(self):
        run_dir = Path(
            "results/instance_level/split_13/model_seed_13"
        )

        with patch.object(
            instance_module,
            "SKIP_COMPLETED_RUNS",
            True,
        ), patch.object(
            instance_module,
            "get_run_dir",
            return_value=run_dir,
        ), patch.object(
            instance_module,
            "build_experiment_fingerprint",
            return_value=self.experiment_fingerprint,
        ), patch.object(
            instance_module,
            "is_run_completed",
            return_value=True,
        ) as completed_mock, patch.object(
            instance_module,
            "prepare_run_directory",
        ) as prepare_mock:
            result = instance_module.run_instance_level(
                split_seed=13,
                model_seed=13,
            )

        self.assertEqual(
            result,
            {
                "status": "skipped",
                "method": "instance_level",
                "split_seed": 13,
                "model_seed": 13,
                "run_dir": str(run_dir),
                "experiment_id": "experiment-id",
            },
        )

        completed_mock.assert_called_once_with(
            run_dir,
            expected_experiment_id="experiment-id",
        )

        prepare_mock.assert_not_called()

    def test_pair_completed_run_is_skipped(self):
        run_dir = Path(
            "results/true_pair/split_13/model_seed_13"
        )

        with patch.object(
            pair_module,
            "SKIP_COMPLETED_RUNS",
            True,
        ), patch.object(
            pair_module,
            "get_run_dir",
            return_value=run_dir,
        ), patch.object(
            pair_module,
            "build_experiment_fingerprint",
            return_value=self.experiment_fingerprint,
        ), patch.object(
            pair_module,
            "is_run_completed",
            return_value=True,
        ) as completed_mock, patch.object(
            pair_module,
            "prepare_run_directory",
        ) as prepare_mock:
            result = pair_module.run_pair_aware(
                method="true_pair",
                split_seed=13,
                model_seed=13,
            )

        self.assertEqual(
            result,
            {
                "status": "skipped",
                "method": "true_pair",
                "split_seed": 13,
                "model_seed": 13,
                "run_dir": str(run_dir),
                "experiment_id": "experiment-id",
            },
        )

        completed_mock.assert_called_once_with(
            run_dir,
            expected_experiment_id="experiment-id",
        )

        prepare_mock.assert_not_called()

    def test_force_bypasses_instance_completed_run_skip(self):
        run_dir = Path(
            "results/instance_level/split_13/model_seed_13"
        )

        with patch.object(
            instance_module,
            "SKIP_COMPLETED_RUNS",
            True,
        ), patch.object(
            instance_module,
            "get_run_dir",
            return_value=run_dir,
        ), patch.object(
            instance_module,
            "build_experiment_fingerprint",
            return_value=self.experiment_fingerprint,
        ), patch.object(
            instance_module,
            "is_run_completed",
            return_value=True,
        ) as completed_mock, patch.object(
            instance_module,
            "prepare_run_directory",
            return_value={},
        ) as prepare_mock, patch.object(
            instance_module,
            "load_split_directory",
            side_effect=RuntimeError(
                "forced execution"
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "forced execution",
            ):
                instance_module.run_instance_level(
                    split_seed=13,
                    model_seed=13,
                    force=True,
                )

        completed_mock.assert_called_once_with(
            run_dir,
            expected_experiment_id="experiment-id",
        )

        prepare_mock.assert_called_once_with(
            run_dir
        )


if __name__ == "__main__":
    unittest.main()