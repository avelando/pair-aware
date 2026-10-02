import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.experiments.fingerprint as fingerprint_module


class FingerprintTest(unittest.TestCase):
    def test_instance_config_snapshot_contains_instance_batch_size(self):
        snapshot = fingerprint_module.get_config_snapshot(
            "instance_level"
        )

        self.assertEqual(
            snapshot["method"],
            "instance_level",
        )

        self.assertEqual(
            snapshot["train_batch_size"],
            fingerprint_module.INSTANCE_TRAIN_BATCH_SIZE,
        )

        self.assertNotIn(
            "pair_batch_size",
            snapshot,
        )

    def test_pair_config_snapshot_contains_pair_parameters(self):
        snapshot = fingerprint_module.get_config_snapshot(
            "true_pair"
        )

        self.assertEqual(
            snapshot["pair_batch_size"],
            fingerprint_module.PAIR_BATCH_SIZE,
        )

        self.assertEqual(
            snapshot["pair_loss_weight"],
            fingerprint_module.PAIR_LOSS_WEIGHT,
        )

        self.assertNotIn(
            "train_batch_size",
            snapshot,
        )

    def test_invalid_method_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid experiment method",
        ):
            fingerprint_module.get_config_snapshot(
                "invalid"
            )

    def test_config_hash_changes_when_configuration_changes(self):
        first = fingerprint_module.get_config_hash(
            "instance_level"
        )

        with patch.object(
            fingerprint_module,
            "LEARNING_RATE",
            9e-5,
        ):
            second = fingerprint_module.get_config_hash(
                "instance_level"
            )

        self.assertNotEqual(
            first,
            second,
        )

    def test_pairing_methods_have_different_config_hashes(self):
        true_pair_hash = fingerprint_module.get_config_hash(
            "true_pair"
        )

        shuffled_pair_hash = fingerprint_module.get_config_hash(
            "shuffled_pair"
        )

        self.assertNotEqual(
            true_pair_hash,
            shuffled_pair_hash,
        )

    def test_dataset_fingerprint_is_deterministic_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            split_dir = Path(directory)

            for split_name in fingerprint_module.SPLIT_NAMES:
                (
                    split_dir
                    / f"{split_name}.jsonl"
                ).write_text(
                    f"{split_name}\n",
                    encoding="utf-8",
                )

            metadata_path = (
                split_dir
                / "metadata.json"
            )

            metadata_path.write_text(
                "{}\n",
                encoding="utf-8",
            )

            with patch.object(
                fingerprint_module,
                "get_split_dir",
                return_value=split_dir,
            ):
                first = fingerprint_module.get_dataset_fingerprint(
                    13
                )

                second = fingerprint_module.get_dataset_fingerprint(
                    13
                )

                metadata_path.write_text(
                    '{"changed": true}\n',
                    encoding="utf-8",
                )

                third = fingerprint_module.get_dataset_fingerprint(
                    13
                )

        self.assertEqual(
            first,
            second,
        )

        self.assertNotEqual(
            first,
            third,
        )

    def test_missing_dataset_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            split_dir = Path(directory)

            with patch.object(
                fingerprint_module,
                "get_split_dir",
                return_value=split_dir,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "Fingerprint file not found",
                ):
                    fingerprint_module.get_dataset_fingerprint(
                        13
                    )

    def test_source_fingerprint_changes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.py"
            second_path = root / "second.py"

            first_path.write_text(
                "value = 1\n",
                encoding="utf-8",
            )

            second_path.write_text(
                "value = 2\n",
                encoding="utf-8",
            )

            with patch.object(
                fingerprint_module,
                "PROJECT_ROOT",
                root,
            ), patch.object(
                fingerprint_module,
                "_source_paths",
                return_value=[
                    first_path,
                    second_path,
                ],
            ):
                first = fingerprint_module.get_source_fingerprint(
                    "instance_level"
                )

                second_path.write_text(
                    "value = 3\n",
                    encoding="utf-8",
                )

                second = fingerprint_module.get_source_fingerprint(
                    "instance_level"
                )

        self.assertNotEqual(
            first,
            second,
        )

    def test_experiment_id_is_deterministic_and_seed_sensitive(self):
        with patch.object(
            fingerprint_module,
            "get_dataset_fingerprint",
            return_value="dataset-hash",
        ), patch.object(
            fingerprint_module,
            "get_source_fingerprint",
            return_value="source-hash",
        ):
            first = fingerprint_module.build_experiment_fingerprint(
                "instance_level",
                13,
                40,
            )

            second = fingerprint_module.build_experiment_fingerprint(
                "instance_level",
                13,
                40,
            )

            third = fingerprint_module.build_experiment_fingerprint(
                "instance_level",
                13,
                42,
            )

        self.assertEqual(
            first,
            second,
        )

        self.assertNotEqual(
            first["experiment_id"],
            third["experiment_id"],
        )

        self.assertEqual(
            len(first["experiment_id"]),
            64,
        )

        self.assertEqual(
            first["fingerprint_version"],
            1,
        )


if __name__ == "__main__":
    unittest.main()