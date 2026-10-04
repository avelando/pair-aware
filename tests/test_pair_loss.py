import unittest
from unittest.mock import patch

import src.experiments.fingerprint as fingerprint_module
import src.experiments.runner as runner_module
from src.config import PAIR_LOSS_WEIGHTS
from src.pair_loss import (
    format_pair_loss_weight,
    resolve_pair_loss_weight,
    validate_pair_loss_weight,
)
from src.paths import RESULTS_ROOT, get_run_dir


class PairLossParameterTest(unittest.TestCase):
    def test_screening_weights_are_valid(self):
        for weight in PAIR_LOSS_WEIGHTS:
            with self.subTest(weight=weight):
                self.assertEqual(validate_pair_loss_weight(weight), weight)

    def test_invalid_weights_are_rejected(self):
        for weight in (-0.1, float("nan"), float("inf"), -float("inf"), True, None, "invalid"):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                validate_pair_loss_weight(weight)

    def test_pair_methods_resolve_default_and_zero(self):
        for method in ("true_pair", "shuffled_pair"):
            with self.subTest(method=method):
                self.assertEqual(resolve_pair_loss_weight(method), 1.0)
                self.assertEqual(resolve_pair_loss_weight(method, 0.0), 0.0)

    def test_instance_level_rejects_pair_weights(self):
        self.assertIsNone(resolve_pair_loss_weight("instance_level"))
        with self.assertRaises(ValueError):
            resolve_pair_loss_weight("instance_level", 0.0)

    def test_invalid_method_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_pair_loss_weight("invalid")

    def test_labels_are_canonical_and_distinct(self):
        self.assertEqual(format_pair_loss_weight(1), format_pair_loss_weight(1.0))
        self.assertEqual(format_pair_loss_weight(-0.0), "0p0")
        labels = {format_pair_loss_weight(weight) for weight in PAIR_LOSS_WEIGHTS}
        self.assertEqual(len(labels), len(PAIR_LOSS_WEIGHTS))

    def test_pair_run_directories_are_separated_by_weight(self):
        for method in ("true_pair", "shuffled_pair"):
            with self.subTest(method=method):
                directories = {get_run_dir(method, 13, 40, weight) for weight in PAIR_LOSS_WEIGHTS}
                self.assertEqual(len(directories), len(PAIR_LOSS_WEIGHTS))
                self.assertEqual(
                    get_run_dir(method, 13, 40, 0.25),
                    RESULTS_ROOT / method / "lambda_0p25" / "split_13" / "model_seed_40",
                )
                self.assertEqual(get_run_dir(method, 13, 40), get_run_dir(method, 13, 40, 1.0))

    def test_instance_run_directory_is_preserved(self):
        self.assertEqual(
            get_run_dir("instance_level", 13, 40),
            RESULTS_ROOT / "instance_level" / "split_13" / "model_seed_40",
        )

    def test_snapshot_changes_only_pair_loss_weight(self):
        baseline = fingerprint_module.get_config_snapshot("true_pair")
        for weight in PAIR_LOSS_WEIGHTS:
            with self.subTest(weight=weight):
                snapshot = fingerprint_module.get_config_snapshot("true_pair", weight)
                self.assertEqual(snapshot, {**baseline, "pair_loss_weight": weight})

    def test_fingerprints_distinguish_weights(self):
        with patch.object(fingerprint_module, "get_dataset_fingerprint", return_value="dataset"), patch.object(
            fingerprint_module, "get_source_fingerprint", return_value="source"
        ):
            fingerprints = [
                fingerprint_module.build_experiment_fingerprint("true_pair", 13, 40, weight)
                for weight in PAIR_LOSS_WEIGHTS
            ]
        self.assertEqual(len({item["experiment_id"] for item in fingerprints}), len(PAIR_LOSS_WEIGHTS))
        self.assertEqual(len({item["config_hash"] for item in fingerprints}), len(PAIR_LOSS_WEIGHTS))
        self.assertEqual(
            fingerprint_module.get_config_hash("true_pair", 1),
            fingerprint_module.get_config_hash("true_pair", 1.0),
        )

    def test_runner_uses_weight_for_directory_and_fingerprint(self):
        run_dir = get_run_dir("true_pair", 13, 40, 0.0)
        fingerprint = {"experiment_id": "zero-weight"}
        with patch.object(runner_module, "get_run_dir", return_value=run_dir) as directory_mock, patch.object(
            runner_module, "build_experiment_fingerprint", return_value=fingerprint
        ) as fingerprint_mock, patch.object(runner_module, "is_run_completed", return_value=True), patch.object(
            runner_module, "SKIP_COMPLETED_RUNS", True
        ):
            result = runner_module.run_experiment("true_pair", 13, 40, execute=None, pair_loss_weight=0.0)
        directory_mock.assert_called_once_with("true_pair", 13, 40, pair_loss_weight=0.0)
        fingerprint_mock.assert_called_once_with("true_pair", 13, 40, pair_loss_weight=0.0)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["run_dir"], str(run_dir))


if __name__ == "__main__":
    unittest.main()
