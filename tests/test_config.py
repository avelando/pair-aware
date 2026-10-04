import os
import subprocess
import sys
import unittest

from src import config, paths


class ConfigTest(unittest.TestCase):
    def _load_config(self, precision=None):
        environment = os.environ.copy()
        environment.pop("PAIR_AWARE_PRECISION", None)

        if precision is not None:
            environment["PAIR_AWARE_PRECISION"] = precision

        return subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.config import TRAINING_PRECISION\n"
                "print(TRAINING_PRECISION)\n",
            ],
            cwd=paths.PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_default_precision_is_bf16(self):
        result = self._load_config()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "bf16")

    def test_explicit_bf16_precision_is_accepted(self):
        result = self._load_config("BF16")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "bf16")

    def test_non_bf16_precision_is_rejected(self):
        for precision in ("fp16", "fp32", "invalid"):
            with self.subTest(precision=precision):
                result = self._load_config(precision)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "Invalid training precision",
                    result.stderr,
                )

    def test_v3_outputs_are_isolated_from_v2(self):
        expected_root = (
            paths.PROJECT_ROOT
            / "results_v3"
            / "bf16"
        )

        self.assertEqual(config.EXPERIMENT_VERSION, "v3")
        self.assertEqual(paths.RESULTS_ROOT, expected_root)

        outputs = (
            paths.get_run_dir("true_pair", 13, 40),
            paths.SUMMARY_PATH,
            paths.SUMMARY_BY_SPLIT_PATH,
            paths.SUMMARY_BY_METHOD_PATH,
            paths.STATISTICS_DESCRIPTIVE_PATH,
            paths.STATISTICS_OMNIBUS_PATH,
            paths.STATISTICS_PAIRWISE_PATH,
            paths.STATISTICS_METADATA_PATH,
        )

        for output in outputs:
            with self.subTest(output=output):
                self.assertTrue(output.is_relative_to(expected_root))

    def test_v3_preserves_v2_training_protocol(self):
        expected = {
            "MODEL_NAME": "neuralmind/bert-large-portuguese-cased",
            "MODEL_REVISION": "aa302f6ea73b759f7df9cad58bd272127b67ec28",
            "SPLIT_SEEDS": (13, 21, 40, 42, 73, 101),
            "MODEL_SEEDS": (13, 21, 40, 42, 73, 101),
            "MAX_LENGTH": 256,
            "NUM_EPOCHS": 6,
            "LEARNING_RATE": 2e-5,
            "INSTANCE_TRAIN_BATCH_SIZE": 8,
            "PAIR_BATCH_SIZE": 4,
            "EVAL_BATCH_SIZE": 8,
            "WEIGHT_DECAY": 0.01,
            "WARMUP_RATIO": 0.1,
            "MAX_GRAD_NORM": 1.0,
            "EARLY_STOPPING_PATIENCE": 2,
            "METRIC_FOR_BEST_MODEL": "f1_macro",
            "LR_SCHEDULER_TYPE": "linear",
        }

        actual = {
            name: getattr(config, name)
            for name in expected
        }

        self.assertEqual(actual, expected)

    def test_screening_defines_108_runs(self):
        self.assertEqual(
            config.PAIR_LOSS_WEIGHTS,
            (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
        )

        self.assertEqual(config.SCREENING_MODEL_SEEDS, (13, 40, 101))
        self.assertTrue(
            set(config.SCREENING_MODEL_SEEDS).issubset(config.MODEL_SEEDS)
        )

        planned_runs = (
            len(config.PAIR_LOSS_WEIGHTS)
            * len(config.SPLIT_SEEDS)
            * len(config.SCREENING_MODEL_SEEDS)
        )

        self.assertEqual(planned_runs, 108)
        self.assertEqual(config.VALIDATION_F1_TOLERANCE, 0.005)

    def test_checkpoints_are_preserved(self):
        self.assertTrue(config.KEEP_CHECKPOINTS)


if __name__ == "__main__":
    unittest.main()