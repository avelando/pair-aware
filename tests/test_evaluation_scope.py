import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.data.loading as loading_module
import src.data.validation as validation_module
import src.experiments.fingerprint as fingerprint_module
from src.evaluation.artifacts import build_evaluation_metrics, evaluate_run_splits
from src.evaluation.scope import get_training_split_names, validate_evaluation_scope, validate_split_names
from src.paths import RESULTS_ROOT, get_run_dir


class EvaluationScopeTest(unittest.TestCase):
    def test_scopes_select_the_expected_input_splits(self):
        self.assertEqual(get_training_split_names("full"), ("train", "validation", "test"))
        self.assertEqual(get_training_split_names("validation"), ("train", "validation"))

    def test_invalid_scope_and_split_selections_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_evaluation_scope("screening")
        for names in ((), ("train", "train"), ("invalid",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_split_names(names)

    def test_validation_only_directories_are_isolated(self):
        for method in ("instance_level", "true_pair", "shuffled_pair"):
            with self.subTest(method=method):
                full = get_run_dir(method, 13, 40)
                validation = get_run_dir(method, 13, 40, evaluation_scope="validation")
                self.assertEqual(validation, RESULTS_ROOT / "validation_only" / full.relative_to(RESULTS_ROOT))

    def test_scope_changes_the_configuration_hash(self):
        full = fingerprint_module.get_config_snapshot("true_pair")
        validation = fingerprint_module.get_config_snapshot("true_pair", evaluation_scope="validation")
        self.assertEqual(validation, {**full, "evaluation_scope": "validation"})
        self.assertNotEqual(
            fingerprint_module.get_config_hash("true_pair"),
            fingerprint_module.get_config_hash("true_pair", evaluation_scope="validation"),
        )

    def test_loading_train_and_validation_does_not_require_test_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train", "validation"):
                (root / f"{name}.jsonl").write_text(json.dumps({"id": f"{name}.H"}) + "\n")
            with patch.object(loading_module, "get_split_dir", return_value=root):
                data = loading_module.load_split_directory(13, split_names=("train", "validation"))
                self.assertEqual(tuple(data), ("train", "validation"))
                with self.assertRaises(FileNotFoundError):
                    loading_module.load_split_directory(13)

    def test_validation_fingerprint_does_not_read_test_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train", "validation"):
                (root / f"{name}.jsonl").write_text("{}\n")
            (root / "metadata.json").write_text("{}")
            with patch.object(fingerprint_module, "get_split_dir", return_value=root):
                first = fingerprint_module.build_experiment_fingerprint("true_pair", 13, 40, evaluation_scope="validation")
                (root / "test.jsonl").write_text("test content must not affect screening")
                second = fingerprint_module.build_experiment_fingerprint("true_pair", 13, 40, evaluation_scope="validation")
                self.assertEqual(first["experiment_id"], second["experiment_id"])
                (root / "validation.jsonl").write_text('{"changed": true}\n')
                third = fingerprint_module.build_experiment_fingerprint("true_pair", 13, 40, evaluation_scope="validation")
                self.assertNotEqual(second["dataset_hash"], third["dataset_hash"])

    def test_partial_input_validation_uses_only_selected_counts(self):
        validated = [
            {"ids": {"1.H", "1.N"}, "pair_ids": {"1"}},
            {"ids": {"2.H", "2.N"}, "pair_ids": {"2"}},
        ]
        with patch.object(validation_module, "validate_split", side_effect=validated) as validate_mock, patch.multiple(
            validation_module,
            EXPECTED_SPLIT_COUNTS={"train": 2, "validation": 2, "test": 8},
            EXPECTED_PAIR_COUNTS={"train": 1, "validation": 1, "test": 4},
        ):
            result = validation_module.validate_input_splits(
                {"train": "train", "validation": "validation"}, "screening", split_names=("train", "validation")
            )
        self.assertEqual(tuple(result), ("train", "validation"))
        self.assertEqual(validate_mock.call_count, 2)

    def test_validation_only_evaluation_never_requests_test_data(self):
        dataframe = pd.DataFrame({"id": ["1.H", "1.N"], "text": ["h", "n"], "label": [1, 0]})
        requested = []

        def predict(split_name, frame):
            requested.append(split_name)
            return np.array([[0.0, 1.0], [1.0, 0.0]]), frame["label"].to_numpy()

        results = evaluate_run_splits({"validation": dataframe}, predict, evaluation_scope="validation")
        metrics = build_evaluation_metrics(results)
        self.assertEqual(requested, ["validation"])
        self.assertTrue(all(name.startswith("validation_") for name in metrics))
        self.assertEqual(metrics["validation_instance"]["f1_macro"], 1.0)


if __name__ == "__main__":
    unittest.main()
