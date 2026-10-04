import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

import src.experiments.run_instance_level as instance_module
import src.experiments.run_pair_aware as pair_module
from src.experiments.fingerprint import get_config_snapshot
from src.results.io import get_run_artifact_paths, is_run_completed


class ValidationExportsTest(unittest.TestCase):
    def execute_run(self, method, run_dir, misaligned=False):
        module = instance_module if method == "instance_level" else pair_module
        splits = {}
        for index, name in enumerate(("train", "validation", "test")):
            first = 1 + index * 2
            splits[name] = pd.DataFrame({
                "id": [f"{first}.H", f"{first}.N", f"{first + 1}.H", f"{first + 1}.N"],
                "text": ["h1", "n1", "h2", "n2"],
                "label": [1, 0, 1, 0],
            })

        model = MagicMock()
        model.to.return_value = model
        model.restored = False
        evaluated_splits = []

        def train(**arguments):
            model.restored = True
            Path(arguments["checkpoint_path"]).write_bytes(b"checkpoint")
            return {
                "best_validation_f1_macro": 1.0, "best_epoch": 1, "epochs_trained": 1,
                "training_steps": 1, "skipped_training_steps": 0,
                "planned_training_steps": 6, "warmup_steps": 0,
                "history": [{"epoch": 1, "validation_f1_macro": 1.0}],
            }

        def predict(model, dataloader, device):
            self.assertTrue(model.restored)
            name = next(name for name in ("validation", "test") if dataloader is splits[name])
            evaluated_splits.append(name)
            logits = np.array([[0.0, 3.0], [3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
            labels = splits[name]["label"].to_numpy()
            if name == "test":
                logits[2:] = logits[2:, ::-1]
            elif misaligned:
                labels = labels[::-1]
            return logits, labels

        weight = None if method == "instance_level" else 0.25
        fingerprint = {
            "experiment_id": "export-test", "fingerprint_version": 1,
            "config_hash": "config", "dataset_hash": "data", "source_hash": "source",
            "config": get_config_snapshot(method, weight),
        }

        with ExitStack() as stack:
            values = {
                "load_split_directory": splits, "load_metadata": {},
                "get_device": torch.device("cpu"), "create_sequence_classifier": model,
                "get_git_provenance": {}, "get_environment_metadata": {},
            }
            for name, value in values.items():
                stack.enter_context(patch.object(module, name, return_value=value))
            for name in ("validate_input_splits", "set_model_seed", "clear_memory", "create_tokenizer", "RunProgressTracker"):
                stack.enter_context(patch.object(module, name))
            stack.enter_context(patch.object(module, "InstanceDataset", side_effect=lambda dataframe, **kwargs: dataframe))
            stack.enter_context(patch.object(module, "create_dataloader", side_effect=lambda dataset, **kwargs: dataset))
            stack.enter_context(patch.object(module, "predict_instances", side_effect=predict))
            training_name = "train_instance_level_model" if method == "instance_level" else "train_pair_aware_model"
            stack.enter_context(patch.object(module, training_name, side_effect=train))
            for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            stack.enter_context(patch.multiple(
                "src.results.io", EXPECTED_SPLIT_COUNTS={"validation": 4, "test": 4},
                EXPECTED_PAIR_COUNTS={"validation": 2, "test": 2},
            ))

            arguments = {
                "split_seed": 13, "model_seed": 40, "run_dir": run_dir,
                "experiment_fingerprint": fingerprint, "attempt": 1,
            }
            if method == "instance_level":
                result = module._execute_instance_level(**arguments)
            else:
                stack.enter_context(patch.object(module, "PairDataset", side_effect=lambda pair_dataframe, **kwargs: pair_dataframe))
                stack.enter_context(patch.object(module, "build_training_pairs", return_value=splits["train"]))
                result = module._execute_pair_aware(method=method, pair_loss_weight=weight, **arguments)

            self.assertTrue(is_run_completed(run_dir, expected_experiment_id="export-test"))

        return result, evaluated_splits

    def test_all_methods_export_validation_from_the_restored_model(self):
        for method in ("instance_level", "true_pair", "shuffled_pair"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                result, evaluated = self.execute_run(method, Path(directory))
                paths = get_run_artifact_paths(directory)
                metrics = json.loads(paths["metrics"].read_text())
                predictions = pd.read_csv(paths["validation_predictions"])
                pairs = pd.read_csv(paths["validation_pair_predictions"])
                self.assertEqual(result["status"], "completed")
                self.assertEqual(evaluated, ["validation", "test"])
                self.assertEqual(predictions["id"].tolist(), ["3.H", "3.N", "4.H", "4.N"])
                self.assertEqual(len(pairs), 2)
                self.assertEqual(metrics["validation_instance"]["f1_macro"], 1.0)
                self.assertEqual(metrics["validation_pair"]["pair_ranking_accuracy"], 1.0)
                self.assertEqual(metrics["test_instance"]["f1_macro"], 0.5)

    def test_misaligned_validation_labels_prevent_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                self.execute_run("true_pair", Path(directory), misaligned=True)
            paths = get_run_artifact_paths(directory)
            self.assertFalse(paths["completed"].exists())
            self.assertEqual(json.loads(paths["metadata"].read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
