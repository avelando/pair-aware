import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import src.experiments.preflight as preflight
from src import config
from src.experiments.fingerprint import get_config_snapshot
from src.paths import PROJECT_ROOT
from src.results.io import write_json


class PreflightTest(unittest.TestCase):
    def baseline(self, root):
        for method in config.METHODS:
            snapshot = get_config_snapshot(method)
            snapshot.pop("evaluation_scope")
            snapshot.pop("evaluation_protocol")
            snapshot["experiment_version"] = "v2"
            for split in config.SPLIT_SEEDS:
                for model in config.MODEL_SEEDS:
                    write_json({"status": "completed", "method": method, "split_seed": split, "model_seed": model, "fingerprint": {"config": snapshot}}, root / method / f"split_{split}" / f"model_seed_{model}" / "metadata.json")

    def test_protocol_rejects_hyperparameter_drift_and_missing_baseline_runs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(preflight, "V2_ROOT", Path(directory)):
            root = Path(directory)
            self.baseline(root)
            self.assertEqual(len(preflight.verify_v2_protocol()), 108)
            path = root / "true_pair" / "split_13" / "model_seed_13" / "metadata.json"
            value = json.loads(path.read_text())
            value["fingerprint"]["config"]["learning_rate"] = 1e-3
            write_json(value, path)
            with self.assertRaises(RuntimeError):
                preflight.verify_v2_protocol()
            path.unlink()
            with self.assertRaises(RuntimeError):
                preflight.verify_v2_protocol()

    def test_validation_audit_reads_train_and_validation_without_test_contents(self):
        original = Path.open
        opened = []

        def guarded(path, *args, **kwargs):
            opened.append(path.name)
            if path.name == "test.jsonl":
                raise AssertionError("Test contents were accessed during preflight.")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            records = preflight.verify_validation_data()
        self.assertEqual(len(records), 6)
        self.assertEqual(records[0]["splits"]["train"]["pair_count"], 1995)
        self.assertIn("train.jsonl", opened)
        self.assertIn("validation.jsonl", opened)
        self.assertNotIn("test.jsonl", opened)

    def test_dependencies_accept_cuda_local_version_and_reject_version_drift(self):
        import tomllib
        pins = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
        versions = dict(requirement.split("==") for requirement in pins)
        versions["torch"] += "+cu130"
        with patch.object(preflight, "version", side_effect=versions.__getitem__):
            self.assertEqual(preflight.verify_dependencies()["packages"]["torch"], "2.13.0+cu130")
            versions["transformers"] = "0.0.0"
            with self.assertRaises(RuntimeError):
                preflight.verify_dependencies()

    def test_runtime_requires_native_bf16_and_probes_gradients_without_training(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = False
        torch.version.cuda = "13.0"
        torch.__version__ = "2.13.0+cu130"
        with patch.object(preflight.importlib, "import_module", return_value=torch):
            with self.assertRaises(RuntimeError):
                preflight.verify_bf16_runtime()
            torch.ones.assert_not_called()
            torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)
            torch.cuda.is_bf16_supported.return_value = True
            output = torch.ones.return_value.__matmul__.return_value
            output.dtype = torch.bfloat16
            torch.isfinite.return_value.all.return_value.item.return_value = True
            result = preflight.verify_bf16_runtime()
            self.assertTrue(result["forward_backward_verified"])
            output.float.return_value.mean.return_value.backward.assert_called_once()
            torch.isfinite.return_value.all.return_value.item.return_value = False
            with self.assertRaises(RuntimeError):
                preflight.verify_bf16_runtime()

    def test_storage_uses_the_results_filesystem_and_counts_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = [{"method": "true_pair", "split_seed": 13, "model_seed": seed, "pair_loss_weight": 0.25, "evaluation_scope": "full", "phases": ["confirmation"]} for seed in (13, 40)]
            existing = root / "13" / "checkpoint.pt"
            existing.parent.mkdir()
            existing.write_bytes(b"checkpoint")
            with patch.object(preflight, "RESULTS_ROOT", root / "results"), patch.object(preflight, "DEFAULT_SELECTION_PATH", root / "missing.json"), patch.object(preflight, "build_collection_plan", return_value=plan), patch.object(preflight, "get_run_dir", side_effect=lambda **task: root / str(task["model_seed"])), patch.object(preflight.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024 ** 3)) as disk:
                result = preflight.verify_storage(True)
                self.assertEqual(result["new_checkpoint_count"], 1)
                self.assertEqual(result["required_free_bytes_estimate"], preflight.CHECKPOINT_BUDGET_BYTES + preflight.RESERVE_BYTES)
                disk.assert_called_with(root / "results")
                disk.return_value.free = 0
                with self.assertRaises(RuntimeError):
                    preflight.verify_storage(True)

    def test_known_selection_uses_the_selected_weight_for_storage_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text("{}")
            with patch.object(preflight, "RESULTS_ROOT", root / "results"), patch.object(preflight, "DEFAULT_SELECTION_PATH", selection), patch.object(preflight, "load_screening_selection", return_value={"selected_pair_loss_weight": 1.0}), patch.object(preflight, "build_collection_plan", return_value=[]) as plan, patch.object(preflight.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024 ** 3)):
                result = preflight.verify_storage(True)
                plan.assert_called_once_with(1.0, True)
                self.assertTrue(result["selection_known"])

    def test_report_records_failure_and_stops_before_gpu_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(preflight, "verify_v2_protocol", side_effect=RuntimeError("changed learning rate")), patch.object(preflight, "verify_bf16_runtime") as gpu:
                with self.assertRaises(RuntimeError):
                    preflight.run_preflight(output_root=root)
                gpu.assert_not_called()
            report = json.loads((root / "preflight_report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error_message"], "changed learning rate")

    def test_cli_preview_does_not_import_torch_or_require_result_files(self):
        result = subprocess.run([sys.executable, "-m", "src.experiments.preflight", "--dry-run"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Test predictions and test split contents are excluded", result.stdout)


if __name__ == "__main__":
    unittest.main()
