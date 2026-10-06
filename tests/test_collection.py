import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.experiments.fingerprint as fingerprints
import src.results.collection as collection
import src.results.io as results_io
from src.evaluation.artifacts import build_evaluation_metrics, evaluate_run_splits
from src.paths import PROJECT_ROOT


class CollectionTest(unittest.TestCase):
    @contextmanager
    def environment(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source_dir = root / "data" / "seed_13"
            source_dir.mkdir(parents=True)
            data = {}
            for split, first in (("train", 5), ("validation", 1), ("test", 3)):
                data[split] = pd.DataFrame({
                    "id": [f"{first:03}.H", f"{first:03}.N", f"{first + 1:03}.H", f"{first + 1:03}.N"],
                    "text": ["NA", "null", "h2", "n2"], "label": [1, 0, 1, 0],
                })
                data[split].to_json(source_dir / f"{split}.jsonl", orient="records", lines=True, force_ascii=False)
            (source_dir / "metadata.json").write_text("{}\n")
            selection = root / "selection.json"
            selection.write_text('{"selected_pair_loss_weight": 0.25}\n')
            decision = {
                "selected_pair_loss_weight": 0.25, "selection_path": str(selection),
                "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
            }
            plan = [
                {"method": method, "split_seed": 13, "model_seed": 13, "pair_loss_weight": weight, "evaluation_scope": scope, "phases": [phase]}
                for method, weight, scope, phase in (
                    ("true_pair", 0.25, "validation", "screening"),
                    ("true_pair", 0.25, "full", "confirmation"),
                    ("instance_level", None, "full", "references"),
                )
            ]
            def run_dir(**task):
                return root / "runs" / task["method"] / task["evaluation_scope"]

            stack.enter_context(patch.object(fingerprints, "get_split_dir", return_value=source_dir))
            stack.enter_context(patch.object(collection, "get_split_dir", return_value=source_dir))
            stack.enter_context(patch.object(collection, "get_run_dir", side_effect=run_dir))
            stack.enter_context(patch.object(collection, "load_screening_selection", return_value=decision))
            stack.enter_context(patch.object(collection, "build_collection_plan", return_value=plan))
            stack.enter_context(patch.multiple(results_io, EXPECTED_SPLIT_COUNTS={"validation": 4, "test": 4}, EXPECTED_PAIR_COUNTS={"validation": 2, "test": 2}))
            paths = []
            for entry in plan:
                task = {key: entry[key] for key in collection.TASK_KEYS}
                expected = fingerprints.build_experiment_fingerprint(**task)
                artifacts = results_io.prepare_run_directory(run_dir(**task))
                results = evaluate_run_splits(data, lambda split, frame: (np.array([[0, 3], [0, 1], [0, 4], [0, 2]]), frame["label"].to_numpy()), task["evaluation_scope"])
                results_io.write_json({
                    **task, "status": "completed", "experiment_id": expected["experiment_id"],
                    "fingerprint": {"version": expected["fingerprint_version"], **{key: expected[key] for key in ("config", "config_hash", "dataset_hash", "source_hash")}},
                }, artifacts["metadata"])
                results_io.write_json(build_evaluation_metrics(results), artifacts["metrics"])
                results_io.write_history([{"epoch": 1}], artifacts["history"])
                for split, result in results.items():
                    prefix = "validation_" if split == "validation" else ""
                    results_io.write_dataframe(result["predictions"], artifacts[f"{prefix}predictions"])
                    results_io.write_dataframe(result["pair_predictions"], artifacts[f"{prefix}pair_predictions"])
                artifacts["checkpoint"].write_bytes(b"checkpoint")
                results_io.mark_run_completed(run_dir(**task))
                paths.append(artifacts)
            yield root, selection, decision, paths

    def test_plan_deduplicates_confirmation_and_references_at_lambda_one(self):
        for weight, include, count in ((0.25, False, 180), (0.25, True, 288), (1.0, True, 216)):
            plan = collection.build_collection_plan(weight, include)
            self.assertEqual(len(plan), count)
            self.assertEqual(len({tuple(entry[key] for key in collection.TASK_KEYS) for entry in plan}), count)
            shared = [entry for entry in plan if entry["phases"] == ["confirmation", "references"]]
            self.assertEqual(len(shared), 72 if weight == 1.0 and include else 0)

    def test_collection_exports_raw_rows_and_verified_artifact_index(self):
        with self.environment() as (root, selection, decision, paths):
            result = collection.collect_v3_results(selection, output_root=root / "collection")
            runs = pd.read_csv(result["outputs"]["runs"]["path"])
            self.assertEqual(result["row_count"], 3)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["source_predictions_verified"])
            self.assertEqual(runs["validation_instance_f1_macro"].tolist(), [1 / 3] * 3)
            self.assertEqual(runs["validation_instance_calibrated_f1_macro"].tolist(), [1.0] * 3)
            self.assertTrue(pd.isna(runs.iloc[0]["test_instance_f1_macro"]))
            self.assertEqual(runs["config_training_precision"].unique().tolist(), ["bf16"])
            for entry in result["outputs"].values():
                self.assertEqual(collection._hash_file(entry["path"]), entry["sha256"])
            self.assertEqual(len(pd.read_csv(result["outputs"]["artifacts"]["path"])), 22)
            self.assertEqual({path.name for path in (root / "collection").iterdir()}, {"run.lock", "runs.csv", "artifacts.csv", "collection_manifest.json"})

    def test_source_text_mismatch_blocks_collection_even_when_internal_exports_are_valid(self):
        with self.environment() as (root, selection, decision, paths):
            frame = pd.read_csv(paths[0]["validation_predictions"], dtype={"id": str, "pair_id": str, "text": str}, keep_default_na=False)
            frame.loc[0, "text"] = "changed text"
            results_io.write_dataframe(frame, paths[0]["validation_predictions"])
            self.assertTrue(results_io.is_run_completed(paths[0]["metadata"].parent))
            with self.assertRaises(ValueError):
                collection.collect_v3_results(selection, output_root=root / "collection")
            saved = json.loads((root / "collection" / "collection_manifest.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertFalse((root / "collection" / "runs.csv").exists())

    def test_stale_metadata_is_rejected(self):
        with self.environment() as (root, selection, decision, paths):
            metadata = json.loads(paths[0]["metadata"].read_text())
            metadata["fingerprint"]["config"]["num_epochs"] = 7
            results_io.write_json(metadata, paths[0]["metadata"])
            with self.assertRaises(RuntimeError):
                collection.collect_v3_results(selection, output_root=root / "collection")

    def test_changed_selection_cannot_complete_collection(self):
        with self.environment() as (root, selection, decision, paths):
            selection.write_text("changed selection")
            with self.assertRaises(RuntimeError):
                collection.collect_v3_results(selection, output_root=root / "collection")

    def test_artifact_change_during_collection_is_rejected_before_csv_export(self):
        with self.environment() as (root, selection, decision, paths):
            original = collection._verify_source_predictions
            def change_artifact(*arguments):
                original(*arguments)
                if arguments[2] == "validation":
                    metrics = json.loads(paths[0]["metrics"].read_text())
                    metrics["validation_instance"]["f1_macro"] = 0.5
                    results_io.write_json(metrics, paths[0]["metrics"])
            with patch.object(collection, "_verify_source_predictions", side_effect=change_artifact):
                with self.assertRaises(RuntimeError):
                    collection.collect_v3_results(selection, output_root=root / "collection")
            self.assertFalse((root / "collection" / "runs.csv").exists())

    def test_validation_source_binding_never_reads_test_source_or_predictions(self):
        with self.environment() as (root, selection, decision, paths):
            original = collection.read_jsonl
            accessed = []
            def read(path):
                accessed.append(Path(path).name)
                self.assertNotEqual(Path(path).name, "test.jsonl")
                return original(path)
            with patch.object(collection, "read_jsonl", side_effect=read):
                collection._verify_source_predictions(paths[0], 13, "validation", {})
            self.assertEqual(accessed, ["validation.jsonl"])

    def test_cli_preview_does_not_require_results_or_training_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.results.collection", "--dry-run", "--include-references", "--preview-weight", "1.0"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Unique result rows planned: 216", result.stdout)


if __name__ == "__main__":
    unittest.main()
