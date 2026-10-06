import io
import json
import subprocess
import sys
import tarfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.results.collection as collection
import src.results.export as export
from src.paths import PROJECT_ROOT
from src.results.io import write_json
import tests.test_collection as fixtures


class ExportTest(unittest.TestCase):
    @contextmanager
    def environment(self):
        with fixtures.CollectionTest().environment() as (root, selection, decision, paths), ExitStack() as stack:
            outputs = {}
            for name, filename in (("runs", "screening_runs.csv"), ("by_split", "screening_by_split.csv"), ("by_lambda", "screening_by_lambda.csv")):
                path = root / filename
                path.write_text("fixture\n", encoding="utf-8")
                outputs[name] = {"path": str(path), "sha256": export._hash(path)}
            write_json({"selected_pair_loss_weight": 0.25, "outputs": outputs}, selection)
            decision["selection_sha256"] = export._hash(selection)
            collected = collection.collect_v3_results(selection, output_root=root / "collection")
            stack.enter_context(patch.object(export, "load_screening_selection", return_value=decision))
            stack.enter_context(patch.object(export, "build_collection_plan", side_effect=collection.build_collection_plan))
            stack.enter_context(patch.object(export, "get_run_dir", side_effect=collection.get_run_dir))
            stack.enter_context(patch.object(export, "get_split_dir", side_effect=collection.get_split_dir))
            yield root, collected, decision, paths

    def test_archive_contains_verified_results_and_portable_file_inventory(self):
        with self.environment() as (root, collected, decision, paths):
            result = export.export_v3_results(collected["manifest_path"], root / "export")
            manifest = export.verify_result_archive(result["archive_path"], result["archive_sha256"])
            self.assertEqual((result["status"], result["row_count"]), ("completed", 3))
            self.assertEqual(manifest["selection"], decision)
            self.assertFalse(manifest["checkpoints_included"])
            names = {row["path"] for row in manifest["files"]}
            self.assertIn("collection/runs.csv", names)
            self.assertIn("screening/selection.json", names)
            self.assertIn("data/pair_controlled/seed_13/train.jsonl", names)
            self.assertIn("src/config.py", names)
            self.assertIn("tests/test_export.py", names)
            self.assertIn("uv.lock", names)
            self.assertIn("results_v2/bf16/instance_level/split_13/model_seed_13/metadata.json", names)
            self.assertEqual(len([name for name in names if name.startswith("runs/")]), 22)
            self.assertTrue(all(not Path(name).is_absolute() for name in names))
            self.assertFalse(any("checkpoint" in name or name.endswith("run.lock") for name in names))
            checksum = Path(result["checksum_path"]).read_text()
            self.assertEqual(checksum, f"{result['archive_sha256']}  v3-results.tar.gz\n")

    def test_changed_input_is_rejected_and_previous_archive_is_preserved(self):
        with self.environment() as (root, collected, decision, paths):
            result = export.export_v3_results(collected["manifest_path"], root / "export")
            paths[1]["metrics"].write_text("{}\n")
            with self.assertRaises(RuntimeError):
                export.export_v3_results(collected["manifest_path"], root / "export")
            self.assertEqual(export._hash(result["archive_path"]), result["archive_sha256"])
            self.assertEqual(json.loads((root / "export" / "export_status.json").read_text())["status"], "failed")
            self.assertFalse((root / "export" / "v3-results.tar.gz.tmp").exists())

    def test_current_dataset_and_incomplete_collection_are_rejected(self):
        for change in ("dataset", "collection"):
            with self.environment() as (root, collected, decision, paths):
                if change == "dataset":
                    (root / "data" / "seed_13" / "train.jsonl").write_text("{}\n")
                else:
                    path = Path(collected["manifest_path"])
                    value = json.loads(path.read_text())
                    value["status"] = "failed"
                    write_json(value, path)
                with self.assertRaises(RuntimeError):
                    export.export_v3_results(collected["manifest_path"], root / "export")
                self.assertFalse((root / "export" / "v3-results.tar.gz").exists())

    def test_duplicate_artifact_inventory_is_rejected_even_with_updated_table_hash(self):
        with self.environment() as (root, collected, decision, paths):
            path = root / "collection" / "artifacts.csv"
            table = pd.read_csv(path)
            pd.concat((table, table.iloc[:1])).to_csv(path, index=False)
            manifest = json.loads(Path(collected["manifest_path"]).read_text())
            manifest["outputs"]["artifacts"].update(sha256=export._hash(path), row_count=len(table) + 1)
            write_json(manifest, collected["manifest_path"])
            with self.assertRaises(RuntimeError):
                export.export_v3_results(collected["manifest_path"], root / "export")

    def test_input_change_after_snapshot_and_interrupt_leave_no_partial_archive(self):
        for interrupt in (False, True):
            with self.environment() as (root, collected, decision, paths):
                original = export._snapshot_files

                def changed(path):
                    report, files = original(path)
                    if interrupt:
                        raise KeyboardInterrupt()
                    paths[0]["history"].write_text("changed\n")
                    return report, files

                with patch.object(export, "_snapshot_files", side_effect=changed), self.assertRaises(KeyboardInterrupt if interrupt else RuntimeError):
                    export.export_v3_results(collected["manifest_path"], root / "export")
                self.assertFalse((root / "export" / "v3-results.tar.gz").exists())
                self.assertFalse((root / "export" / "v3-results.tar.gz.tmp").exists())

    def test_verifier_rejects_corrupted_members_unsafe_names_and_wrong_checksum(self):
        with self.environment() as (root, collected, decision, paths):
            result = export.export_v3_results(collected["manifest_path"], root / "export")
            with self.assertRaises(ValueError):
                export.verify_result_archive(result["archive_path"], "0" * 64)
            for unsafe in (False, True):
                target = root / "invalid.tar.gz"
                with tarfile.open(result["archive_path"], "r:gz") as source, tarfile.open(target, "w:gz") as output:
                    for member in source.getmembers():
                        content = source.extractfile(member).read()
                        if member.name == "collection/runs.csv":
                            content = b"changed"
                            if unsafe:
                                member.name = "../runs.csv"
                        member.size = len(content)
                        output.addfile(member, io.BytesIO(content))
                with self.assertRaises(ValueError):
                    export.verify_result_archive(target)

    def test_cli_preview_does_not_require_training_or_result_files(self):
        result = subprocess.run([sys.executable, "-m", "src.results.export", "--dry-run"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Checkpoints included: False", result.stdout)


if __name__ == "__main__":
    unittest.main()
