import builtins
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.experiments.fingerprint as fingerprint_module
import src.experiments.run_screening as screening_module
import src.results.selection as selection_module
from src.evaluation.artifacts import build_evaluation_metrics, evaluate_run_splits
from src.results.io import mark_run_completed, prepare_run_directory, write_dataframe, write_history, write_json


class SelectionTest(unittest.TestCase):
    def records(self):
        return pd.DataFrame([
            {**task, **dict.fromkeys(selection_module.METRIC_FIELDS, 0.7)}
            for task in screening_module.build_screening_grid()
        ])

    def select(self, records):
        _, summary = selection_module.build_screening_summaries(records)
        return selection_module.select_pair_loss_weight(summary)

    def test_ranking_is_maximized_only_within_the_f1_constraint(self):
        records = self.records()
        records.loc[:, "f1_macro"] = 0.79
        records.loc[records["pair_loss_weight"] == 1.0, ["f1_macro", "pair_ranking_accuracy"]] = [0.8, 0.8]
        records.loc[records["pair_loss_weight"] == 0.1, ["f1_macro", "pair_ranking_accuracy"]] = [0.796, 0.9]
        records.loc[records["pair_loss_weight"] == 0.25, ["f1_macro", "pair_ranking_accuracy"]] = [0.793, 0.99]
        selected, summary = self.select(records)
        self.assertEqual(selected, 0.1)
        self.assertFalse(summary.set_index("pair_loss_weight").loc[0.25, "eligible"])

    def test_tolerance_boundary_is_inclusive(self):
        records = self.records()
        records.loc[:, "f1_macro"] = 0.79
        records.loc[records["pair_loss_weight"] == 1.0, "f1_macro"] = 0.8
        records.loc[records["pair_loss_weight"] == 0.25, ["f1_macro", "pair_ranking_accuracy"]] = [0.795, 0.9]
        self.assertEqual(self.select(records)[0], 0.25)

    def test_ties_prefer_higher_f1_then_lower_weight(self):
        records = self.records()
        records.loc[:, "f1_macro"] = 0.8
        records.loc[records["pair_loss_weight"] == 0.0, "f1_macro"] = 0.796
        self.assertEqual(self.select(records)[0], 0.1)

    def test_calibrated_f1_does_not_drive_selection(self):
        records = self.records()
        first = self.select(records)[0]
        records.loc[records["pair_loss_weight"] == 2.0, "f1_macro_calibrated"] = 1.0
        self.assertEqual(self.select(records)[0], first)

    def test_aggregation_separates_split_and_model_variability(self):
        records = self.records()
        split_order = {value: index for index, value in enumerate(selection_module.SPLIT_SEEDS)}
        model_order = {value: index for index, value in enumerate(selection_module.SCREENING_MODEL_SEEDS)}
        records["f1_macro"] = 0.7 + records["split_seed"].map(split_order) * 0.01 + records["model_seed"].map(model_order) * 0.001
        by_split, summary = selection_module.build_screening_summaries(records)
        row = summary.iloc[0]
        self.assertEqual(len(by_split), 36)
        self.assertEqual(row["run_count"], 18)
        self.assertEqual(row["split_count"], 6)
        self.assertAlmostEqual(row["f1_macro_mean"], 0.726)
        self.assertAlmostEqual(row["f1_macro_std_between_splits"], np.std(np.arange(6) * 0.01, ddof=1))
        self.assertAlmostEqual(row["f1_macro_mean_std_model_seeds"], 0.001)

    def test_missing_duplicate_and_unexpected_runs_are_rejected(self):
        records = self.records()
        invalid = [records.iloc[:-1], pd.concat([records.iloc[:-1], records.iloc[:1]])]
        changed = records.copy()
        changed.loc[0, "pair_loss_weight"] = 3.0
        invalid.append(changed)
        for frame in invalid:
            with self.assertRaises(ValueError):
                selection_module.build_screening_summaries(frame)

    def test_wrong_scope_invalid_metrics_and_wrong_method_are_rejected(self):
        for column, value in (
            ("evaluation_scope", "full"), ("method", "shuffled_pair"),
            ("f1_macro", float("nan")), ("pair_ranking_accuracy", 1.1), ("log_loss", -1),
        ):
            records = self.records()
            records.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                selection_module.build_screening_summaries(records)

    def test_row_order_does_not_change_selection(self):
        records = self.records()
        self.assertEqual(self.select(records)[0], self.select(records.sample(frac=1, random_state=13))[0])

    def test_selection_is_skipped_for_dry_runs_and_training_failures(self):
        for dry_run, failures in ((True, 0), (False, 1)):
            with patch.object(screening_module, "run_grid", return_value={"failed_runs": failures}), patch.object(
                screening_module, "generate_screening_selection",
            ) as select:
                screening_module.run_screening(dry_run=dry_run)
            select.assert_not_called()

    def test_select_only_never_starts_training(self):
        with patch.object(screening_module, "parse_args", return_value=type("Args", (), {"select_only": True})()), patch.object(
            screening_module, "generate_screening_selection",
            return_value={"selected_pair_loss_weight": 0.25, "selection_path": "selection.json"},
        ) as select, patch.object(screening_module, "run_grid") as train, redirect_stdout(io.StringIO()):
            screening_module.main()
        select.assert_called_once_with()
        train.assert_not_called()

    def test_complete_artifacts_are_audited_without_reading_test_data(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data = root / "data"
            for seed in selection_module.SPLIT_SEEDS:
                split = data / str(seed)
                split.mkdir(parents=True)
                for filename in ("train.jsonl", "validation.jsonl", "metadata.json"):
                    (split / filename).write_text("{}\n")
            stack.enter_context(patch.object(fingerprint_module, "get_split_dir", side_effect=lambda seed: data / str(seed)))
            stack.enter_context(patch.object(
                selection_module, "get_run_dir",
                side_effect=lambda **task: root / "runs" / str(task["pair_loss_weight"]) / str(task["split_seed"]) / str(task["model_seed"]),
            ))
            stack.enter_context(patch.multiple(
                "src.results.io", EXPECTED_SPLIT_COUNTS={"validation": 4}, EXPECTED_PAIR_COUNTS={"validation": 2},
            ))
            dataframe = pd.DataFrame({
                "id": ["1.H", "1.N", "2.H", "2.N"], "text": ["h1", "n1", "h2", "n2"], "label": [1, 0, 1, 0],
            })
            logits = np.array([[0, 3], [0, 1], [0, 4], [0, 2]])
            results = evaluate_run_splits(
                {"validation": dataframe}, lambda name, frame: (logits, frame["label"].to_numpy()), "validation",
            )
            first_paths = None
            for task in screening_module.build_screening_grid():
                paths = prepare_run_directory(selection_module.get_run_dir(**task))
                fingerprint = selection_module.build_experiment_fingerprint(**task)
                write_json({
                    **task, "status": "completed", "experiment_id": fingerprint["experiment_id"],
                    "fingerprint": {
                        "version": fingerprint["fingerprint_version"],
                        **{key: fingerprint[key] for key in ("config", "config_hash", "dataset_hash", "source_hash")},
                    },
                }, paths["metadata"])
                write_json(build_evaluation_metrics(results), paths["metrics"])
                write_history([{"epoch": 1}], paths["history"])
                write_dataframe(results["validation"]["predictions"], paths["validation_predictions"])
                write_dataframe(results["validation"]["pair_predictions"], paths["validation_pair_predictions"])
                paths["checkpoint"].write_bytes(b"checkpoint")
                mark_run_completed(selection_module.get_run_dir(**task))
                if first_paths is None:
                    first_paths = paths
            original_open = builtins.open
            original_path_open = Path.open
            forbidden = {"test.jsonl", "predictions.csv", "pair_predictions.csv"}

            def guarded_open(path, *args, **kwargs):
                self.assertNotIn(Path(path).name, forbidden)
                return original_open(path, *args, **kwargs)

            def guarded_path_open(path, *args, **kwargs):
                self.assertNotIn(path.name, forbidden)
                return original_path_open(path, *args, **kwargs)

            stack.enter_context(patch("builtins.open", side_effect=guarded_open))
            stack.enter_context(patch.object(Path, "open", guarded_path_open))
            report_dir = root / "reports"
            result = selection_module.generate_screening_selection(report_dir)
            self.assertEqual(result["selected_pair_loss_weight"], 0.0)
            selection = json.loads(Path(result["selection_path"]).read_text())
            self.assertEqual(selection["completed_runs"], 108)
            self.assertEqual(len(selection["inputs"]), 108)
            self.assertEqual(selection["rule"]["selection_split"], "validation")
            for output in selection["outputs"].values():
                self.assertEqual(output["sha256"], selection_module._hash_file(output["path"]))
            for artifact, mutation, exception in (
                ("metadata", lambda obj: obj.update(evaluation_scope="full"), RuntimeError),
                ("metadata", lambda obj: obj.update(experiment_id="stale"), RuntimeError),
                ("metadata", lambda obj: obj["fingerprint"].update(config_hash="stale"), RuntimeError),
                ("metrics", lambda obj: obj.update(test_instance={"f1_macro": 1.0}), RuntimeError),
                ("metrics", lambda obj: obj["validation_instance"].update(f1_macro=0.5), RuntimeError),
            ):
                path = first_paths[artifact]
                original = path.read_bytes()
                obj = json.loads(original)
                mutation(obj)
                write_json(obj, path)
                try:
                    with self.assertRaises(exception):
                        selection_module.generate_screening_selection(root / "invalid")
                    self.assertFalse((root / "invalid" / "selection.json").exists())
                finally:
                    path.write_bytes(original)
            path = first_paths["validation_pair_predictions"]
            original = path.read_bytes()
            pairs = pd.read_csv(path)
            pairs.loc[0, "pair_margin"] += 1.0
            write_dataframe(pairs, path)
            with self.assertRaises(RuntimeError):
                selection_module.generate_screening_selection(root / "invalid")
            path.write_bytes(original)
            first_paths["checkpoint"].unlink()
            with self.assertRaises(RuntimeError):
                selection_module.generate_screening_selection(root / "invalid")


if __name__ == "__main__":
    unittest.main()
