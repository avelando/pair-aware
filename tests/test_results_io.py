import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.predictions import build_complete_pair_predictions, build_instance_predictions
from src.results.io import (
    RUN_ARTIFACT_NAMES,
    get_run_artifact_paths,
    is_run_completed,
    mark_run_completed,
    prepare_run_directory,
    remove_checkpoint,
    write_dataframe,
    write_history,
    write_json,
)


class ResultsIOTest(unittest.TestCase):
    def setUp(self):
        self.config_patcher = patch.multiple(
            "src.results.io",
            EXPECTED_SPLIT_COUNTS={
                "train": 8,
                "validation": 4,
                "test": 4,
            },
            EXPECTED_PAIR_COUNTS={
                "train": 4,
                "validation": 2,
                "test": 2,
            },
        )
        self.config_patcher.start()
        self.addCleanup(
            self.config_patcher.stop
        )

    def create_completed_run(
        self,
        run_dir,
        experiment_id="experiment-1",
        evaluation_scope="full",
    ):
        paths = prepare_run_directory(
            run_dir
        )

        write_json(
            {
                "status": "completed",
                "experiment_id": experiment_id,
                "evaluation_scope": evaluation_scope,
            },
            paths["metadata"],
        )

        write_json(
            {
                "validation_instance": {"f1_macro": 1.0},
                "validation_pair": {"pair_ranking_accuracy": 1.0},
                "test_instance": {
                    "accuracy": 1.0,
                },
                "test_pair": {
                    "pair_ranking_accuracy": 1.0,
                },
            },
            paths["metrics"],
        )

        write_history(
            [
                {
                    "epoch": 1,
                    "validation_f1_macro": 1.0,
                }
            ],
            paths["history"],
        )

        test_dataframe = pd.DataFrame({
            "id": ["1.H", "1.N", "2.H", "2.N"],
            "text": ["h1", "n1", "h2", "n2"], "label": [1, 0, 1, 0],
        })
        test_predictions = build_instance_predictions(
            test_dataframe, np.array([[0, 1], [1, 0], [0, 1], [1, 0]]),
        )
        write_dataframe(test_predictions, paths["predictions"])
        write_dataframe(build_complete_pair_predictions(test_predictions), paths["pair_predictions"])

        validation_dataframe = pd.DataFrame({
            "id": ["3.H", "3.N", "4.H", "4.N"],
            "text": ["h3", "n3", "h4", "n4"],
            "label": [1, 0, 1, 0],
        })
        validation_predictions = build_instance_predictions(
            validation_dataframe,
            np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        )
        write_dataframe(validation_predictions, paths["validation_predictions"])
        write_dataframe(build_complete_pair_predictions(validation_predictions), paths["validation_pair_predictions"])

        if evaluation_scope == "validation":
            metrics = json.loads(paths["metrics"].read_text())
            del metrics["test_instance"]
            del metrics["test_pair"]
            write_json(metrics, paths["metrics"])
            paths["predictions"].unlink()
            paths["pair_predictions"].unlink()
            paths["checkpoint"].write_bytes(b"checkpoint")

        mark_run_completed(
            run_dir
        )

        return paths

    def test_run_artifact_paths_use_expected_names(self):
        run_dir = Path("results/run")

        paths = get_run_artifact_paths(
            run_dir
        )

        self.assertEqual(
            set(paths),
            set(RUN_ARTIFACT_NAMES),
        )

        for name, filename in RUN_ARTIFACT_NAMES.items():
            self.assertEqual(
                paths[name],
                run_dir / filename,
            )

    def test_prepare_run_directory_removes_existing_artifacts_and_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = prepare_run_directory(
                run_dir
            )

            for path in paths.values():
                path.write_text(
                    "artifact",
                    encoding="utf-8",
                )

                temporary_path = path.with_suffix(
                    path.suffix + ".tmp"
                )

                temporary_path.write_text(
                    "temporary",
                    encoding="utf-8",
                )

            returned_paths = prepare_run_directory(
                run_dir
            )

            self.assertEqual(
                returned_paths,
                paths,
            )

            for path in paths.values():
                self.assertFalse(
                    path.exists()
                )

                self.assertFalse(
                    path.with_suffix(
                        path.suffix + ".tmp"
                    ).exists()
                )

    def test_write_json_serializes_supported_non_standard_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"

            write_json(
                {
                    "path": Path("example/file.txt"),
                    "value": np.int64(7),
                },
                path,
            )

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                data,
                {
                    "path": "example/file.txt",
                    "value": 7,
                },
            )

            self.assertFalse(
                path.with_suffix(
                    ".json.tmp"
                ).exists()
            )

    def test_write_dataframe_writes_without_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"

            dataframe = pd.DataFrame(
                {
                    "id": ["1.H", "1.N"],
                    "value": [1, 2],
                }
            )

            write_dataframe(
                dataframe,
                path,
            )

            loaded = pd.read_csv(
                path
            )

            self.assertEqual(
                loaded.columns.tolist(),
                ["id", "value"],
            )

            self.assertEqual(
                loaded["id"].tolist(),
                ["1.H", "1.N"],
            )

            self.assertFalse(
                path.with_suffix(
                    ".csv.tmp"
                ).exists()
            )

    def test_empty_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"

            with self.assertRaisesRegex(
                ValueError,
                "Training history cannot be empty",
            ):
                write_history(
                    [],
                    path,
                )

    def test_completed_run_with_valid_artifacts_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            self.create_completed_run(
                run_dir
            )

            self.assertTrue(
                is_run_completed(
                    run_dir
                )
            )

    def test_completed_run_with_matching_experiment_id_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            self.create_completed_run(
                run_dir,
                experiment_id="expected-id",
            )

            self.assertTrue(
                is_run_completed(
                    run_dir,
                    expected_experiment_id="expected-id",
                )
            )

    def test_completed_run_with_mismatched_experiment_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            self.create_completed_run(
                run_dir,
                experiment_id="old-id",
            )

            self.assertFalse(
                is_run_completed(
                    run_dir,
                    expected_experiment_id="new-id",
                )
            )

    def test_completed_run_without_experiment_id_is_rejected_when_expected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = self.create_completed_run(
                run_dir
            )

            write_json(
                {
                    "status": "completed",
                },
                paths["metadata"],
            )

            self.assertFalse(
                is_run_completed(
                    run_dir,
                    expected_experiment_id="expected-id",
                )
            )

    def test_completed_run_requires_every_final_artifact(self):
        for artifact_name in (
            "metadata",
            "metrics",
            "history",
            "predictions",
            "pair_predictions",
            "validation_predictions",
            "validation_pair_predictions",
            "completed",
        ):
            with self.subTest(
                artifact_name=artifact_name
            ):
                with tempfile.TemporaryDirectory() as directory:
                    run_dir = Path(directory)
                    paths = self.create_completed_run(
                        run_dir
                    )

                    paths[
                        artifact_name
                    ].unlink()

                    self.assertFalse(
                        is_run_completed(
                            run_dir
                        )
                    )

    def test_invalid_completed_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = self.create_completed_run(
                run_dir
            )

            paths["completed"].write_text(
                "running\n",
                encoding="utf-8",
            )

            self.assertFalse(
                is_run_completed(
                    run_dir
                )
            )

    def test_non_completed_metadata_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = self.create_completed_run(
                run_dir
            )

            write_json(
                {
                    "status": "failed",
                },
                paths["metadata"],
            )

            self.assertFalse(
                is_run_completed(
                    run_dir
                )
            )

    def test_missing_metric_sections_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = self.create_completed_run(
                run_dir
            )

            write_json(
                {
                    "test_instance": {},
                },
                paths["metrics"],
            )

            self.assertFalse(
                is_run_completed(
                    run_dir
                )
            )

    def test_malformed_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = self.create_completed_run(
                run_dir
            )

            paths["metrics"].write_text(
                "{invalid",
                encoding="utf-8",
            )

            self.assertFalse(
                is_run_completed(
                    run_dir
                )
            )

    def test_empty_history_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = self.create_completed_run(
                run_dir
            )

            paths["history"].write_text(
                "epoch\n",
                encoding="utf-8",
            )

            self.assertFalse(
                is_run_completed(
                    run_dir
                )
            )

    def test_wrong_prediction_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = self.create_completed_run(
                run_dir
            )

            write_dataframe(
                pd.DataFrame(
                    {
                        "id": [
                            "1.H",
                            "1.N",
                            "2.H",
                        ]
                    }
                ),
                paths["predictions"],
            )

            self.assertFalse(
                is_run_completed(
                    run_dir
                )
            )

    def test_wrong_pair_prediction_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = self.create_completed_run(
                run_dir
            )

            write_dataframe(
                pd.DataFrame(
                    {
                        "pair_id": ["1"]
                    }
                ),
                paths[
                    "pair_predictions"
                ],
            )

            self.assertFalse(
                is_run_completed(
                    run_dir
                )
            )

    def test_validation_artifact_schema_is_required(self):
        for name, column in (("validation_predictions", "probability_pun"), ("validation_pair_predictions", "pun_logit_pun")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                paths = self.create_completed_run(directory)
                frame = pd.read_csv(paths[name]).drop(columns=[column])
                write_dataframe(frame, paths[name])
                self.assertFalse(is_run_completed(directory))

    def test_validation_artifact_counts_are_required(self):
        for name in ("validation_predictions", "validation_pair_predictions"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                paths = self.create_completed_run(directory)
                frame = pd.read_csv(paths[name]).iloc[:-1]
                write_dataframe(frame, paths[name])
                self.assertFalse(is_run_completed(directory))

    def test_validation_identifiers_must_be_unique(self):
        for name, column in (("validation_predictions", "id"), ("validation_pair_predictions", "pair_id")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                paths = self.create_completed_run(directory)
                frame = pd.read_csv(paths[name])
                frame.loc[1, column] = frame.loc[0, column]
                write_dataframe(frame, paths[name])
                self.assertFalse(is_run_completed(directory))

    def test_validation_pair_members_must_match_their_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_completed_run(directory)
            frame = pd.read_csv(paths["validation_pair_predictions"])
            frame["pun_id"] = frame["pun_id"].iloc[::-1].to_numpy()
            write_dataframe(frame, paths["validation_pair_predictions"])
            self.assertFalse(is_run_completed(directory))

    def test_validation_scores_must_be_finite(self):
        for name, column in (("validation_predictions", "logit_pun"), ("validation_pair_predictions", "pair_margin")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                paths = self.create_completed_run(directory)
                frame = pd.read_csv(paths[name])
                frame.loc[0, column] = float("inf")
                write_dataframe(frame, paths[name])
                self.assertFalse(is_run_completed(directory))

    def test_validation_metric_sections_are_required(self):
        for section in ("validation_instance", "validation_pair"):
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                paths = self.create_completed_run(directory)
                metrics = json.loads(paths["metrics"].read_text())
                del metrics[section]
                write_json(metrics, paths["metrics"])
                self.assertFalse(is_run_completed(directory))

    def test_validation_only_run_is_completed_without_test_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            self.create_completed_run(directory, evaluation_scope="validation")
            self.assertTrue(is_run_completed(directory))

    def test_validation_only_run_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.create_completed_run(directory, evaluation_scope="validation")
            paths["checkpoint"].unlink()
            self.assertFalse(is_run_completed(directory))

    def test_validation_only_run_rejects_test_outputs(self):
        for artifact in ("metrics", "predictions", "pair_predictions"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                paths = self.create_completed_run(directory, evaluation_scope="validation")
                if artifact == "metrics":
                    metrics = json.loads(paths["metrics"].read_text())
                    metrics["test_instance"] = {"f1_macro": 1.0}
                    write_json(metrics, paths["metrics"])
                else:
                    paths[artifact].write_text("test output")
                self.assertFalse(is_run_completed(directory))

    def test_invalid_evaluation_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.create_completed_run(directory, evaluation_scope="invalid")
            self.assertFalse(is_run_completed(directory))

    def test_remove_checkpoint_deletes_checkpoint_only(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = prepare_run_directory(
                run_dir
            )

            paths["checkpoint"].write_bytes(
                b"checkpoint"
            )

            paths["metadata"].write_text(
                "metadata",
                encoding="utf-8",
            )

            remove_checkpoint(
                run_dir
            )

            self.assertFalse(
                paths[
                    "checkpoint"
                ].exists()
            )

            self.assertTrue(
                paths[
                    "metadata"
                ].exists()
            )


if __name__ == "__main__":
    unittest.main()
