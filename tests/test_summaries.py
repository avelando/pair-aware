import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.results.summaries as summaries_module


class SummariesTest(unittest.TestCase):
    def build_run(
        self,
        run_dir,
        method="instance_level",
        split_seed=13,
        model_seed=40,
    ):
        run_dir = Path(
            run_dir
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = {
            "status": "completed",
            "method": method,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "experiment_id": "experiment-id",
            "attempt": 2,
            "fingerprint": {
                "config_hash": "config-hash",
                "dataset_hash": "dataset-hash",
                "source_hash": "source-hash",
            },
        }

        metrics = {
            "training": {
                "best_validation_f1_macro": 0.8,
                "best_epoch": 3,
                "epochs_trained": 5,
                "optimizer_steps": 100,
                "skipped_optimizer_steps": 1,
                "planned_training_steps": 120,
                "warmup_steps": 12,
            },
            "test_instance": {
                "accuracy": 0.75,
                "precision_non_pun": 0.70,
                "recall_non_pun": 0.80,
                "f1_non_pun": 0.74,
                "precision_pun": 0.80,
                "recall_pun": 0.70,
                "f1_pun": 0.76,
                "precision_macro": 0.75,
                "recall_macro": 0.75,
                "f1_macro": 0.75,
                "precision_weighted": 0.75,
                "recall_weighted": 0.75,
                "f1_weighted": 0.75,
                "tp": 400,
                "tn": 455,
                "fp": 115,
                "fn": 170,
            },
            "test_pair": {
                "pair_count": 570,
                "pair_ranking_accuracy": 0.82,
                "pair_exact_match": 0.55,
                "pair_ties": 4,
                "mean_pair_margin": 1.2,
                "median_pair_margin": 1.0,
                "std_pair_margin": 0.5,
                "min_pair_margin": -0.4,
                "max_pair_margin": 3.1,
            },
            "runtime": {
                "duration_seconds": 90.0,
                "peak_allocated_gb": 6.0,
                "peak_reserved_gb": 6.75,
            },
        }

        (
            run_dir
            / "metadata.json"
        ).write_text(
            json.dumps(
                metadata
            ),
            encoding="utf-8",
        )

        (
            run_dir
            / "metrics.json"
        ).write_text(
            json.dumps(
                metrics
            ),
            encoding="utf-8",
        )

    def test_load_run_summary_flattens_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(
                directory
            )

            self.build_run(
                run_dir
            )

            result = (
                summaries_module.load_run_summary(
                    run_dir
                )
            )

        self.assertEqual(
            result["method"],
            "instance_level",
        )

        self.assertEqual(
            result["split_seed"],
            13,
        )

        self.assertEqual(
            result["model_seed"],
            40,
        )

        self.assertEqual(
            result["attempt"],
            2,
        )

        self.assertEqual(
            result["best_epoch"],
            3,
        )

        self.assertAlmostEqual(
            result["f1_macro"],
            0.75,
        )

        self.assertAlmostEqual(
            result[
                "pair_ranking_accuracy"
            ],
            0.82,
        )

        self.assertAlmostEqual(
            result[
                "peak_reserved_gb"
            ],
            6.75,
        )

    def test_load_run_summary_rejects_missing_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(
                directory
            )

            self.build_run(
                run_dir
            )

            metrics_path = (
                run_dir
                / "metrics.json"
            )

            metrics = json.loads(
                metrics_path.read_text(
                    encoding="utf-8"
                )
            )

            del metrics[
                "test_instance"
            ][
                "f1_macro"
            ]

            metrics_path.write_text(
                json.dumps(
                    metrics
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Missing 'f1_macro'",
            ):
                summaries_module.load_run_summary(
                    run_dir
                )

    def test_collect_run_summaries_tracks_missing_runs(self):
        run_dir = Path(
            "results/instance_level/"
            "split_13/model_seed_13"
        )

        record = {
            "method": "instance_level",
            "split_seed": 13,
            "model_seed": 13,
            "experiment_id": "experiment-id",
        }

        with patch.object(
            summaries_module,
            "get_run_dir",
            return_value=run_dir,
        ), patch.object(
            summaries_module,
            "build_experiment_fingerprint",
            return_value={
                "experiment_id": "experiment-id",
            },
        ), patch.object(
            summaries_module,
            "is_run_completed",
            side_effect=[
                True,
                False,
            ],
        ) as completed_mock, patch.object(
            summaries_module,
            "load_run_summary",
            return_value=record,
        ):
            (
                summary,
                missing,
            ) = summaries_module.collect_run_summaries(
                methods=(
                    "instance_level",
                ),
                split_seeds=(
                    13,
                ),
                model_seeds=(
                    13,
                    21,
                ),
            )

        self.assertEqual(
            len(summary),
            1,
        )

        self.assertEqual(
            len(missing),
            1,
        )

        self.assertEqual(
            missing[0][
                "model_seed"
            ],
            21,
        )

        self.assertEqual(
            completed_mock.call_count,
            2,
        )

        completed_mock.assert_any_call(
            run_dir,
            expected_experiment_id=(
                "experiment-id"
            ),
        )

    def test_collect_run_summaries_rejects_loaded_fingerprint_mismatch(self):
        with patch.object(
            summaries_module,
            "get_run_dir",
            return_value=Path(
                "results/run"
            ),
        ), patch.object(
            summaries_module,
            "build_experiment_fingerprint",
            return_value={
                "experiment_id": "current-id",
            },
        ), patch.object(
            summaries_module,
            "is_run_completed",
            return_value=True,
        ), patch.object(
            summaries_module,
            "load_run_summary",
            return_value={
                "experiment_id": "stored-id",
            },
        ):
            with self.assertRaisesRegex(
                ValueError,
                "experiment ID does not match",
            ):
                summaries_module.collect_run_summaries(
                    methods=(
                        "instance_level",
                    ),
                    split_seeds=(
                        13,
                    ),
                    model_seeds=(
                        13,
                    ),
                )

    def test_summary_by_split_aggregates_model_seeds(self):
        summary = pd.DataFrame(
            [
                self._summary_row(
                    "instance_level",
                    13,
                    13,
                    0.6,
                ),
                self._summary_row(
                    "instance_level",
                    13,
                    21,
                    0.8,
                ),
            ]
        )

        result = (
            summaries_module.build_summary_by_split(
                summary
            )
        )

        self.assertEqual(
            len(result),
            1,
        )

        self.assertEqual(
            result.loc[
                0,
                "run_count",
            ],
            2,
        )

        self.assertAlmostEqual(
            result.loc[
                0,
                "f1_macro_mean",
            ],
            0.7,
        )

        self.assertAlmostEqual(
            result.loc[
                0,
                "f1_macro_std",
            ],
            0.14142135623730953,
        )

    def test_summary_by_method_aggregates_split_means(self):
        summary_by_split = pd.DataFrame(
            [
                self._split_row(
                    "instance_level",
                    13,
                    6,
                    0.6,
                ),
                self._split_row(
                    "instance_level",
                    21,
                    6,
                    0.8,
                ),
            ]
        )

        result = (
            summaries_module.build_summary_by_method(
                summary_by_split
            )
        )

        self.assertEqual(
            result.loc[
                0,
                "split_count",
            ],
            2,
        )

        self.assertEqual(
            result.loc[
                0,
                "run_count",
            ],
            12,
        )

        self.assertAlmostEqual(
            result.loc[
                0,
                "f1_macro_mean",
            ],
            0.7,
        )

        self.assertAlmostEqual(
            result.loc[
                0,
                "f1_macro_std",
            ],
            0.14142135623730953,
        )

    def test_generate_summaries_requires_complete_grid_when_requested(self):
        summary = pd.DataFrame(
            [
                self._summary_row(
                    "instance_level",
                    13,
                    13,
                    0.7,
                )
            ]
        )

        with patch.object(
            summaries_module,
            "collect_run_summaries",
            return_value=(
                summary,
                [
                    {
                        "method": "instance_level",
                        "split_seed": 13,
                        "model_seed": 21,
                    }
                ],
            ),
        ), patch.object(
            summaries_module,
            "write_summary_files",
        ) as write_mock:
            with self.assertRaisesRegex(
                RuntimeError,
                "Incomplete experiment grid",
            ):
                summaries_module.generate_summaries(
                    require_complete=True,
                    methods=(
                        "instance_level",
                    ),
                    split_seeds=(
                        13,
                    ),
                    model_seeds=(
                        13,
                        21,
                    ),
                )

        write_mock.assert_not_called()

    def test_generate_summaries_writes_three_files(self):
        summary = pd.DataFrame(
            [
                self._summary_row(
                    "instance_level",
                    13,
                    13,
                    0.7,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            summary_path = (
                root
                / "summary.csv"
            )

            split_path = (
                root
                / "summary_by_split.csv"
            )

            method_path = (
                root
                / "summary_by_method.csv"
            )

            with patch.object(
                summaries_module,
                "collect_run_summaries",
                return_value=(
                    summary,
                    [],
                ),
            ):
                result = (
                    summaries_module.generate_summaries(
                        require_complete=True,
                        methods=(
                            "instance_level",
                        ),
                        split_seeds=(
                            13,
                        ),
                        model_seeds=(
                            13,
                        ),
                        summary_path=summary_path,
                        summary_by_split_path=split_path,
                        summary_by_method_path=method_path,
                    )
                )

            self.assertTrue(
                summary_path.is_file()
            )

            self.assertTrue(
                split_path.is_file()
            )

            self.assertTrue(
                method_path.is_file()
            )

            self.assertEqual(
                result[
                    "completed_runs"
                ],
                1,
            )

            self.assertEqual(
                result[
                    "expected_runs"
                ],
                1,
            )

            self.assertEqual(
                result[
                    "missing_runs"
                ],
                [],
            )

    def _summary_row(
        self,
        method,
        split_seed,
        model_seed,
        value,
    ):
        row = {
            "method": method,
            "split_seed": split_seed,
            "model_seed": model_seed,
        }

        for metric in (
            summaries_module
            .SUMMARY_METRIC_COLUMNS
        ):
            row[
                metric
            ] = value

        return row

    def _split_row(
        self,
        method,
        split_seed,
        run_count,
        value,
    ):
        row = {
            "method": method,
            "split_seed": split_seed,
            "run_count": run_count,
        }

        for metric in (
            summaries_module
            .SUMMARY_METRIC_COLUMNS
        ):
            row[
                f"{metric}_mean"
            ] = value

            row[
                f"{metric}_std"
            ] = 0.1

        return row


if __name__ == "__main__":
    unittest.main()