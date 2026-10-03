import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.results.statistics as statistics_module


class StatisticsTest(unittest.TestCase):
    def _split_summary(
        self,
        methods=(
            "instance_level",
            "true_pair",
            "shuffled_pair",
        ),
        split_seeds=(
            13,
            21,
            40,
            42,
            73,
            101,
        ),
        run_count=6,
    ):
        rows = []

        offsets = {
            "instance_level": 0.00,
            "true_pair": 0.02,
            "shuffled_pair": -0.01,
        }

        for split_index, split_seed in enumerate(
            split_seeds
        ):
            for method in methods:
                base_value = (
                    0.70
                    + 0.01
                    * split_index
                    + offsets[
                        method
                    ]
                )

                row = {
                    "method": method,
                    "split_seed": split_seed,
                    "run_count": run_count,
                }

                for metric in (
                    statistics_module
                    .INFERENTIAL_METRICS
                ):
                    row[
                        f"{metric}_mean"
                    ] = base_value

                    row[
                        f"{metric}_std"
                    ] = 0.01

                rows.append(
                    row
                )

        return pd.DataFrame(
            rows
        )

    def _run_summary(
        self,
        methods=(
            "instance_level",
            "true_pair",
            "shuffled_pair",
        ),
        split_seeds=(
            13,
            21,
        ),
        model_seeds=(
            13,
            21,
        ),
    ):
        rows = []

        offsets = {
            "instance_level": 0.00,
            "true_pair": 0.02,
            "shuffled_pair": -0.01,
        }

        for method in methods:
            for split_index, split_seed in enumerate(
                split_seeds
            ):
                for model_index, model_seed in enumerate(
                    model_seeds
                ):
                    value = (
                        0.70
                        + 0.01
                        * split_index
                        + 0.001
                        * model_index
                        + offsets[
                            method
                        ]
                    )

                    rows.append(
                        {
                            "method": method,
                            "split_seed": split_seed,
                            "model_seed": model_seed,
                            "accuracy": value,
                            "f1_macro": value,
                            "f1_pun": value,
                            "f1_non_pun": value,
                            "pair_ranking_accuracy": value,
                            "pair_exact_match": value,
                            "mean_pair_margin": value,
                            "median_pair_margin": value,
                            "std_pair_margin": value,
                            "duration_seconds": 100.0,
                            "peak_allocated_gb": 6.0,
                            "peak_reserved_gb": 6.75,
                        }
                    )

        return pd.DataFrame(
            rows
        )

    def test_holm_adjustment_is_monotonic_and_bounded(self):
        result = (
            statistics_module.holm_adjust(
                [
                    0.01,
                    0.04,
                    0.03,
                ]
            )
        )

        np.testing.assert_allclose(
            result,
            [
                0.03,
                0.06,
                0.06,
            ],
        )

    def test_rank_biserial_is_positive_when_first_method_is_higher(self):
        result = (
            statistics_module
            .paired_rank_biserial(
                [
                    0.9,
                    0.8,
                    0.7,
                ],
                [
                    0.8,
                    0.7,
                    0.6,
                ],
            )
        )

        self.assertAlmostEqual(
            result,
            1.0,
        )

    def test_build_metric_matrix_preserves_requested_order(self):
        summary = self._split_summary()

        result = (
            statistics_module
            .build_metric_matrix(
                summary_by_split=summary,
                metric="f1_macro",
                methods=(
                    "true_pair",
                    "instance_level",
                    "shuffled_pair",
                ),
                split_seeds=(
                    21,
                    13,
                ),
            )
        )

        self.assertEqual(
            list(
                result.index
            ),
            [
                21,
                13,
            ],
        )

        self.assertEqual(
            list(
                result.columns
            ),
            [
                "true_pair",
                "instance_level",
                "shuffled_pair",
            ],
        )

    def test_identical_methods_have_neutral_omnibus_statistics(self):
        summary = self._split_summary()

        for metric in (
            statistics_module
            .INFERENTIAL_METRICS
        ):
            column = (
                f"{metric}_mean"
            )

            summary[
                column
            ] = summary[
                "split_seed"
            ].map(
                {
                    13: 0.70,
                    21: 0.71,
                    40: 0.72,
                    42: 0.73,
                    73: 0.74,
                    101: 0.75,
                }
            )

        result = (
            statistics_module
            .build_omnibus_statistics(
                summary
            )
        )

        self.assertTrue(
            (
                result[
                    "friedman_statistic"
                ]
                == 0.0
            ).all()
        )

        self.assertTrue(
            (
                result[
                    "p_value"
                ]
                == 1.0
            ).all()
        )

        self.assertTrue(
            (
                result[
                    "kendalls_w"
                ]
                == 0.0
            ).all()
        )

    def test_pairwise_statistics_builds_three_comparisons_per_metric(self):
        summary = self._split_summary()

        result = (
            statistics_module
            .build_pairwise_statistics(
                summary
            )
        )

        self.assertEqual(
            len(result),
            9,
        )

        self.assertTrue(
            (
                result[
                    "p_value_holm"
                ]
                >= result[
                    "p_value_raw"
                ]
            ).all()
        )

        self.assertTrue(
            set(
                result[
                    "rank_biserial"
                ]
            ).issubset(
                {
                    -1.0,
                    1.0,
                }
            )
        )

    def test_descriptive_statistics_uses_split_level_values(self):
        summary = self._split_summary()

        result = (
            statistics_module
            .build_descriptive_statistics(
                summary
            )
        )

        row = result[
            (
                result[
                    "metric"
                ]
                == "f1_macro"
            )
            & (
                result[
                    "method"
                ]
                == "true_pair"
            )
        ].iloc[
            0
        ]

        self.assertEqual(
            row[
                "split_count"
            ],
            6,
        )

        self.assertAlmostEqual(
            row[
                "mean"
            ],
            0.745,
        )

    def test_generate_statistical_analysis_requires_complete_grid(self):
        summary = self._run_summary()

        with patch.object(
            statistics_module,
            "collect_run_summaries",
            return_value=(
                summary,
                [
                    {
                        "method": "instance_level",
                        "split_seed": 13,
                        "model_seed": 40,
                    }
                ],
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "requires a complete experiment grid",
            ):
                (
                    statistics_module
                    .generate_statistical_analysis(
                        methods=(
                            "instance_level",
                            "true_pair",
                            "shuffled_pair",
                        ),
                        split_seeds=(
                            13,
                            21,
                        ),
                        model_seeds=(
                            13,
                            21,
                        ),
                    )
                )

    def test_generate_statistical_analysis_writes_all_outputs(self):
        methods = (
            "instance_level",
            "true_pair",
            "shuffled_pair",
        )

        split_seeds = (
            13,
            21,
        )

        model_seeds = (
            13,
            21,
        )

        summary = self._run_summary(
            methods=methods,
            split_seeds=split_seeds,
            model_seeds=model_seeds,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            descriptive_path = (
                root
                / "descriptive.csv"
            )

            omnibus_path = (
                root
                / "omnibus.csv"
            )

            pairwise_path = (
                root
                / "pairwise.csv"
            )

            metadata_path = (
                root
                / "metadata.json"
            )

            with patch.object(
                statistics_module,
                "collect_run_summaries",
                return_value=(
                    summary,
                    [],
                ),
            ):
                result = (
                    statistics_module
                    .generate_statistical_analysis(
                        methods=methods,
                        split_seeds=split_seeds,
                        model_seeds=model_seeds,
                        descriptive_path=descriptive_path,
                        omnibus_path=omnibus_path,
                        pairwise_path=pairwise_path,
                        metadata_path=metadata_path,
                    )
                )

            self.assertEqual(
                result[
                    "completed_runs"
                ],
                12,
            )

            self.assertTrue(
                descriptive_path.is_file()
            )

            self.assertTrue(
                omnibus_path.is_file()
            )

            self.assertTrue(
                pairwise_path.is_file()
            )

            self.assertTrue(
                metadata_path.is_file()
            )

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                metadata[
                    "inferential_unit"
                ],
                "split_seed",
            )

            self.assertEqual(
                metadata[
                    "multiple_testing_correction"
                ],
                "holm_within_metric",
            )


if __name__ == "__main__":
    unittest.main()