import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import scipy
from scipy.stats import (
    friedmanchisquare,
    rankdata,
    wilcoxon,
)

from src.config import (
    METHODS,
    MODEL_SEEDS,
    SPLIT_SEEDS,
)
from src.paths import (
    STATISTICS_DESCRIPTIVE_PATH,
    STATISTICS_METADATA_PATH,
    STATISTICS_OMNIBUS_PATH,
    STATISTICS_PAIRWISE_PATH,
)
from src.results.io import (
    write_dataframe,
    write_json,
)
from src.results.summaries import (
    build_summary_by_split,
    collect_run_summaries,
)


STATISTICAL_ANALYSIS_VERSION = 1

DEFAULT_ALPHA = 0.05

INFERENTIAL_METRICS = (
    "f1_macro",
    "pair_ranking_accuracy",
    "pair_exact_match",
)


def validate_alpha(alpha):
    alpha = float(alpha)

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must be between zero and one."
        )

    return alpha


def holm_adjust(p_values):
    values = np.asarray(
        p_values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "p_values must be one-dimensional."
        )

    if len(values) == 0:
        return np.asarray(
            [],
            dtype=float,
        )

    if (
        np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(
            "p_values must contain finite values "
            "between zero and one."
        )

    order = np.argsort(
        values,
        kind="stable",
    )

    adjusted_sorted = np.empty(
        len(values),
        dtype=float,
    )

    running_maximum = 0.0
    total = len(values)

    for position, index in enumerate(
        order
    ):
        adjusted = min(
            1.0,
            float(
                values[index]
                * (
                    total
                    - position
                )
            ),
        )

        running_maximum = max(
            running_maximum,
            adjusted,
        )

        adjusted_sorted[
            position
        ] = running_maximum

    adjusted = np.empty(
        len(values),
        dtype=float,
    )

    for position, index in enumerate(
        order
    ):
        adjusted[index] = (
            adjusted_sorted[
                position
            ]
        )

    return adjusted


def paired_rank_biserial(
    first,
    second,
):
    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    if first.shape != second.shape:
        raise ValueError(
            "Paired samples must have the same shape."
        )

    if first.ndim != 1:
        raise ValueError(
            "Paired samples must be one-dimensional."
        )

    differences = (
        first
        - second
    )

    nonzero = (
        differences != 0.0
    )

    differences = differences[
        nonzero
    ]

    if len(differences) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(
            differences
        ),
        method="average",
    )

    positive = float(
        ranks[
            differences > 0.0
        ].sum()
    )

    negative = float(
        ranks[
            differences < 0.0
        ].sum()
    )

    denominator = (
        positive
        + negative
    )

    if denominator == 0.0:
        return 0.0

    return float(
        (
            positive
            - negative
        )
        / denominator
    )


def build_metric_matrix(
    summary_by_split,
    metric,
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
):
    column = (
        f"{metric}_mean"
    )

    if column not in summary_by_split:
        raise ValueError(
            "Missing split-level metric column: "
            f"{column}."
        )

    duplicated = (
        summary_by_split[
            [
                "method",
                "split_seed",
            ]
        ]
        .duplicated()
        .any()
    )

    if duplicated:
        raise ValueError(
            "Split summary contains duplicate "
            "method/split_seed rows."
        )

    matrix = (
        summary_by_split
        .pivot(
            index="split_seed",
            columns="method",
            values=column,
        )
        .reindex(
            index=list(
                split_seeds
            ),
            columns=list(
                methods
            ),
        )
    )

    if matrix.isna().any().any():
        raise ValueError(
            "Incomplete split-level data "
            f"for metric '{metric}'."
        )

    return matrix.astype(
        float
    )


def validate_split_run_counts(
    summary_by_split,
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
    model_seeds=MODEL_SEEDS,
):
    expected_run_count = len(
        model_seeds
    )

    expected_rows = (
        len(methods)
        * len(split_seeds)
    )

    selected = (
        summary_by_split[
            summary_by_split[
                "method"
            ].isin(
                methods
            )
            & summary_by_split[
                "split_seed"
            ].isin(
                split_seeds
            )
        ]
    )

    if len(selected) != expected_rows:
        raise ValueError(
            "Split summary does not contain "
            "every expected method/split_seed row."
        )

    if not (
        selected[
            "run_count"
        ].astype(int)
        == expected_run_count
    ).all():
        raise ValueError(
            "Every method/split_seed row must "
            "contain all configured model seeds."
        )


def build_descriptive_statistics(
    summary_by_split,
    metrics=INFERENTIAL_METRICS,
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
):
    records = []

    for metric in metrics:
        matrix = build_metric_matrix(
            summary_by_split=summary_by_split,
            metric=metric,
            methods=methods,
            split_seeds=split_seeds,
        )

        for method in methods:
            values = matrix[
                method
            ].to_numpy(
                dtype=float
            )

            records.append(
                {
                    "metric": metric,
                    "method": method,
                    "split_count": int(
                        len(values)
                    ),
                    "mean": float(
                        values.mean()
                    ),
                    "std": (
                        float(
                            values.std(
                                ddof=1
                            )
                        )
                        if len(values) > 1
                        else 0.0
                    ),
                    "median": float(
                        np.median(
                            values
                        )
                    ),
                    "min": float(
                        values.min()
                    ),
                    "max": float(
                        values.max()
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def build_omnibus_statistics(
    summary_by_split,
    metrics=INFERENTIAL_METRICS,
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
    alpha=DEFAULT_ALPHA,
):
    alpha = validate_alpha(
        alpha
    )

    records = []

    for metric in metrics:
        matrix = build_metric_matrix(
            summary_by_split=summary_by_split,
            metric=metric,
            methods=methods,
            split_seeds=split_seeds,
        )

        samples = [
            matrix[
                method
            ].to_numpy(
                dtype=float
            )
            for method in methods
        ]

        if all(
            np.array_equal(
                samples[0],
                sample,
            )
            for sample in samples[
                1:
            ]
        ):
            statistic = 0.0
            p_value = 1.0

        else:
            test = friedmanchisquare(
                *samples
            )

            statistic = float(
                test.statistic
            )

            p_value = float(
                test.pvalue
            )

        n_blocks = len(
            matrix
        )

        method_count = len(
            methods
        )

        kendalls_w = float(
            statistic
            / (
                n_blocks
                * (
                    method_count
                    - 1
                )
            )
        )

        records.append(
            {
                "metric": metric,
                "n_splits": int(
                    n_blocks
                ),
                "method_count": int(
                    method_count
                ),
                "friedman_statistic": (
                    statistic
                ),
                "p_value": p_value,
                "alpha": alpha,
                "significant": bool(
                    p_value
                    <= alpha
                ),
                "kendalls_w": (
                    kendalls_w
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def _wilcoxon_test(
    first,
    second,
):
    differences = (
        first
        - second
    )

    if np.all(
        differences
        == 0.0
    ):
        return (
            0.0,
            1.0,
        )

    result = wilcoxon(
        first,
        second,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def build_pairwise_statistics(
    summary_by_split,
    metrics=INFERENTIAL_METRICS,
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
    alpha=DEFAULT_ALPHA,
):
    alpha = validate_alpha(
        alpha
    )

    records = []

    for metric in metrics:
        matrix = build_metric_matrix(
            summary_by_split=summary_by_split,
            metric=metric,
            methods=methods,
            split_seeds=split_seeds,
        )

        metric_records = []

        for (
            method_a,
            method_b,
        ) in combinations(
            methods,
            2,
        ):
            first = matrix[
                method_a
            ].to_numpy(
                dtype=float
            )

            second = matrix[
                method_b
            ].to_numpy(
                dtype=float
            )

            differences = (
                first
                - second
            )

            (
                statistic,
                p_value,
            ) = _wilcoxon_test(
                first,
                second,
            )

            metric_records.append(
                {
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_splits": int(
                        len(
                            differences
                        )
                    ),
                    "method_a_mean": float(
                        first.mean()
                    ),
                    "method_b_mean": float(
                        second.mean()
                    ),
                    "mean_difference_a_minus_b": float(
                        differences.mean()
                    ),
                    "median_difference_a_minus_b": float(
                        np.median(
                            differences
                        )
                    ),
                    "wilcoxon_statistic": (
                        statistic
                    ),
                    "p_value_raw": (
                        p_value
                    ),
                    "rank_biserial": (
                        paired_rank_biserial(
                            first,
                            second,
                        )
                    ),
                }
            )

        adjusted = holm_adjust(
            [
                record[
                    "p_value_raw"
                ]
                for record in metric_records
            ]
        )

        for (
            record,
            p_value_holm,
        ) in zip(
            metric_records,
            adjusted,
        ):
            record[
                "p_value_holm"
            ] = float(
                p_value_holm
            )

            record[
                "alpha"
            ] = alpha

            record[
                "significant_holm"
            ] = bool(
                p_value_holm
                <= alpha
            )

            records.append(
                record
            )

    return pd.DataFrame(
        records
    )


def write_statistical_analysis(
    descriptive,
    omnibus,
    pairwise,
    metadata,
    descriptive_path=(
        STATISTICS_DESCRIPTIVE_PATH
    ),
    omnibus_path=(
        STATISTICS_OMNIBUS_PATH
    ),
    pairwise_path=(
        STATISTICS_PAIRWISE_PATH
    ),
    metadata_path=(
        STATISTICS_METADATA_PATH
    ),
):
    write_dataframe(
        descriptive,
        descriptive_path,
    )

    write_dataframe(
        omnibus,
        omnibus_path,
    )

    write_dataframe(
        pairwise,
        pairwise_path,
    )

    write_json(
        metadata,
        metadata_path,
    )

    return {
        "descriptive": str(
            descriptive_path
        ),
        "omnibus": str(
            omnibus_path
        ),
        "pairwise": str(
            pairwise_path
        ),
        "metadata": str(
            metadata_path
        ),
    }


def generate_statistical_analysis(
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
    model_seeds=MODEL_SEEDS,
    metrics=INFERENTIAL_METRICS,
    alpha=DEFAULT_ALPHA,
    descriptive_path=(
        STATISTICS_DESCRIPTIVE_PATH
    ),
    omnibus_path=(
        STATISTICS_OMNIBUS_PATH
    ),
    pairwise_path=(
        STATISTICS_PAIRWISE_PATH
    ),
    metadata_path=(
        STATISTICS_METADATA_PATH
    ),
):
    alpha = validate_alpha(
        alpha
    )

    (
        summary,
        missing_runs,
    ) = collect_run_summaries(
        methods=methods,
        split_seeds=split_seeds,
        model_seeds=model_seeds,
    )

    expected_runs = (
        len(methods)
        * len(split_seeds)
        * len(model_seeds)
    )

    if missing_runs:
        raise RuntimeError(
            "Statistical analysis requires "
            "a complete experiment grid: "
            f"{len(summary)} of "
            f"{expected_runs} runs are complete."
        )

    if len(summary) != expected_runs:
        raise RuntimeError(
            "Statistical analysis requires "
            "exactly the complete configured grid."
        )

    summary_by_split = (
        build_summary_by_split(
            summary
        )
    )

    validate_split_run_counts(
        summary_by_split=summary_by_split,
        methods=methods,
        split_seeds=split_seeds,
        model_seeds=model_seeds,
    )

    descriptive = (
        build_descriptive_statistics(
            summary_by_split=summary_by_split,
            metrics=metrics,
            methods=methods,
            split_seeds=split_seeds,
        )
    )

    omnibus = (
        build_omnibus_statistics(
            summary_by_split=summary_by_split,
            metrics=metrics,
            methods=methods,
            split_seeds=split_seeds,
            alpha=alpha,
        )
    )

    pairwise = (
        build_pairwise_statistics(
            summary_by_split=summary_by_split,
            metrics=metrics,
            methods=methods,
            split_seeds=split_seeds,
            alpha=alpha,
        )
    )

    metadata = {
        "analysis_version": (
            STATISTICAL_ANALYSIS_VERSION
        ),
        "alpha": alpha,
        "methods": list(
            methods
        ),
        "split_seeds": [
            int(seed)
            for seed in split_seeds
        ],
        "model_seeds": [
            int(seed)
            for seed in model_seeds
        ],
        "metrics": list(
            metrics
        ),
        "expected_runs": int(
            expected_runs
        ),
        "completed_runs": int(
            len(summary)
        ),
        "inferential_unit": (
            "split_seed"
        ),
        "model_seed_aggregation": (
            "arithmetic_mean_within_split_seed"
        ),
        "omnibus_test": (
            "friedman"
        ),
        "post_hoc_test": (
            "two_sided_wilcoxon_signed_rank"
        ),
        "multiple_testing_correction": (
            "holm_within_metric"
        ),
        "omnibus_effect_size": (
            "kendalls_w"
        ),
        "pairwise_effect_size": (
            "paired_rank_biserial"
        ),
        "pairwise_effect_direction": (
            "positive_means_method_a_higher"
        ),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }

    paths = (
        write_statistical_analysis(
            descriptive=descriptive,
            omnibus=omnibus,
            pairwise=pairwise,
            metadata=metadata,
            descriptive_path=descriptive_path,
            omnibus_path=omnibus_path,
            pairwise_path=pairwise_path,
            metadata_path=metadata_path,
        )
    )

    return {
        "expected_runs": int(
            expected_runs
        ),
        "completed_runs": int(
            len(summary)
        ),
        "paths": paths,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = (
        generate_statistical_analysis(
            alpha=args.alpha
        )
    )

    print(
        f"Completed runs: "
        f"{result['completed_runs']} / "
        f"{result['expected_runs']}"
    )

    print(
        f"Descriptive statistics: "
        f"{result['paths']['descriptive']}"
    )

    print(
        f"Omnibus statistics: "
        f"{result['paths']['omnibus']}"
    )

    print(
        f"Pairwise statistics: "
        f"{result['paths']['pairwise']}"
    )

    print(
        f"Statistical metadata: "
        f"{result['paths']['metadata']}"
    )


if __name__ == "__main__":
    main()