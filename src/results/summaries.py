import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    METHODS,
    MODEL_SEEDS,
    SPLIT_SEEDS,
)
from src.experiments.fingerprint import build_experiment_fingerprint
from src.paths import (
    SUMMARY_BY_METHOD_PATH,
    SUMMARY_BY_SPLIT_PATH,
    SUMMARY_PATH,
    get_run_dir,
)
from src.results.io import (
    get_run_artifact_paths,
    is_run_completed,
    write_dataframe,
)


TRAINING_COLUMNS = (
    "best_validation_f1_macro",
    "best_epoch",
    "epochs_trained",
    "optimizer_steps",
    "skipped_optimizer_steps",
    "planned_training_steps",
    "warmup_steps",
)

TRAINING_INTEGER_COLUMNS = {
    "best_epoch",
    "epochs_trained",
    "optimizer_steps",
    "skipped_optimizer_steps",
    "planned_training_steps",
    "warmup_steps",
}

INSTANCE_COLUMNS = (
    "accuracy",
    "precision_non_pun",
    "recall_non_pun",
    "f1_non_pun",
    "precision_pun",
    "recall_pun",
    "f1_pun",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "tp",
    "tn",
    "fp",
    "fn",
)

INSTANCE_INTEGER_COLUMNS = {
    "tp",
    "tn",
    "fp",
    "fn",
}

PAIR_COLUMNS = (
    "pair_count",
    "pair_ranking_accuracy",
    "pair_exact_match",
    "pair_ties",
    "mean_pair_margin",
    "median_pair_margin",
    "std_pair_margin",
    "min_pair_margin",
    "max_pair_margin",
)

PAIR_INTEGER_COLUMNS = {
    "pair_count",
    "pair_ties",
}

RUNTIME_COLUMNS = (
    "duration_seconds",
    "peak_allocated_gb",
    "peak_reserved_gb",
)

SUMMARY_METRIC_COLUMNS = (
    "accuracy",
    "f1_macro",
    "f1_pun",
    "f1_non_pun",
    "pair_ranking_accuracy",
    "pair_exact_match",
    "mean_pair_margin",
    "median_pair_margin",
    "std_pair_margin",
    "duration_seconds",
    "peak_allocated_gb",
    "peak_reserved_gb",
)


def _read_json(path):
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object in {path}."
        )

    return value


def _require_mapping(
    mapping,
    key,
    source,
):
    value = mapping.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"Missing or invalid '{key}' section in {source}."
        )

    return value


def _require_value(
    mapping,
    key,
    source,
):
    if key not in mapping:
        raise ValueError(
            f"Missing '{key}' in {source}."
        )

    return mapping[key]


def _extract_section(
    mapping,
    columns,
    integer_columns,
    source,
):
    result = {}

    for column in columns:
        value = _require_value(
            mapping,
            column,
            source,
        )

        result[column] = (
            int(value)
            if column in integer_columns
            else float(value)
        )

    return result


def load_run_summary(run_dir):
    run_dir = Path(run_dir)

    paths = get_run_artifact_paths(
        run_dir
    )

    metadata = _read_json(
        paths["metadata"]
    )

    metrics = _read_json(
        paths["metrics"]
    )

    training = _require_mapping(
        metrics,
        "training",
        paths["metrics"],
    )

    instance = _require_mapping(
        metrics,
        "test_instance",
        paths["metrics"],
    )

    pair = _require_mapping(
        metrics,
        "test_pair",
        paths["metrics"],
    )

    runtime = _require_mapping(
        metrics,
        "runtime",
        paths["metrics"],
    )

    fingerprint = _require_mapping(
        metadata,
        "fingerprint",
        paths["metadata"],
    )

    result = {
        "method": _require_value(
            metadata,
            "method",
            paths["metadata"],
        ),
        "split_seed": int(
            _require_value(
                metadata,
                "split_seed",
                paths["metadata"],
            )
        ),
        "model_seed": int(
            _require_value(
                metadata,
                "model_seed",
                paths["metadata"],
            )
        ),
        "experiment_id": _require_value(
            metadata,
            "experiment_id",
            paths["metadata"],
        ),
        "attempt": int(
            _require_value(
                metadata,
                "attempt",
                paths["metadata"],
            )
        ),
        "config_hash": _require_value(
            fingerprint,
            "config_hash",
            paths["metadata"],
        ),
        "dataset_hash": _require_value(
            fingerprint,
            "dataset_hash",
            paths["metadata"],
        ),
        "source_hash": _require_value(
            fingerprint,
            "source_hash",
            paths["metadata"],
        ),
    }

    result.update(
        _extract_section(
            training,
            TRAINING_COLUMNS,
            TRAINING_INTEGER_COLUMNS,
            paths["metrics"],
        )
    )

    result.update(
        _extract_section(
            instance,
            INSTANCE_COLUMNS,
            INSTANCE_INTEGER_COLUMNS,
            paths["metrics"],
        )
    )

    result.update(
        _extract_section(
            pair,
            PAIR_COLUMNS,
            PAIR_INTEGER_COLUMNS,
            paths["metrics"],
        )
    )

    result.update(
        _extract_section(
            runtime,
            RUNTIME_COLUMNS,
            set(),
            paths["metrics"],
        )
    )

    return result


def collect_run_summaries(
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
    model_seeds=MODEL_SEEDS,
):
    records = []
    missing_runs = []

    for method in methods:
        for split_seed in split_seeds:
            for model_seed in model_seeds:
                run_dir = get_run_dir(
                    method,
                    split_seed,
                    model_seed,
                )

                experiment_fingerprint = (
                    build_experiment_fingerprint(
                        method,
                        split_seed,
                        model_seed,
                    )
                )

                experiment_id = (
                    experiment_fingerprint[
                        "experiment_id"
                    ]
                )

                if not is_run_completed(
                    run_dir,
                    expected_experiment_id=experiment_id,
                ):
                    missing_runs.append(
                        {
                            "method": method,
                            "split_seed": split_seed,
                            "model_seed": model_seed,
                            "run_dir": str(run_dir),
                            "experiment_id": experiment_id,
                        }
                    )

                    continue

                record = load_run_summary(
                    run_dir
                )

                if (
                    record["experiment_id"]
                    != experiment_id
                ):
                    raise ValueError(
                        "Completed run experiment ID does not match "
                        f"the current fingerprint: {run_dir}."
                    )

                records.append(
                    record
                )

    return (
        pd.DataFrame(records),
        missing_runs,
    )


def build_summary_by_split(summary):
    if summary.empty:
        raise ValueError(
            "Run summary cannot be empty."
        )

    aggregations = {
        "model_seed": "count",
    }

    for column in SUMMARY_METRIC_COLUMNS:
        aggregations[column] = [
            "mean",
            "std",
        ]

    grouped = (
        summary.groupby(
            [
                "method",
                "split_seed",
            ],
            sort=False,
            as_index=False,
        )
        .agg(aggregations)
    )

    grouped.columns = [
        "method",
        "split_seed",
        "run_count",
        *[
            f"{metric}_{statistic}"
            for metric in SUMMARY_METRIC_COLUMNS
            for statistic in (
                "mean",
                "std",
            )
        ],
    ]

    grouped[
        "run_count"
    ] = grouped[
        "run_count"
    ].astype(int)

    return grouped


def build_summary_by_method(
    summary_by_split,
):
    if summary_by_split.empty:
        raise ValueError(
            "Split summary cannot be empty."
        )

    records = []

    for (
        method,
        group,
    ) in summary_by_split.groupby(
        "method",
        sort=False,
    ):
        record = {
            "method": method,
            "split_count": int(
                len(group)
            ),
            "run_count": int(
                group[
                    "run_count"
                ].sum()
            ),
        }

        for metric in SUMMARY_METRIC_COLUMNS:
            values = group[
                f"{metric}_mean"
            ].astype(float)

            record[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            record[
                f"{metric}_std"
            ] = (
                float(
                    values.std(
                        ddof=1
                    )
                )
                if len(values) > 1
                else 0.0
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


def write_summary_files(
    summary,
    summary_by_split,
    summary_by_method,
    summary_path=SUMMARY_PATH,
    summary_by_split_path=SUMMARY_BY_SPLIT_PATH,
    summary_by_method_path=SUMMARY_BY_METHOD_PATH,
):
    write_dataframe(
        summary,
        summary_path,
    )

    write_dataframe(
        summary_by_split,
        summary_by_split_path,
    )

    write_dataframe(
        summary_by_method,
        summary_by_method_path,
    )

    return {
        "summary": str(
            summary_path
        ),
        "summary_by_split": str(
            summary_by_split_path
        ),
        "summary_by_method": str(
            summary_by_method_path
        ),
    }


def generate_summaries(
    require_complete=False,
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
    model_seeds=MODEL_SEEDS,
    summary_path=SUMMARY_PATH,
    summary_by_split_path=SUMMARY_BY_SPLIT_PATH,
    summary_by_method_path=SUMMARY_BY_METHOD_PATH,
):
    (
        summary,
        missing_runs,
    ) = collect_run_summaries(
        methods=methods,
        split_seeds=split_seeds,
        model_seeds=model_seeds,
    )

    expected_count = (
        len(methods)
        * len(split_seeds)
        * len(model_seeds)
    )

    if summary.empty:
        raise RuntimeError(
            "No completed current experiment runs were found."
        )

    if (
        require_complete
        and missing_runs
    ):
        raise RuntimeError(
            "Incomplete experiment grid: "
            f"{len(summary)} of "
            f"{expected_count} runs are complete."
        )

    summary_by_split = (
        build_summary_by_split(
            summary
        )
    )

    summary_by_method = (
        build_summary_by_method(
            summary_by_split
        )
    )

    paths = write_summary_files(
        summary=summary,
        summary_by_split=summary_by_split,
        summary_by_method=summary_by_method,
        summary_path=summary_path,
        summary_by_split_path=summary_by_split_path,
        summary_by_method_path=summary_by_method_path,
    )

    return {
        "completed_runs": int(
            len(summary)
        ),
        "expected_runs": int(
            expected_count
        ),
        "missing_runs": missing_runs,
        "paths": paths,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--require-complete",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = generate_summaries(
        require_complete=(
            args.require_complete
        )
    )

    print(
        f"Completed runs: "
        f"{result['completed_runs']} / "
        f"{result['expected_runs']}"
    )

    print(
        f"Missing runs: "
        f"{len(result['missing_runs'])}"
    )

    print(
        f"Summary: "
        f"{result['paths']['summary']}"
    )

    print(
        f"Summary by split: "
        f"{result['paths']['summary_by_split']}"
    )

    print(
        f"Summary by method: "
        f"{result['paths']['summary_by_method']}"
    )


if __name__ == "__main__":
    main()