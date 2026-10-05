import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_SEEDS, SPLIT_SEEDS
from src.data.loading import load_split_directory
from src.evaluation.artifacts import build_evaluation_metrics, evaluate_run_splits
from src.experiments.fingerprint import build_experiment_fingerprint
from src.experiments.run_confirmation import CONFIRMATION_ROOT, build_confirmation_grid
from src.pair_loss import format_pair_loss_weight
from src.paths import get_run_dir
from src.results.confirmation import DEFAULT_SELECTION_PATH, load_screening_selection
from src.results.io import get_run_artifact_paths, is_run_completed, write_dataframe, write_json
from src.results.selection import METRIC_FIELDS, _canonical, _hash_file, _read_json


RUN_METRICS = tuple(
    f"{split}_{name}"
    for split in ("validation", "test")
    for name in (*METRIC_FIELDS, "f1_calibration_gain", "pair_exact_match_calibration_gain")
)


def _assert_metrics(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Evaluation metric sections do not match the prediction exports.")
        for key in expected:
            _assert_metrics(actual[key], expected[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("Evaluation metric arrays do not match the prediction exports.")
        for first, second in zip(actual, expected, strict=True):
            _assert_metrics(first, second)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not np.isclose(actual, expected, rtol=1e-6, atol=1e-7):
            raise ValueError("Evaluation metrics do not match the prediction exports.")
    elif actual != expected:
        raise ValueError("Evaluation metadata do not match the prediction exports.")


def _verify_run_exports(paths, split_data):
    saved = {}
    for split in ("validation", "test"):
        prefix = "validation_" if split == "validation" else ""
        instances = pd.read_csv(paths[f"{prefix}predictions"], dtype={"id": str, "pair_id": str, "suffix": str})
        pairs = pd.read_csv(paths[f"{prefix}pair_predictions"], dtype={"pair_id": str, "pun_id": str, "non_pun_id": str})
        reference = split_data[split][["id", "text", "label"]].reset_index(drop=True)
        exported = instances[["id", "text", "true_label"]].rename(columns={"true_label": "label"})
        pd.testing.assert_frame_equal(exported, reference, check_dtype=False)
        saved[split] = (instances, pairs)
    results = evaluate_run_splits(
        split_data,
        lambda split, frame: (
            saved[split][0][["logit_non_pun", "logit_pun"]].to_numpy(), frame["label"].to_numpy(),
        ),
        evaluation_scope="full",
    )
    for split, (instances, pairs) in saved.items():
        for actual, expected in ((instances, results[split]["predictions"]), (pairs, results[split]["pair_predictions"])):
            pd.testing.assert_frame_equal(
                actual.loc[:, expected.columns].reset_index(drop=True), expected.reset_index(drop=True),
                check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-7,
            )
    metrics = _read_json(paths["metrics"])
    rebuilt = build_evaluation_metrics(results)
    for section, expected in rebuilt.items():
        _assert_metrics(metrics.get(section), expected)
    return rebuilt


def _flatten_metrics(metrics):
    result = {}
    for split in ("validation", "test"):
        for name, (section, field) in METRIC_FIELDS.items():
            section = section.replace("validation_", f"{split}_") if section != "validation_threshold" else section
            result[f"{split}_{name}"] = float(metrics[section][field])
        result[f"{split}_f1_calibration_gain"] = result[f"{split}_f1_macro_calibrated"] - result[f"{split}_f1_macro"]
        result[f"{split}_pair_exact_match_calibration_gain"] = result[f"{split}_pair_exact_match_calibrated"] - result[f"{split}_pair_exact_match"]
    return result


def collect_confirmation_records(tasks, split_data):
    records = []
    for task in tasks:
        run_dir = get_run_dir(**task)
        paths = get_run_artifact_paths(run_dir)
        expected = build_experiment_fingerprint(**task)
        metadata = _read_json(paths["metadata"])
        fingerprint = metadata.get("fingerprint", {})
        if (
            any(metadata.get(key) != task[key] for key in ("method", "split_seed", "model_seed", "evaluation_scope"))
            or any(fingerprint.get(key) != expected[key] for key in ("config_hash", "dataset_hash", "source_hash"))
            or fingerprint.get("version") != expected["fingerprint_version"]
            or _canonical(fingerprint.get("config")) != _canonical(expected["config"])
            or metadata.get("training", {}).get("pair_loss_weight") != task["pair_loss_weight"]
            or not is_run_completed(run_dir, expected_experiment_id=expected["experiment_id"])
        ):
            raise RuntimeError(f"Incomplete or incompatible confirmation run: {run_dir}.")
        metrics = _verify_run_exports(paths, split_data[task["split_seed"]])
        records.append({
            **task, **_flatten_metrics(metrics), "run_dir": str(run_dir),
            **{key: expected[key] for key in ("experiment_id", "config_hash", "dataset_hash", "source_hash")},
            **{f"{name}_sha256": _hash_file(paths[name]) for name in (
                "metadata", "metrics", "validation_predictions", "validation_pair_predictions", "predictions", "pair_predictions",
            )},
        })
    return pd.DataFrame(records)


def _within_splits(records, groups, metrics):
    grouped = records.groupby(groups, sort=True)
    return pd.concat((
        grouped.size().rename("run_count"),
        grouped[list(metrics)].mean().add_suffix("_mean"),
        grouped[list(metrics)].std(ddof=1).add_suffix("_std_model_seeds"),
    ), axis=1).reset_index()


def _across_splits(by_split, groups, metrics):
    rows = []
    for identity, group in by_split.groupby(groups, sort=True):
        identity = identity if isinstance(identity, tuple) else (identity,)
        row = {**dict(zip(groups, identity, strict=True)), "split_count": len(group), "run_count": int(group["run_count"].sum())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[f"{metric}_mean"].mean())
            row[f"{metric}_std_between_splits"] = float(group[f"{metric}_mean"].std(ddof=1))
            row[f"{metric}_mean_std_model_seeds"] = float(group[f"{metric}_std_model_seeds"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_confirmation_tables(records, weight):
    keys = ["method", "pair_loss_weight", "split_seed", "model_seed"]
    tasks = build_confirmation_grid(weight)
    expected = {tuple(task[key] for key in keys) for task in tasks}
    if not {"evaluation_scope", *keys, *RUN_METRICS}.issubset(records.columns):
        raise ValueError("Confirmation records are missing required columns.")
    actual = set(records[keys].itertuples(index=False, name=None))
    if len(records) != len(tasks) or actual != expected or records.duplicated(keys).any():
        raise ValueError("Analysis requires the complete unique confirmation grid for one weight.")
    if not records["evaluation_scope"].eq("full").all() or not np.isfinite(records[list(RUN_METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Confirmation requires full evaluation with finite metrics.")
    for split in ("validation", "test"):
        bounded = [f"{split}_{name}" for name in METRIC_FIELDS if name not in {"mean_pair_margin", "score_threshold", "log_loss"}]
        gains = [f"{split}_f1_calibration_gain", f"{split}_pair_exact_match_calibration_gain"]
        if not records[bounded].ge(0).all().all() or not records[bounded].le(1).all().all():
            raise ValueError("Probability and classification metrics must lie in [0, 1].")
        if not records[gains].ge(-1).all().all() or not records[gains].le(1).all().all() or not records[f"{split}_log_loss"].ge(0).all():
            raise ValueError("Calibration gains or log losses are invalid.")
    by_split = _within_splits(records, ["method", "pair_loss_weight", "split_seed"], RUN_METRICS)
    by_method = _across_splits(by_split, ["method", "pair_loss_weight"], RUN_METRICS)
    paired_keys = ["pair_loss_weight", "split_seed", "model_seed"]
    first = records.loc[records["method"] == "true_pair"].set_index(paired_keys)[list(RUN_METRICS)].sort_index()
    second = records.loc[records["method"] == "shuffled_pair"].set_index(paired_keys)[list(RUN_METRICS)].sort_index()
    paired = (first - second).add_suffix("_true_minus_shuffled").reset_index()
    differences = tuple(f"{metric}_true_minus_shuffled" for metric in RUN_METRICS)
    paired_by_split = _within_splits(paired, ["pair_loss_weight", "split_seed"], differences)
    paired_summary = _across_splits(paired_by_split, ["pair_loss_weight"], differences)
    return {
        "runs": records, "by_split": by_split, "by_method": by_method,
        "paired_runs": paired, "paired_by_split": paired_by_split, "paired_summary": paired_summary,
    }


def _selection_overlap(split_data):
    rows = []
    for seed in SPLIT_SEEDS:
        test_ids = set(split_data[seed]["test"]["id"])
        validation_ids = set().union(*(
            set(split_data[other]["validation"]["id"]) for other in SPLIT_SEEDS if other != seed
        ))
        rows.append({
            "split_seed": seed, "test_instance_count": len(test_ids),
            "test_instances_in_other_validation_splits": len(test_ids & validation_ids),
        })
    return rows


def generate_confirmation_analysis(selection_path=DEFAULT_SELECTION_PATH, confirmation_root=CONFIRMATION_ROOT):
    decision = {**load_screening_selection(selection_path), "status": "validated"}
    weight = decision["selected_pair_loss_weight"]
    tasks = build_confirmation_grid(weight)
    root = Path(confirmation_root) / f"lambda_{format_pair_loss_weight(weight)}"
    plan = _read_json(root / "confirmation_plan.json")
    report = _read_json(root / "confirmation_result.json")
    if (
        plan.get("phase") != "confirmation" or plan.get("tasks") != tasks or plan.get("expected_runs") != len(tasks)
        or report.get("status") != "completed" or report.get("artifacts_verified") is not True
        or report.get("tasks") != tasks
        or any(
            saved.get(key) != decision[key]
            for saved in (plan.get("selection", {}), report.get("selection", {}))
            for key in ("status", "selected_pair_loss_weight", "selection_sha256")
        )
    ):
        raise RuntimeError("Analysis requires a completed confirmation linked to the validated selection.")
    split_data = {seed: load_split_directory(seed, split_names=("validation", "test")) for seed in SPLIT_SEEDS}
    records = collect_confirmation_records(tasks, split_data)
    tables = build_confirmation_tables(records, weight)
    output_dir = root / "analysis"
    outputs = {}
    for name, dataframe in tables.items():
        path = output_dir / f"confirmation_{name}.csv"
        write_dataframe(dataframe, path)
        outputs[name] = {"path": str(path), "sha256": _hash_file(path), "rows": len(dataframe)}
    metadata = {
        "analysis_version": 1, "status": "completed", "analysis_type": "descriptive_and_paired_differences",
        "selection": decision, "expected_runs": len(tasks), "completed_runs": len(records),
        "split_seeds": list(SPLIT_SEEDS), "model_seeds": list(MODEL_SEEDS),
        "classification_decisions": ["default", "validation_fitted_threshold"],
        "aggregation": "mean_over_model_seeds_then_equal_mean_over_split_seeds",
        "paired_difference": "true_pair_minus_shuffled_pair_with_matching_split_and_model_seeds",
        "threshold_fit_split": "validation", "calibration_changes_probabilities": False,
        "analysis_source_sha256": _hash_file(__file__),
        "confirmation_plan_sha256": _hash_file(root / "confirmation_plan.json"),
        "confirmation_result_sha256": _hash_file(root / "confirmation_result.json"),
        "selection_test_overlap": _selection_overlap(split_data),
        "interpretation": "repeated_partition_post_selection_analysis_not_independent_holdout_confirmation",
        "outputs": outputs,
    }
    metadata_path = output_dir / "analysis.json"
    write_json(metadata, metadata_path)
    return {"completed_runs": len(records), "metadata_path": str(metadata_path), "outputs": outputs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    args = parser.parse_args()
    result = generate_confirmation_analysis(args.selection)
    print(f"Audited confirmation runs: {result['completed_runs']}")
    print(f"Analysis: {result['metadata_path']}")


if __name__ == "__main__":
    main()
