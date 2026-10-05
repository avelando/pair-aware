import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PAIR_LOSS_WEIGHTS, SCREENING_MODEL_SEEDS, SPLIT_SEEDS, VALIDATION_F1_TOLERANCE
from src.evaluation.artifacts import build_evaluation_metrics, evaluate_run_splits
from src.experiments.fingerprint import build_experiment_fingerprint
from src.experiments.run_grid import build_grid
from src.paths import RESULTS_ROOT, get_run_dir
from src.results.io import get_run_artifact_paths, is_run_completed, write_dataframe, write_json


SCREENING_REPORT_DIR = RESULTS_ROOT / "screening"
NUMERICAL_TOLERANCE = 1e-12
METRIC_FIELDS = {
    "f1_macro": ("validation_instance", "f1_macro"),
    "pair_ranking_accuracy": ("validation_pair", "pair_ranking_accuracy"),
    "pair_exact_match": ("validation_pair", "pair_exact_match"),
    "mean_pair_margin": ("validation_pair", "mean_pair_margin"),
    "f1_macro_calibrated": ("validation_instance_calibrated", "f1_macro"),
    "pair_exact_match_calibrated": ("validation_pair_calibrated", "pair_exact_match"),
    "brier_score": ("validation_calibration", "brier_score"),
    "log_loss": ("validation_calibration", "log_loss"),
    "ece_pun": ("validation_calibration", "ece_pun"),
    "predicted_pun_rate": ("validation_calibration", "predicted_pun_rate"),
    "predicted_pun_rate_calibrated": ("validation_instance_calibrated", "predicted_pun_rate"),
    "score_threshold": ("validation_threshold", "score_threshold"),
    "probability_threshold": ("validation_threshold", "probability_threshold"),
}


def _screening_tasks():
    return build_grid(
        methods=("true_pair",), split_seeds=SPLIT_SEEDS, model_seeds=SCREENING_MODEL_SEEDS,
        pair_loss_weights=PAIR_LOSS_WEIGHTS, evaluation_scope="validation",
    )


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metric_values(metrics):
    return {name: float(metrics[section][field]) for name, (section, field) in METRIC_FIELDS.items()}


def _verify_prediction_exports(paths):
    predictions = pd.read_csv(paths["validation_predictions"], dtype={"id": str, "pair_id": str, "suffix": str})
    pairs = pd.read_csv(paths["validation_pair_predictions"], dtype={"pair_id": str, "pun_id": str, "non_pun_id": str})
    dataframe = predictions[["id", "text", "true_label"]].rename(columns={"true_label": "label"})
    logits = predictions[["logit_non_pun", "logit_pun"]].to_numpy()
    results = evaluate_run_splits(
        {"validation": dataframe}, lambda name, frame: (logits, frame["label"].to_numpy()),
        evaluation_scope="validation",
    )
    rebuilt = results["validation"]
    for saved, expected in ((predictions, rebuilt["predictions"]), (pairs, rebuilt["pair_predictions"])):
        try:
            pd.testing.assert_frame_equal(
                saved.loc[:, expected.columns].reset_index(drop=True), expected.reset_index(drop=True),
                check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-7,
            )
        except AssertionError as error:
            raise ValueError("Validation prediction exports are inconsistent with their logits.") from error
    return build_evaluation_metrics(results)


def collect_screening_results():
    records = []
    for task in _screening_tasks():
        run_dir = get_run_dir(**task)
        paths = get_run_artifact_paths(run_dir)
        expected = build_experiment_fingerprint(**task)
        if not paths["metadata"].is_file():
            raise RuntimeError(f"Incomplete screening run: {run_dir}.")
        metadata = _read_json(paths["metadata"])
        fingerprint = metadata.get("fingerprint", {})
        identity_matches = all(metadata.get(key) == task[key] for key in (
            "method", "split_seed", "model_seed", "evaluation_scope",
        ))
        fingerprint_matches = all(fingerprint.get(key) == expected[key] for key in (
            "config_hash", "dataset_hash", "source_hash",
        ))
        if (
            not identity_matches or not fingerprint_matches
            or fingerprint.get("version") != expected["fingerprint_version"]
            or _canonical(fingerprint.get("config")) != _canonical(expected["config"])
            or not is_run_completed(run_dir, expected_experiment_id=expected["experiment_id"])
        ):
            raise RuntimeError(f"Incomplete or incompatible screening run: {run_dir}.")
        metrics = _read_json(paths["metrics"])
        values = _metric_values(metrics)
        rebuilt_values = _metric_values(_verify_prediction_exports(paths))
        if not all(np.isclose(value, rebuilt_values[name], rtol=1e-6, atol=1e-7) for name, value in values.items()):
            raise ValueError(f"Validation metrics do not match the prediction exports: {run_dir}.")
        records.append({
            **task, **values, "run_dir": str(run_dir), "experiment_id": expected["experiment_id"],
            **{key: expected[key] for key in ("config_hash", "dataset_hash", "source_hash")},
            **{f"{name}_sha256": _hash_file(paths[name]) for name in (
                "metadata", "metrics", "validation_predictions", "validation_pair_predictions",
            )},
        })
    return pd.DataFrame(records)


def build_screening_summaries(records):
    keys = ["pair_loss_weight", "split_seed", "model_seed"]
    expected = {tuple(task[key] for key in keys) for task in _screening_tasks()}
    required = {"method", "evaluation_scope", *keys, *METRIC_FIELDS}
    if not required.issubset(records.columns):
        raise ValueError("Screening records are missing required columns.")
    actual = set(records[keys].itertuples(index=False, name=None))
    if len(records) != len(expected) or actual != expected or records.duplicated(keys).any():
        raise ValueError("Selection requires the complete unique screening grid.")
    if not records["method"].eq("true_pair").all() or not records["evaluation_scope"].eq("validation").all():
        raise ValueError("Selection requires validation-only true-pair runs.")
    numeric = records[list(METRIC_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Screening metrics must be finite.")
    bounded = set(METRIC_FIELDS) - {"mean_pair_margin", "score_threshold", "log_loss"}
    if not records[list(bounded)].ge(0).all().all() or not records[list(bounded)].le(1).all().all():
        raise ValueError("Probability and classification metrics must lie in [0, 1].")
    if not records["log_loss"].ge(0).all():
        raise ValueError("Log loss must be nonnegative.")
    grouped = records.groupby(["pair_loss_weight", "split_seed"], sort=True)
    by_split = pd.concat((
        grouped.size().rename("run_count"),
        grouped[list(METRIC_FIELDS)].mean().add_suffix("_mean"),
        grouped[list(METRIC_FIELDS)].std(ddof=1).add_suffix("_std_model_seeds"),
    ), axis=1).reset_index()
    rows = []
    for weight, group in by_split.groupby("pair_loss_weight", sort=True):
        row = {"pair_loss_weight": float(weight), "split_count": len(group), "run_count": int(group["run_count"].sum())}
        for name in METRIC_FIELDS:
            row[f"{name}_mean"] = float(group[f"{name}_mean"].mean())
            row[f"{name}_std_between_splits"] = float(group[f"{name}_mean"].std(ddof=1))
            row[f"{name}_mean_std_model_seeds"] = float(group[f"{name}_std_model_seeds"].mean())
        rows.append(row)
    return by_split, pd.DataFrame(rows)


def select_pair_loss_weight(summary):
    summary = summary.copy()
    columns = ["f1_macro_mean", "pair_ranking_accuracy_mean"]
    if (
        len(summary) != len(PAIR_LOSS_WEIGHTS)
        or set(summary["pair_loss_weight"]) != set(PAIR_LOSS_WEIGHTS)
        or not np.isfinite(summary[columns].to_numpy(dtype=float)).all()
        or not summary[columns].ge(0).all().all() or not summary[columns].le(1).all().all()
    ):
        raise ValueError("Selection requires a complete finite summary for every screening weight.")
    best_f1 = float(summary["f1_macro_mean"].max())
    summary["f1_gap_from_best"] = best_f1 - summary["f1_macro_mean"]
    summary["eligible"] = summary["f1_gap_from_best"] <= VALIDATION_F1_TOLERANCE + NUMERICAL_TOLERANCE
    candidates = summary.loc[summary["eligible"]]
    best_ranking = float(candidates["pair_ranking_accuracy_mean"].max())
    candidates = candidates.loc[np.isclose(candidates["pair_ranking_accuracy_mean"], best_ranking, rtol=0, atol=NUMERICAL_TOLERANCE)]
    best_candidate_f1 = float(candidates["f1_macro_mean"].max())
    candidates = candidates.loc[np.isclose(candidates["f1_macro_mean"], best_candidate_f1, rtol=0, atol=NUMERICAL_TOLERANCE)]
    selected = float(candidates["pair_loss_weight"].min())
    summary["selected"] = summary["pair_loss_weight"].eq(selected)
    return selected, summary


def generate_screening_selection(output_dir=SCREENING_REPORT_DIR):
    records = collect_screening_results()
    by_split, by_weight = build_screening_summaries(records)
    selected, by_weight = select_pair_loss_weight(by_weight)
    output_dir = Path(output_dir)
    tables = {
        "runs": (records, output_dir / "screening_runs.csv"),
        "by_split": (by_split, output_dir / "screening_by_split.csv"),
        "by_lambda": (by_weight, output_dir / "screening_by_lambda.csv"),
    }
    for dataframe, path in tables.values():
        write_dataframe(dataframe, path)
    rule = {
        "selection_split": "validation", "classification_decision": "default",
        "constraint_metric": "f1_macro", "f1_tolerance": VALIDATION_F1_TOLERANCE,
        "objective": "pair_ranking_accuracy",
        "tie_break": ["higher_f1_macro", "lower_pair_loss_weight"],
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "aggregation": "mean_over_model_seeds_then_equal_mean_over_split_seeds",
    }
    selection = {
        "status": "selected", "selected_pair_loss_weight": selected,
        "expected_runs": len(_screening_tasks()), "completed_runs": len(records),
        "split_seeds": list(SPLIT_SEEDS), "model_seeds": list(SCREENING_MODEL_SEEDS),
        "pair_loss_weights": list(PAIR_LOSS_WEIGHTS), "rule": rule,
        "selection_source_sha256": _hash_file(__file__),
        "best_validation_f1_macro": float(by_weight["f1_macro_mean"].max()),
        "selected_validation_metrics": by_weight.loc[by_weight["selected"]].iloc[0].to_dict(),
        "eligible_pair_loss_weights": by_weight.loc[by_weight["eligible"], "pair_loss_weight"].tolist(),
        "inputs": records.drop(columns=list(METRIC_FIELDS)).to_dict("records"),
        "outputs": {name: {"path": str(path), "sha256": _hash_file(path)} for name, (_, path) in tables.items()},
    }
    selection_path = output_dir / "selection.json"
    write_json(selection, selection_path)
    return {"selected_pair_loss_weight": selected, "selection_path": str(selection_path)}
