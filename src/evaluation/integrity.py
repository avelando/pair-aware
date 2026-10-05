import numpy as np
import pandas as pd

from src.evaluation.artifacts import build_evaluation_metrics, evaluate_run_splits
from src.evaluation.predictions import COMPLETE_PAIR_PREDICTION_COLUMNS, INSTANCE_PREDICTION_COLUMNS
from src.evaluation.scope import get_evaluation_split_names


def _read_predictions(path, identifiers):
    return pd.read_csv(
        path, dtype={column: str for column in identifiers},
        keep_default_na=False, float_precision="round_trip",
    )


def _check_frame(saved, expected, key):
    if not set(expected.columns).issubset(saved.columns):
        raise ValueError("Prediction exports are missing required columns.")
    saved = saved.sort_values(key).reset_index(drop=True)
    expected = expected.sort_values(key).reset_index(drop=True)
    if len(saved) != len(expected):
        raise ValueError("Prediction exports have inconsistent row counts.")
    for column in expected.columns:
        first, second = saved[column], expected[column]
        if pd.api.types.is_float_dtype(second):
            values = first.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Prediction values must be finite: {column}.")
            matches = np.array_equal(values, second.to_numpy()) if column == "score_threshold" else np.allclose(
                values, second.to_numpy(), rtol=1e-6, atol=1e-7,
            )
        else:
            matches = first.equals(second) or first.eq(second).all()
        if not matches:
            raise ValueError(f"Prediction exports are inconsistent: {column}.")


def _check_values(saved, expected):
    if isinstance(expected, dict):
        if not isinstance(saved, dict) or not set(expected).issubset(saved):
            raise ValueError("Saved metrics are missing required fields.")
        for key, value in expected.items():
            _check_values(saved[key], value)
    elif isinstance(expected, list):
        if not isinstance(saved, list) or len(saved) != len(expected):
            raise ValueError("Saved metric arrays have inconsistent lengths.")
        for first, second in zip(saved, expected, strict=True):
            _check_values(first, second)
    elif isinstance(expected, float):
        if isinstance(saved, bool) or not np.isfinite(float(saved)) or not np.isclose(saved, expected, rtol=1e-6, atol=1e-7):
            raise ValueError("Saved metrics do not match the prediction exports.")
    elif saved != expected:
        raise ValueError("Saved metrics do not match the prediction exports.")


def verify_run_prediction_exports(paths, scope, example_counts, pair_counts, metrics, calibrated):
    frames, pairs, split_data, logits = {}, {}, {}, {}
    observed_pairs = set()
    for split in get_evaluation_split_names(scope):
        instance_key = "validation_predictions" if split == "validation" else "predictions"
        pair_key = "validation_pair_predictions" if split == "validation" else "pair_predictions"
        frame = _read_predictions(paths[instance_key], ("id", "pair_id", "suffix", "text"))
        pair_frame = _read_predictions(paths[pair_key], ("pair_id", "pun_id", "non_pun_id"))
        if not set(INSTANCE_PREDICTION_COLUMNS).issubset(frame) or not set(COMPLETE_PAIR_PREDICTION_COLUMNS).issubset(pair_frame):
            raise ValueError("Prediction exports are missing required columns.")
        if len(frame) != example_counts[split] or len(pair_frame) != pair_counts[split]:
            raise ValueError("Prediction exports have unexpected row counts.")
        for data, key in ((frame, "id"), (pair_frame, "pair_id")):
            if data[key].eq("").any() or data[key].isna().any() or data[key].duplicated().any():
                raise ValueError("Prediction identifiers must be nonempty and unique.")
        if not frame["true_label"].isin((0, 1)).all() or not frame["predicted_label"].isin((0, 1)).all():
            raise ValueError("Prediction labels must be binary.")
        if not frame["text"].str.strip().ne("").all():
            raise ValueError("Prediction texts must be nonempty.")
        for data in (frame, pair_frame):
            columns = [column for column in data if "probability" in column]
            if not data[columns].ge(0).all().all() or not data[columns].le(1).all().all():
                raise ValueError("Prediction probabilities must lie in [0, 1].")
        split_pairs = set(frame["pair_id"])
        if observed_pairs.intersection(split_pairs):
            raise ValueError("Validation and test prediction pairs must be disjoint.")
        observed_pairs.update(split_pairs)
        frames[split], pairs[split] = frame, pair_frame
        split_data[split] = frame[["id", "text", "true_label"]].rename(columns={"true_label": "label"})
        logits[split] = frame[["logit_non_pun", "logit_pun"]].to_numpy(dtype=float)
        if not np.isfinite(logits[split]).all() or (np.abs(logits[split]) > np.finfo(np.float32).max).any():
            raise ValueError("Prediction logits must be finite and representable in float32.")
    results = evaluate_run_splits(
        split_data, lambda split, frame: (logits[split], frame["label"].to_numpy()), scope,
    )
    for split, result in results.items():
        expected_instances, expected_pairs = result["predictions"], result["pair_predictions"]
        if not calibrated:
            expected_instances = expected_instances.loc[:, INSTANCE_PREDICTION_COLUMNS]
            expected_pairs = expected_pairs.loc[:, COMPLETE_PAIR_PREDICTION_COLUMNS]
        _check_frame(frames[split], expected_instances, "id")
        _check_frame(pairs[split], expected_pairs, "pair_id")
    rebuilt_metrics = build_evaluation_metrics(results)
    if calibrated:
        if metrics.get("validation_threshold", {}).get("score_threshold") != rebuilt_metrics["validation_threshold"]["score_threshold"]:
            raise ValueError("The saved threshold does not match validation-only fitting.")
        _check_values(metrics, rebuilt_metrics)
    else:
        for split in results:
            for section in ("instance", "pair"):
                key = f"{split}_{section}"
                expected = {name: rebuilt_metrics[key][name] for name in metrics[key]}
                _check_values(metrics[key], expected)
