import numpy as np

from src.evaluation.calibration import (
    evaluate_probability_calibration,
    predictions_at_threshold,
    select_validation_threshold,
)
from src.evaluation.metrics import evaluate_instance_predictions, evaluate_pair_predictions
from src.evaluation.predictions import build_complete_pair_predictions, build_instance_predictions
from src.evaluation.scope import get_evaluation_split_names


def evaluate_split_predictions(dataframe, logits, labels, split_name):
    if split_name not in {"validation", "test"}:
        raise ValueError(f"Invalid evaluation split: {split_name}.")

    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels)

    if dataframe.empty:
        raise ValueError("Evaluation dataframe cannot be empty.")
    if logits.shape != (len(dataframe), 2):
        raise ValueError("Evaluation logits must have shape (number of instances, 2).")
    if not np.isfinite(logits).all():
        raise ValueError("Evaluation logits must be finite.")
    if labels.ndim != 1 or not np.array_equal(labels, dataframe["label"].to_numpy()):
        raise RuntimeError(
            f"{split_name.capitalize()} labels returned by the dataloader "
            f"do not match the {split_name} dataframe order."
        )

    predictions = build_instance_predictions(dataframe, logits)
    pair_predictions = build_complete_pair_predictions(predictions)
    instance_metrics, report, matrix = evaluate_instance_predictions(
        predictions["true_label"].to_numpy(),
        predictions["predicted_label"].to_numpy(),
    )

    return {
        "instance": instance_metrics,
        "pair": evaluate_pair_predictions(pair_predictions),
        "calibration": evaluate_probability_calibration(
            predictions["true_label"].to_numpy(),
            predictions["pun_score"].to_numpy(),
            predictions["predicted_label"].to_numpy(),
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions,
        "pair_predictions": pair_predictions,
    }


def _add_threshold_evaluation(result, selection):
    threshold = selection["score_threshold"]
    predictions = result["predictions"]
    calibrated = predictions.copy()
    calibrated["predicted_label"] = predictions_at_threshold(
        calibrated["pun_score"].to_numpy(), threshold,
    )
    calibrated["correct"] = calibrated["predicted_label"] == calibrated["true_label"]
    pairs = build_complete_pair_predictions(calibrated)
    metrics, report, matrix = evaluate_instance_predictions(
        calibrated["true_label"].to_numpy(), calibrated["predicted_label"].to_numpy(),
    )
    metrics["predicted_pun_rate"] = float(calibrated["predicted_label"].mean())
    result.update({
        "instance_calibrated": metrics,
        "pair_calibrated": evaluate_pair_predictions(pairs),
        "classification_report_calibrated": report,
        "confusion_matrix_calibrated": matrix.tolist(),
    })
    predictions["score_threshold"] = threshold
    predictions["calibrated_predicted_label"] = calibrated["predicted_label"]
    predictions["calibrated_correct"] = calibrated["correct"]
    original_pairs = result["pair_predictions"]
    calibrated_pairs = pairs.set_index("pair_id")
    original_pairs["score_threshold"] = threshold
    for source, target in (
        ("pun_predicted_label", "pun_calibrated_predicted_label"),
        ("non_pun_predicted_label", "non_pun_calibrated_predicted_label"),
        ("exact_match", "calibrated_exact_match"),
    ):
        original_pairs[target] = original_pairs["pair_id"].map(calibrated_pairs[source])


def evaluate_run_splits(split_data, predict_split, evaluation_scope="full"):
    results = {}
    selection = None
    for split_name in get_evaluation_split_names(evaluation_scope):
        dataframe = split_data[split_name]
        logits, labels = predict_split(split_name, dataframe)
        results[split_name] = evaluate_split_predictions(dataframe, logits, labels, split_name)
        result = results[split_name]
        if split_name == "validation":
            predictions = result["predictions"]
            selection = select_validation_threshold(
                predictions["true_label"].to_numpy(), predictions["pun_score"].to_numpy(),
            )
            result["threshold"] = selection
        _add_threshold_evaluation(result, selection)
    return results


def build_evaluation_metrics(results):
    metrics = {
        f"{split_name}_{section}": result[section]
        for split_name, result in results.items()
        for section in (
            "instance", "pair", "classification_report", "confusion_matrix", "calibration",
            "instance_calibrated", "pair_calibrated",
            "classification_report_calibrated", "confusion_matrix_calibrated",
        )
    }
    metrics["validation_threshold"] = results["validation"]["threshold"]
    return metrics
