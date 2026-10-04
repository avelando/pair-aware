import numpy as np

from src.evaluation.metrics import evaluate_instance_predictions, evaluate_pair_predictions
from src.evaluation.predictions import build_complete_pair_predictions, build_instance_predictions


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
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions,
        "pair_predictions": pair_predictions,
    }
