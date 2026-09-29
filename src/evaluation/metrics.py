import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _to_numpy(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    return np.asarray(values)


def predictions_from_logits(logits):
    logits = _to_numpy(logits)

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            f"Expected logits with shape (n, 2), found {logits.shape}."
        )

    return np.argmax(
        logits,
        axis=1,
    ).astype(int)


def evaluate_instance_predictions(
    labels,
    predictions,
):
    labels = _to_numpy(labels).astype(int)
    predictions = _to_numpy(predictions).astype(int)

    if labels.ndim != 1:
        raise ValueError(
            f"Labels must be one-dimensional, found {labels.shape}."
        )

    if predictions.ndim != 1:
        raise ValueError(
            f"Predictions must be one-dimensional, found {predictions.shape}."
        )

    if len(labels) != len(predictions):
        raise ValueError(
            "Labels and predictions must have the same length."
        )

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be binary.")

    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("Predictions must be binary.")

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=[
            "non_pun",
            "pun",
        ],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision_non_pun": float(
            report["non_pun"]["precision"]
        ),
        "recall_non_pun": float(
            report["non_pun"]["recall"]
        ),
        "f1_non_pun": float(
            report["non_pun"]["f1-score"]
        ),
        "precision_pun": float(
            report["pun"]["precision"]
        ),
        "recall_pun": float(
            report["pun"]["recall"]
        ),
        "f1_pun": float(
            report["pun"]["f1-score"]
        ),
        "precision_macro": float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    return metrics, report, matrix


def evaluate_instance_logits(
    labels,
    logits,
):
    predictions = predictions_from_logits(
        logits
    )

    metrics, report, matrix = evaluate_instance_predictions(
        labels,
        predictions,
    )

    return (
        metrics,
        report,
        matrix,
        predictions,
    )


def evaluate_pair_predictions(pair_predictions):
    required_columns = {
        "ranking_correct",
        "ranking_tie",
        "exact_match",
        "pair_margin",
    }

    missing_columns = (
        required_columns
        - set(pair_predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing pair prediction columns: "
            f"{sorted(missing_columns)}."
        )

    if pair_predictions.empty:
        raise ValueError(
            "Pair predictions cannot be empty."
        )

    margins = pair_predictions[
        "pair_margin"
    ].astype(float)

    margin_std = (
        float(margins.std(ddof=1))
        if len(margins) > 1
        else 0.0
    )

    return {
        "pair_count": int(
            len(pair_predictions)
        ),
        "pair_ranking_accuracy": float(
            pair_predictions[
                "ranking_correct"
            ].mean()
        ),
        "pair_exact_match": float(
            pair_predictions[
                "exact_match"
            ].mean()
        ),
        "pair_ties": int(
            pair_predictions[
                "ranking_tie"
            ].sum()
        ),
        "mean_pair_margin": float(
            margins.mean()
        ),
        "median_pair_margin": float(
            margins.median()
        ),
        "std_pair_margin": margin_std,
        "min_pair_margin": float(
            margins.min()
        ),
        "max_pair_margin": float(
            margins.max()
        ),
    }