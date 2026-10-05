import numpy as np


ECE_BINS = 15
EVALUATION_PROTOCOL = {
    "version": 1,
    "threshold_fit_split": "validation",
    "threshold_objective": "f1_macro",
    "threshold_comparison": ">",
    "threshold_tie_break": "closest_to_zero_then_lower",
    "ece_bins": ECE_BINS,
    "ece_definition": "positive_class_equal_width",
}


def _validate_inputs(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or not labels.size:
        raise ValueError("Labels and scores must be nonempty aligned one-dimensional arrays.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    return labels.astype(int), scores


def probabilities_from_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    return np.exp(-np.logaddexp(0.0, -scores))


def predictions_at_threshold(scores, score_threshold):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(score_threshold):
        raise ValueError("Scores and threshold must be finite.")
    return (scores > score_threshold).astype(int)


def select_validation_threshold(labels, scores):
    labels, scores = _validate_inputs(labels, scores)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Threshold fitting requires both validation classes.")
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    candidates = np.unique(np.concatenate((
        ordered_scores,
        [0.0, np.nextafter(ordered_scores[0], -np.inf)],
    )))
    if not np.isfinite(candidates).all():
        raise ValueError("Scores are too extreme for finite threshold candidates.")
    counts = np.searchsorted(ordered_scores, candidates, side="right")
    cumulative = np.concatenate(([0], np.cumsum(labels[order])))
    positives = int(labels.sum())
    tp = positives - cumulative[counts]
    fp = len(labels) - counts - tp
    fn = positives - tp
    tn = len(labels) - positives - fp
    macro_f1 = 0.5 * (
        2.0 * tp / (2 * tp + fp + fn)
        + 2.0 * tn / (2 * tn + fp + fn)
    )
    best = np.flatnonzero(macro_f1 == macro_f1.max())
    chosen = min(best, key=lambda index: (abs(candidates[index]), candidates[index]))
    threshold = float(candidates[chosen])
    default_index = int(np.searchsorted(candidates, 0.0))
    return {
        "fit_split": "validation",
        "objective": "f1_macro",
        "comparison": ">",
        "tie_break": "closest_to_zero_then_lower",
        "score_threshold": threshold,
        "probability_threshold": float(probabilities_from_scores(threshold)),
        "candidate_count": int(len(candidates)),
        "validation_f1_macro": float(macro_f1[chosen]),
        "default_validation_f1_macro": float(macro_f1[default_index]),
    }


def evaluate_probability_calibration(labels, scores, predictions, n_bins=ECE_BINS):
    labels, scores = _validate_inputs(labels, scores)
    predictions = np.asarray(predictions)
    if predictions.shape != labels.shape or not np.isin(predictions, (0, 1)).all():
        raise ValueError("Predictions must be binary and aligned with labels.")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("The number of calibration bins must be a positive integer.")
    probabilities = probabilities_from_scores(scores)
    bin_ids = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        selected = bin_ids == index
        count = int(selected.sum())
        if not count:
            continue
        mean_probability = float(probabilities[selected].mean())
        positive_fraction = float(labels[selected].mean())
        ece += count / len(labels) * abs(mean_probability - positive_fraction)
        bins.append({
            "bin_index": index,
            "lower_bound": index / n_bins,
            "upper_bound": (index + 1) / n_bins,
            "count": count,
            "mean_probability_pun": mean_probability,
            "observed_pun_fraction": positive_fraction,
        })
    signed_scores = np.where(labels == 1, -scores, scores)
    return {
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(np.logaddexp(0.0, signed_scores).mean()),
        "predicted_pun_rate": float(predictions.mean()),
        "observed_pun_rate": float(labels.mean()),
        "mean_probability_pun": float(probabilities.mean()),
        "ece_pun": float(ece),
        "ece_bins": int(n_bins),
        "ece_definition": "positive_class_equal_width",
        "reliability_bins": bins,
    }
