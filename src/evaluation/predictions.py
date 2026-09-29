import numpy as np
import pandas as pd

from src.data.validation import parse_pair_id


INSTANCE_PREDICTION_COLUMNS = (
    "id",
    "pair_id",
    "suffix",
    "text",
    "true_label",
    "predicted_label",
    "logit_non_pun",
    "logit_pun",
    "pun_score",
    "probability_non_pun",
    "probability_pun",
    "correct",
)

PAIR_PREDICTION_COLUMNS = (
    "pair_id",
    "pun_id",
    "non_pun_id",
    "pun_predicted_label",
    "non_pun_predicted_label",
    "pun_score",
    "non_pun_score",
    "pun_probability",
    "non_pun_probability",
    "pair_margin",
    "ranking_correct",
    "ranking_tie",
    "exact_match",
)


def _to_numpy(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    return np.asarray(values)


def _softmax(logits):
    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exponentials = np.exp(shifted)

    return exponentials / exponentials.sum(
        axis=1,
        keepdims=True,
    )


def build_instance_predictions(
    dataframe,
    logits,
):
    logits = _to_numpy(logits).astype(
        np.float32,
        copy=False,
    )

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            f"Expected logits with shape (n, 2), found {logits.shape}."
        )

    if len(dataframe) != len(logits):
        raise ValueError(
            "Dataframe and logits must have the same length."
        )

    probabilities = _softmax(logits)

    predicted_labels = np.argmax(
        logits,
        axis=1,
    ).astype(int)

    rows = []

    for index, row in enumerate(
        dataframe.itertuples(index=False)
    ):
        pair_id, suffix = parse_pair_id(
            row.id
        )

        true_label = int(
            row.label
        )

        expected_label = (
            1
            if suffix == "H"
            else 0
        )

        if true_label != expected_label:
            raise ValueError(
                f"Suffix and label are inconsistent for {row.id}."
            )

        logit_non_pun = float(
            logits[index, 0]
        )

        logit_pun = float(
            logits[index, 1]
        )

        pun_score = (
            logit_pun
            - logit_non_pun
        )

        predicted_label = int(
            predicted_labels[index]
        )

        rows.append(
            {
                "id": row.id,
                "pair_id": pair_id,
                "suffix": suffix,
                "text": row.text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "logit_non_pun": logit_non_pun,
                "logit_pun": logit_pun,
                "pun_score": float(
                    pun_score
                ),
                "probability_non_pun": float(
                    probabilities[index, 0]
                ),
                "probability_pun": float(
                    probabilities[index, 1]
                ),
                "correct": bool(
                    predicted_label
                    == true_label
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=INSTANCE_PREDICTION_COLUMNS,
    )


def build_pair_predictions(
    instance_predictions,
):
    required_columns = {
        "id",
        "pair_id",
        "suffix",
        "predicted_label",
        "pun_score",
        "probability_pun",
    }

    missing_columns = (
        required_columns
        - set(instance_predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing instance prediction columns: "
            f"{sorted(missing_columns)}."
        )

    pair_rows = []

    grouped = instance_predictions.groupby(
        "pair_id",
        sort=True,
    )

    for pair_id, pair_df in grouped:
        if len(pair_df) != 2:
            raise ValueError(
                f"Pair {pair_id} does not contain exactly two instances."
            )

        suffixes = set(
            pair_df["suffix"].tolist()
        )

        if suffixes != {"H", "N"}:
            raise ValueError(
                f"Pair {pair_id} does not contain one H and one N instance."
            )

        pun_row = pair_df.loc[
            pair_df["suffix"] == "H"
        ].iloc[0]

        non_pun_row = pair_df.loc[
            pair_df["suffix"] == "N"
        ].iloc[0]

        margin = float(
            pun_row["pun_score"]
            - non_pun_row["pun_score"]
        )

        ranking_correct = bool(
            margin > 0.0
        )

        ranking_tie = bool(
            margin == 0.0
        )

        exact_match = bool(
            pun_row["predicted_label"] == 1
            and non_pun_row["predicted_label"] == 0
        )

        pair_rows.append(
            {
                "pair_id": pair_id,
                "pun_id": pun_row["id"],
                "non_pun_id": non_pun_row["id"],
                "pun_predicted_label": int(
                    pun_row[
                        "predicted_label"
                    ]
                ),
                "non_pun_predicted_label": int(
                    non_pun_row[
                        "predicted_label"
                    ]
                ),
                "pun_score": float(
                    pun_row[
                        "pun_score"
                    ]
                ),
                "non_pun_score": float(
                    non_pun_row[
                        "pun_score"
                    ]
                ),
                "pun_probability": float(
                    pun_row[
                        "probability_pun"
                    ]
                ),
                "non_pun_probability": float(
                    non_pun_row[
                        "probability_pun"
                    ]
                ),
                "pair_margin": margin,
                "ranking_correct": ranking_correct,
                "ranking_tie": ranking_tie,
                "exact_match": exact_match,
            }
        )

    return pd.DataFrame(
        pair_rows,
        columns=PAIR_PREDICTION_COLUMNS,
    )