import random

import pandas as pd

from src.config import PAIRING_STRATEGIES
from src.data.validation import parse_pair_id


PAIR_COLUMNS = (
    "h_pair_id",
    "n_pair_id",
    "h_id",
    "n_id",
    "h_text",
    "n_text",
    "h_label",
    "n_label",
    "is_true_pair",
)


def _prepare_instances(split_df):
    data = split_df.copy()

    parsed_ids = data["id"].map(parse_pair_id)

    data["pair_id"] = parsed_ids.map(lambda value: value[0])
    data["suffix"] = parsed_ids.map(lambda value: value[1])

    return data


def _random_derangement(values, seed):
    values = list(values)

    if len(values) < 2:
        raise ValueError(
            "At least two pairs are required to create a derangement."
        )

    if len(set(values)) != len(values):
        raise ValueError(
            "Derangement values must be unique."
        )

    rng = random.Random(seed)

    for _ in range(10000):
        shuffled = values.copy()
        rng.shuffle(shuffled)

        if all(
            original != replacement
            for original, replacement in zip(
                values,
                shuffled,
            )
        ):
            return shuffled

    raise RuntimeError(
        "Could not generate a derangement after 10000 attempts."
    )


def build_true_pairs(split_df):
    data = _prepare_instances(split_df)

    h_instances = (
        data[data["suffix"] == "H"]
        .set_index("pair_id")
        .sort_index()
    )

    n_instances = (
        data[data["suffix"] == "N"]
        .set_index("pair_id")
        .sort_index()
    )

    h_pair_ids = h_instances.index.tolist()
    n_pair_ids = n_instances.index.tolist()

    if h_pair_ids != n_pair_ids:
        raise ValueError(
            "Pun and non-pun instances do not define the same pair IDs."
        )

    rows = []

    for pair_id in h_pair_ids:
        h_row = h_instances.loc[pair_id]
        n_row = n_instances.loc[pair_id]

        rows.append(
            {
                "h_pair_id": pair_id,
                "n_pair_id": pair_id,
                "h_id": h_row["id"],
                "n_id": n_row["id"],
                "h_text": h_row["text"],
                "n_text": n_row["text"],
                "h_label": int(h_row["label"]),
                "n_label": int(n_row["label"]),
                "is_true_pair": True,
            }
        )

    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def build_shuffled_pairs(split_df, seed):
    true_pairs = build_true_pairs(split_df)

    h_pair_ids = true_pairs["h_pair_id"].tolist()
    shuffled_n_pair_ids = _random_derangement(
        h_pair_ids,
        seed,
    )

    n_lookup = (
        true_pairs
        .set_index("n_pair_id")
        [["n_id", "n_text", "n_label"]]
        .to_dict("index")
    )

    rows = []

    for h_row, n_pair_id in zip(
        true_pairs.itertuples(index=False),
        shuffled_n_pair_ids,
    ):
        n_row = n_lookup[n_pair_id]

        rows.append(
            {
                "h_pair_id": h_row.h_pair_id,
                "n_pair_id": n_pair_id,
                "h_id": h_row.h_id,
                "n_id": n_row["n_id"],
                "h_text": h_row.h_text,
                "n_text": n_row["n_text"],
                "h_label": int(h_row.h_label),
                "n_label": int(n_row["n_label"]),
                "is_true_pair": False,
            }
        )

    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def validate_pair_table(pair_df, expected_true_pairs):
    missing_columns = set(PAIR_COLUMNS) - set(pair_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing pair columns: {sorted(missing_columns)}."
        )

    if pair_df["h_id"].duplicated().any():
        raise ValueError("Duplicated pun instances in pair table.")

    if pair_df["n_id"].duplicated().any():
        raise ValueError("Duplicated non-pun instances in pair table.")

    if not (pair_df["h_label"] == 1).all():
        raise ValueError("All H instances must have label 1.")

    if not (pair_df["n_label"] == 0).all():
        raise ValueError("All N instances must have label 0.")

    true_matches = (
        pair_df["h_pair_id"] == pair_df["n_pair_id"]
    )

    if expected_true_pairs and not true_matches.all():
        raise ValueError(
            "True-pair table contains mismatched pair IDs."
        )

    if not expected_true_pairs and true_matches.any():
        raise ValueError(
            "Shuffled-pair table contains true H/N matches."
        )

    if expected_true_pairs and not pair_df["is_true_pair"].all():
        raise ValueError(
            "True-pair table contains invalid pair flags."
        )

    if (
        not expected_true_pairs
        and pair_df["is_true_pair"].any()
    ):
        raise ValueError(
            "Shuffled-pair table contains invalid pair flags."
        )

    return {
        "pair_count": len(pair_df),
        "true_matches": int(true_matches.sum()),
        "unique_h": int(pair_df["h_id"].nunique()),
        "unique_n": int(pair_df["n_id"].nunique()),
    }


def build_pairs(split_df, strategy, seed=None):
    if strategy not in PAIRING_STRATEGIES:
        raise ValueError(
            f"Unknown pairing strategy: {strategy}"
        )

    if strategy == "true_pair":
        pairs = build_true_pairs(split_df)
        validate_pair_table(
            pairs,
            expected_true_pairs=True,
        )
        return pairs

    if seed is None:
        raise ValueError(
            "A seed is required for shuffled pairing."
        )

    pairs = build_shuffled_pairs(
        split_df,
        seed,
    )

    validate_pair_table(
        pairs,
        expected_true_pairs=False,
    )

    return pairs