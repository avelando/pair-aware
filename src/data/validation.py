from collections import Counter, defaultdict

from src.config import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_PAIR_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    SPLIT_NAMES,
)
from src.evaluation.scope import validate_split_names


REQUIRED_COLUMNS = {
    "id",
    "text",
    "label",
    "tokens",
    "labels",
}


def parse_pair_id(record_id):
    if not isinstance(record_id, str):
        raise ValueError("Record ID must be a string.")

    parts = record_id.rsplit(".", 1)

    if len(parts) != 2:
        raise ValueError(f"Invalid record ID format: {record_id}")

    pair_id, suffix = parts

    if not pair_id or suffix not in {"H", "N"}:
        raise ValueError(f"Invalid record ID format: {record_id}")

    return pair_id, suffix


def validate_split(split_df, split_name, run_name):
    missing_columns = REQUIRED_COLUMNS - set(split_df.columns)

    if missing_columns:
        raise ValueError(
            f"{run_name}/{split_name}: missing columns "
            f"{sorted(missing_columns)}."
        )

    expected_examples = EXPECTED_SPLIT_COUNTS[split_name]

    if len(split_df) != expected_examples:
        raise ValueError(
            f"{run_name}/{split_name}: expected "
            f"{expected_examples} examples, found {len(split_df)}."
        )

    ids = split_df["id"].tolist()

    if any(not isinstance(example_id, str) for example_id in ids):
        raise ValueError(
            f"{run_name}/{split_name}: all IDs must be strings."
        )

    if split_df["id"].duplicated().any():
        raise ValueError(
            f"{run_name}/{split_name}: duplicated IDs."
        )

    labels = split_df["label"].tolist()

    if any(label not in {0, 1} for label in labels):
        raise ValueError(
            f"{run_name}/{split_name}: labels must be binary."
        )

    class_counts = dict(
        sorted(
            Counter(int(label) for label in labels).items()
        )
    )

    if class_counts != EXPECTED_CLASS_COUNTS[split_name]:
        raise ValueError(
            f"{run_name}/{split_name}: expected class distribution "
            f"{EXPECTED_CLASS_COUNTS[split_name]}, "
            f"found {class_counts}."
        )

    pair_members = defaultdict(list)

    for row_number, row in enumerate(
        split_df.itertuples(index=False),
        start=1,
    ):
        pair_id, suffix = parse_pair_id(row.id)

        if not isinstance(row.text, str) or not row.text.strip():
            raise ValueError(
                f"{run_name}/{split_name}: invalid text "
                f"at row {row_number}."
            )

        expected_label = 1 if suffix == "H" else 0

        if int(row.label) != expected_label:
            raise ValueError(
                f"{run_name}/{split_name}: suffix and label are "
                f"inconsistent for {row.id}."
            )

        if not isinstance(row.tokens, list):
            raise ValueError(
                f"{run_name}/{split_name}: tokens must be a list "
                f"for {row.id}."
            )

        if not isinstance(row.labels, list):
            raise ValueError(
                f"{run_name}/{split_name}: token labels must be a list "
                f"for {row.id}."
            )

        if len(row.tokens) != len(row.labels):
            raise ValueError(
                f"{run_name}/{split_name}: token and token-label "
                f"lengths differ for {row.id}."
            )

        if any(label not in {0, 1} for label in row.labels):
            raise ValueError(
                f"{run_name}/{split_name}: token labels must be binary "
                f"for {row.id}."
            )

        if suffix == "H" and not any(row.labels):
            raise ValueError(
                f"{run_name}/{split_name}: pun instance has no positive "
                f"token label for {row.id}."
            )

        if suffix == "N" and any(row.labels):
            raise ValueError(
                f"{run_name}/{split_name}: non-pun instance has positive "
                f"token labels for {row.id}."
            )

        pair_members[pair_id].append(suffix)

    invalid_pairs = {
        pair_id: suffixes
        for pair_id, suffixes in pair_members.items()
        if sorted(suffixes) != ["H", "N"]
    }

    if invalid_pairs:
        sample = list(invalid_pairs.items())[:10]

        raise ValueError(
            f"{run_name}/{split_name}: invalid H/N pairs found: {sample}"
        )

    expected_pairs = EXPECTED_PAIR_COUNTS[split_name]

    if len(pair_members) != expected_pairs:
        raise ValueError(
            f"{run_name}/{split_name}: expected "
            f"{expected_pairs} pairs, found {len(pair_members)}."
        )

    return {
        "ids": set(ids),
        "pair_ids": set(pair_members),
        "class_counts": class_counts,
        "example_count": len(split_df),
        "pair_count": len(pair_members),
    }


def validate_input_splits(split_data, run_name, split_names=SPLIT_NAMES):
    split_names = validate_split_names(split_names)
    missing_splits = set(split_names) - set(split_data)

    if missing_splits:
        raise ValueError(
            f"{run_name}: missing splits {sorted(missing_splits)}."
        )

    validated = {
        split_name: validate_split(
            split_data[split_name],
            split_name,
            run_name,
        )
        for split_name in split_names
    }

    all_ids = set()
    all_pair_ids = set()

    for split_name in split_names:
        split_ids = validated[split_name]["ids"]
        split_pair_ids = validated[split_name]["pair_ids"]

        if all_ids.intersection(split_ids):
            raise ValueError(
                f"{run_name}: record IDs overlap across splits."
            )

        if all_pair_ids.intersection(split_pair_ids):
            raise ValueError(
                f"{run_name}: pair IDs overlap across splits."
            )

        all_ids.update(split_ids)
        all_pair_ids.update(split_pair_ids)

    expected_total_examples = sum(
        EXPECTED_SPLIT_COUNTS[split_name] for split_name in split_names
    )

    expected_total_pairs = sum(
        EXPECTED_PAIR_COUNTS[split_name] for split_name in split_names
    )

    if len(all_ids) != expected_total_examples:
        raise ValueError(
            f"{run_name}: expected {expected_total_examples} unique "
            f"examples, found {len(all_ids)}."
        )

    if len(all_pair_ids) != expected_total_pairs:
        raise ValueError(
            f"{run_name}: expected {expected_total_pairs} unique pairs, "
            f"found {len(all_pair_ids)}."
        )

    return validated
