import json
from pathlib import Path

import pandas as pd

from src.config import SPLIT_NAMES
from src.paths import get_split_dir


def read_jsonl(file_path):
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error

    return pd.DataFrame(rows)


def load_split_directory(split_seed):
    split_dir = get_split_dir(split_seed)

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    split_data = {}

    for split_name in SPLIT_NAMES:
        split_path = split_dir / f"{split_name}.jsonl"
        split_data[split_name] = read_jsonl(split_path)

    return split_data


def load_metadata(split_seed):
    metadata_path = get_split_dir(split_seed) / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata format: {metadata_path}")

    return metadata