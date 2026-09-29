import json
from pathlib import Path

import pandas as pd


RUN_ARTIFACT_NAMES = {
    "metadata": "metadata.json",
    "metrics": "metrics.json",
    "history": "history.csv",
    "predictions": "predictions.csv",
    "pair_predictions": "pair_predictions.csv",
    "checkpoint": "checkpoint.pt",
    "completed": "completed",
}


def get_run_artifact_paths(run_dir):
    run_dir = Path(run_dir)

    return {
        name: run_dir / filename
        for name, filename in RUN_ARTIFACT_NAMES.items()
    }


def prepare_run_directory(run_dir):
    run_dir = Path(run_dir)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = get_run_artifact_paths(
        run_dir
    )

    for path in paths.values():
        path.unlink(
            missing_ok=True
        )

    return paths


def _json_default(value):
    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        return value.item()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        f"is not JSON serializable."
    )


def write_json(
    data,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )

        file.write("\n")

    temporary_path.replace(
        path
    )


def write_dataframe(
    dataframe,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        path
    )


def write_history(
    history,
    path,
):
    dataframe = pd.DataFrame(
        history
    )

    if dataframe.empty:
        raise ValueError(
            "Training history cannot be empty."
        )

    write_dataframe(
        dataframe,
        path,
    )


def mark_run_completed(run_dir):
    completed_path = (
        get_run_artifact_paths(
            run_dir
        )["completed"]
    )

    temporary_path = completed_path.with_name(
        completed_path.name + ".tmp"
    )

    temporary_path.write_text(
        "completed\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        completed_path
    )


def is_run_completed(run_dir):
    return (
        get_run_artifact_paths(
            run_dir
        )["completed"]
        .is_file()
    )


def remove_checkpoint(run_dir):
    checkpoint_path = (
        get_run_artifact_paths(
            run_dir
        )["checkpoint"]
    )

    checkpoint_path.unlink(
        missing_ok=True
    )