import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EXPECTED_PAIR_COUNTS,
    EXPECTED_SPLIT_COUNTS,
)
from src.evaluation.calibration import EVALUATION_PROTOCOL
from src.evaluation.integrity import verify_run_prediction_exports
from src.evaluation.scope import validate_evaluation_scope


RUN_ARTIFACT_NAMES = {
    "metadata": "metadata.json",
    "metrics": "metrics.json",
    "history": "history.csv",
    "predictions": "predictions.csv",
    "pair_predictions": "pair_predictions.csv",
    "validation_predictions": "validation_predictions.csv",
    "validation_pair_predictions": "validation_pair_predictions.csv",
    "checkpoint": "checkpoint.pt",
    "progress": "progress.json",
    "completed": "completed",
}

FINAL_ARTIFACT_NAMES = (
    "metadata",
    "metrics",
    "history",
    "predictions",
    "pair_predictions",
    "validation_predictions",
    "validation_pair_predictions",
    "completed",
)

VALIDATION_FINAL_ARTIFACT_NAMES = tuple(
    name for name in FINAL_ARTIFACT_NAMES if name not in {"predictions", "pair_predictions"}
) + ("checkpoint",)


def get_run_artifact_paths(run_dir):
    run_dir = Path(run_dir)

    return {
        name: run_dir / filename
        for name, filename in RUN_ARTIFACT_NAMES.items()
    }


def _temporary_path(path):
    path = Path(path)

    return path.with_suffix(
        path.suffix + ".tmp"
    )


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

        _temporary_path(
            path
        ).unlink(
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

    temporary_path = _temporary_path(
        path
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

    temporary_path = _temporary_path(
        path
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

    temporary_path = _temporary_path(
        completed_path
    )

    temporary_path.write_text(
        "completed\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        completed_path
    )


def _read_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def is_run_completed(
    run_dir,
    expected_experiment_id=None,
):
    paths = get_run_artifact_paths(
        run_dir
    )

    try:
        metadata = _read_json(paths["metadata"])
        evaluation_scope = validate_evaluation_scope(metadata.get("evaluation_scope", "full"))
        final_artifacts = FINAL_ARTIFACT_NAMES if evaluation_scope == "full" else VALIDATION_FINAL_ARTIFACT_NAMES

        for name in final_artifacts:
            path = paths[name]

            if not path.is_file():
                return False

            if path.stat().st_size == 0:
                return False

        if (
            paths["completed"]
            .read_text(
                encoding="utf-8"
            )
            .strip()
            != "completed"
        ):
            return False

        metrics = _read_json(
            paths["metrics"]
        )

        if metadata.get("status") != "completed":
            return False

        if (
            expected_experiment_id is not None
            and metadata.get("experiment_id")
            != expected_experiment_id
        ):
            return False

        required_metrics = {"validation_instance", "validation_pair"}
        if evaluation_scope == "full":
            required_metrics.update({"test_instance", "test_pair"})
        elif any(name.startswith("test_") for name in metrics):
            return False
        elif paths["predictions"].exists() or paths["pair_predictions"].exists():
            return False

        if not required_metrics.issubset(metrics):
            return False

        protocol = metadata.get("fingerprint", {}).get("config", {}).get("evaluation_protocol")
        if protocol is not None:
            if protocol != EVALUATION_PROTOCOL:
                return False
            splits = ("validation", "test") if evaluation_scope == "full" else ("validation",)
            sections = (
                "calibration", "instance_calibrated", "pair_calibrated",
                "classification_report_calibrated", "confusion_matrix_calibrated",
            )
            if not all(f"{split}_{section}" in metrics for split in splits for section in sections):
                return False
            threshold = metrics.get("validation_threshold", {})
            if threshold.get("fit_split") != "validation" or threshold.get("comparison") != ">":
                return False
            if not np.isfinite(float(threshold.get("score_threshold", float("nan")))):
                return False

        history = pd.read_csv(
            paths["history"],
            usecols=["epoch"],
        )

        if history.empty:
            return False

        verify_run_prediction_exports(
            paths, evaluation_scope, EXPECTED_SPLIT_COUNTS, EXPECTED_PAIR_COUNTS,
            metrics, calibrated=protocol is not None,
        )

    except (
        AttributeError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        OverflowError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ):
        return False

    return True


def remove_checkpoint(run_dir):
    checkpoint_path = (
        get_run_artifact_paths(
            run_dir
        )["checkpoint"]
    )

    checkpoint_path.unlink(
        missing_ok=True
    )
