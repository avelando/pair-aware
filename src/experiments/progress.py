import json
from pathlib import Path
from time import perf_counter

from src.experiments.provenance import utc_now
from src.results.io import (
    get_run_artifact_paths,
    write_json,
)


def get_progress_path(run_dir):
    return get_run_artifact_paths(
        run_dir
    )["progress"]


def read_progress(run_dir):
    path = get_progress_path(
        run_dir
    )

    if not path.is_file():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            value = json.load(
                file
            )
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ):
        return None

    if not isinstance(
        value,
        dict,
    ):
        return None

    return value


def format_duration(seconds):
    if seconds is None:
        return "--:--:--"

    total_seconds = max(
        0,
        int(
            round(
                float(seconds)
            )
        ),
    )

    hours, remainder = divmod(
        total_seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def format_progress_line(progress):
    method = progress.get(
        "method",
        "unknown",
    )

    split_seed = progress.get(
        "split_seed",
        "?",
    )

    model_seed = progress.get(
        "model_seed",
        "?",
    )

    phase = progress.get(
        "phase",
        "unknown",
    )

    epoch = progress.get(
        "epoch",
        0,
    )

    total_epochs = progress.get(
        "total_epochs",
        "?",
    )

    elapsed = format_duration(
        progress.get(
            "elapsed_seconds"
        )
    )

    eta = format_duration(
        progress.get(
            "eta_seconds"
        )
    )

    line = (
        f"method={method} "
        f"split_seed={split_seed} "
        f"model_seed={model_seed} "
        f"phase={phase} "
        f"epoch={epoch}/{total_epochs} "
        f"elapsed={elapsed} "
        f"eta={eta}"
    )

    epoch_metrics = progress.get(
        "epoch_metrics"
    )

    if isinstance(
        epoch_metrics,
        dict,
    ):
        validation_f1 = (
            epoch_metrics.get(
                "validation_f1_macro"
            )
        )

        if validation_f1 is not None:
            line += (
                f" validation_f1="
                f"{float(validation_f1):.6f}"
            )

    best_score = progress.get(
        "best_validation_f1_macro"
    )

    if best_score is not None:
        line += (
            f" best_validation_f1="
            f"{float(best_score):.6f}"
        )

    return line


class RunProgressTracker:
    def __init__(
        self,
        path,
        method,
        split_seed,
        model_seed,
        experiment_id,
        attempt,
        total_epochs,
        emit_console=True,
    ):
        if total_epochs <= 0:
            raise ValueError(
                "total_epochs must be greater than zero."
            )

        self.path = Path(
            path
        )
        self.method = method
        self.split_seed = split_seed
        self.model_seed = model_seed
        self.experiment_id = experiment_id
        self.attempt = attempt
        self.total_epochs = int(
            total_epochs
        )
        self.emit_console = bool(
            emit_console
        )
        self._started_at = None
        self._last_epoch_at = None
        self.state = None

    def _ensure_started(self):
        if self.state is None:
            raise RuntimeError(
                "Progress tracker has not been started."
            )

    def _write(self):
        write_json(
            self.state,
            self.path,
        )

    def _emit(self):
        if self.emit_console:
            print(
                "[PROGRESS] "
                + format_progress_line(
                    self.state
                ),
                flush=True,
            )

    def start(self):
        now = perf_counter()

        self._started_at = now
        self._last_epoch_at = now

        self.state = {
            "status": "running",
            "phase": "training",
            "method": self.method,
            "split_seed": self.split_seed,
            "model_seed": self.model_seed,
            "experiment_id": self.experiment_id,
            "attempt": self.attempt,
            "epoch": 0,
            "total_epochs": self.total_epochs,
            "progress_fraction": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "average_epoch_seconds": None,
            "last_epoch_seconds": None,
            "best_epoch": None,
            "best_validation_f1_macro": None,
            "epochs_without_improvement": 0,
            "epoch_metrics": None,
            "started_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
        }

        self._write()
        self._emit()

        return dict(
            self.state
        )

    def update_epoch(
        self,
        epoch,
        epoch_metrics,
        best_epoch,
        best_validation_f1_macro,
        epochs_without_improvement,
    ):
        self._ensure_started()

        epoch = int(
            epoch
        )

        if (
            epoch <= self.state["epoch"]
            or epoch > self.total_epochs
        ):
            raise ValueError(
                "Epoch progress must increase and remain within total_epochs."
            )

        now = perf_counter()

        elapsed_seconds = float(
            now
            - self._started_at
        )

        last_epoch_seconds = float(
            now
            - self._last_epoch_at
        )

        average_epoch_seconds = (
            elapsed_seconds
            / epoch
        )

        eta_seconds = float(
            average_epoch_seconds
            * (
                self.total_epochs
                - epoch
            )
        )

        self.state.update(
            {
                "status": "running",
                "phase": "training",
                "epoch": epoch,
                "progress_fraction": float(
                    epoch
                    / self.total_epochs
                ),
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
                "average_epoch_seconds": float(
                    average_epoch_seconds
                ),
                "last_epoch_seconds": last_epoch_seconds,
                "best_epoch": (
                    None
                    if best_epoch is None
                    else int(best_epoch)
                ),
                "best_validation_f1_macro": float(
                    best_validation_f1_macro
                ),
                "epochs_without_improvement": int(
                    epochs_without_improvement
                ),
                "epoch_metrics": dict(
                    epoch_metrics
                ),
                "updated_at_utc": utc_now(),
            }
        )

        self._last_epoch_at = now

        self._write()
        self._emit()

        return dict(
            self.state
        )

    def set_phase(self, phase):
        self._ensure_started()

        if (
            not isinstance(
                phase,
                str,
            )
            or not phase.strip()
        ):
            raise ValueError(
                "Progress phase must be a non-empty string."
            )

        elapsed_seconds = float(
            perf_counter()
            - self._started_at
        )

        self.state.update(
            {
                "phase": phase,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": None,
                "updated_at_utc": utc_now(),
            }
        )

        self._write()
        self._emit()

        return dict(
            self.state
        )

    def complete(self):
        self._ensure_started()

        elapsed_seconds = float(
            perf_counter()
            - self._started_at
        )

        self.state.update(
            {
                "status": "completed",
                "phase": "completed",
                "progress_fraction": 1.0,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": 0.0,
                "stopped_early": bool(
                    self.state["epoch"]
                    < self.total_epochs
                ),
                "completed_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
            }
        )

        self._write()
        self._emit()

        return dict(
            self.state
        )

    def fail(self, error):
        self._ensure_started()

        elapsed_seconds = float(
            perf_counter()
            - self._started_at
        )

        self.state.update(
            {
                "status": "failed",
                "phase": "failed",
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": None,
                "error_type": type(
                    error
                ).__name__,
                "error_message": str(
                    error
                ),
                "failed_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
            }
        )

        self._write()
        self._emit()

        return dict(
            self.state
        )