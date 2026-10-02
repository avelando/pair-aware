import fcntl
import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.results.io import write_json


RUN_LOCK_FILENAME = "run.lock"
RUN_STATUS_FILENAME = "status.json"


class RunLockedError(RuntimeError):
    pass


def _utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def get_run_lock_path(run_dir):
    return Path(run_dir) / RUN_LOCK_FILENAME


def get_run_status_path(run_dir):
    return Path(run_dir) / RUN_STATUS_FILENAME


def read_run_status(run_dir):
    path = get_run_status_path(
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


def _next_attempt(
    run_dir,
    experiment_id,
):
    status = read_run_status(
        run_dir
    )

    if (
        status is None
        or status.get(
            "experiment_id"
        )
        != experiment_id
    ):
        return 1

    attempt = status.get(
        "attempt"
    )

    if (
        not isinstance(
            attempt,
            int,
        )
        or attempt < 1
    ):
        return 1

    return attempt + 1


def start_run_attempt(
    run_dir,
    experiment_id,
):
    attempt = _next_attempt(
        run_dir,
        experiment_id,
    )

    write_json(
        {
            "status": "running",
            "experiment_id": experiment_id,
            "attempt": attempt,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at_utc": _utc_now(),
        },
        get_run_status_path(
            run_dir
        ),
    )

    return attempt


def _get_attempt_status(
    run_dir,
    experiment_id,
    attempt,
):
    status = read_run_status(
        run_dir
    )

    if (
        status is not None
        and status.get(
            "experiment_id"
        )
        == experiment_id
        and status.get(
            "attempt"
        )
        == attempt
    ):
        return dict(
            status
        )

    return {
        "experiment_id": experiment_id,
        "attempt": attempt,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }


def complete_run_attempt(
    run_dir,
    experiment_id,
    attempt,
    duration_seconds,
):
    status = _get_attempt_status(
        run_dir,
        experiment_id,
        attempt,
    )

    status.update(
        {
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "duration_seconds": float(
                duration_seconds
            ),
        }
    )

    write_json(
        status,
        get_run_status_path(
            run_dir
        ),
    )


def fail_run_attempt(
    run_dir,
    experiment_id,
    attempt,
    duration_seconds,
    error,
):
    status = _get_attempt_status(
        run_dir,
        experiment_id,
        attempt,
    )

    status.update(
        {
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "duration_seconds": float(
                duration_seconds
            ),
            "error_type": type(
                error
            ).__name__,
            "error_message": str(
                error
            ),
        }
    )

    write_json(
        status,
        get_run_status_path(
            run_dir
        ),
    )


@contextmanager
def run_lock(
    run_dir,
    experiment_id,
):
    lock_path = get_run_lock_path(
        run_dir
    )

    lock_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file = lock_path.open(
        "a+",
        encoding="utf-8",
    )

    try:
        try:
            fcntl.flock(
                file.fileno(),
                fcntl.LOCK_EX
                | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RunLockedError(
                f"Run is already locked: {run_dir}"
            ) from error

        lock_metadata = {
            "active": True,
            "experiment_id": experiment_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at_utc": _utc_now(),
        }

        file.seek(0)
        file.truncate()

        json.dump(
            lock_metadata,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")
        file.flush()

        try:
            yield lock_metadata
        finally:
            lock_metadata[
                "active"
            ] = False

            lock_metadata[
                "released_at_utc"
            ] = _utc_now()

            file.seek(0)
            file.truncate()

            json.dump(
                lock_metadata,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

            file.write("\n")
            file.flush()

            fcntl.flock(
                file.fileno(),
                fcntl.LOCK_UN,
            )
    finally:
        file.close()