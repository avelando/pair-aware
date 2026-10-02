from time import perf_counter

from src.config import (
    MODEL_SEEDS,
    SKIP_COMPLETED_RUNS,
    SPLIT_SEEDS,
)
from src.experiments.fingerprint import build_experiment_fingerprint
from src.experiments.lifecycle import (
    complete_run_attempt,
    fail_run_attempt,
    run_lock,
    start_run_attempt,
)
from src.paths import get_run_dir
from src.results.io import is_run_completed


def validate_seeds(
    split_seed,
    model_seed,
):
    if split_seed not in SPLIT_SEEDS:
        raise ValueError(
            f"Invalid split seed: {split_seed}."
        )

    if model_seed not in MODEL_SEEDS:
        raise ValueError(
            f"Invalid model seed: {model_seed}."
        )


def build_skipped_result(
    method,
    split_seed,
    model_seed,
    run_dir,
    experiment_id,
):
    return {
        "status": "skipped",
        "method": method,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "run_dir": str(run_dir),
        "experiment_id": experiment_id,
    }


def run_experiment(
    method,
    split_seed,
    model_seed,
    execute,
    force=False,
):
    run_dir = get_run_dir(
        method,
        split_seed,
        model_seed,
    )

    experiment_fingerprint = (
        build_experiment_fingerprint(
            method,
            split_seed,
            model_seed,
        )
    )

    experiment_id = (
        experiment_fingerprint[
            "experiment_id"
        ]
    )

    if (
        not force
        and SKIP_COMPLETED_RUNS
        and is_run_completed(
            run_dir,
            expected_experiment_id=experiment_id,
        )
    ):
        return build_skipped_result(
            method,
            split_seed,
            model_seed,
            run_dir,
            experiment_id,
        )

    with run_lock(
        run_dir,
        experiment_id,
    ):
        if (
            not force
            and SKIP_COMPLETED_RUNS
            and is_run_completed(
                run_dir,
                expected_experiment_id=experiment_id,
            )
        ):
            return build_skipped_result(
                method,
                split_seed,
                model_seed,
                run_dir,
                experiment_id,
            )

        attempt = start_run_attempt(
            run_dir,
            experiment_id,
        )

        lifecycle_start_time = (
            perf_counter()
        )

        try:
            result = execute(
                split_seed=split_seed,
                model_seed=model_seed,
                run_dir=run_dir,
                experiment_fingerprint=(
                    experiment_fingerprint
                ),
                attempt=attempt,
            )
        except BaseException as error:
            duration_seconds = float(
                perf_counter()
                - lifecycle_start_time
            )

            fail_run_attempt(
                run_dir=run_dir,
                experiment_id=experiment_id,
                attempt=attempt,
                duration_seconds=duration_seconds,
                error=error,
            )

            raise

        duration_seconds = float(
            perf_counter()
            - lifecycle_start_time
        )

        complete_run_attempt(
            run_dir=run_dir,
            experiment_id=experiment_id,
            attempt=attempt,
            duration_seconds=duration_seconds,
        )

        result["attempt"] = attempt

        return result