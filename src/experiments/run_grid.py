import argparse
from time import perf_counter

from src.config import (
    METHODS,
    MODEL_SEEDS,
    PAIRING_STRATEGIES,
    PAIR_LOSS_WEIGHT,
    SPLIT_SEEDS,
)
from src.evaluation.scope import EVALUATION_SCOPES, validate_evaluation_scope
from src.experiments.retry import (
    DEFAULT_MAX_RETRIES,
    validate_max_retries,
)
from src.pair_loss import resolve_pair_loss_weight, validate_pair_loss_weight


VALID_RUN_STATUSES = {
    "completed",
    "skipped",
}


def run_instance_level(**arguments):
    from src.experiments.run_instance_level import run_instance_level as execute
    return execute(**arguments)


def run_pair_aware(**arguments):
    from src.experiments.run_pair_aware import run_pair_aware as execute
    return execute(**arguments)


def format_grid_task(task):
    return (
        f"method={task['method']} "
        f"split_seed={task['split_seed']} "
        f"model_seed={task['model_seed']} "
        f"pair_loss_weight={task['pair_loss_weight']} "
        f"evaluation_scope={task['evaluation_scope']}"
    )


def _normalize_selection(
    values,
    allowed_values,
    name,
):
    values = tuple(values)

    if not values:
        raise ValueError(
            f"{name} cannot be empty."
        )

    invalid_values = [
        value
        for value in values
        if value not in allowed_values
    ]

    if invalid_values:
        raise ValueError(
            f"Invalid {name}: {invalid_values}."
        )

    if len(set(values)) != len(values):
        raise ValueError(
            f"{name} cannot contain duplicates."
        )

    return values


def build_grid(
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
    model_seeds=MODEL_SEEDS,
    pair_loss_weights=None,
    evaluation_scope="full",
):
    evaluation_scope = validate_evaluation_scope(evaluation_scope)
    methods = _normalize_selection(
        methods,
        METHODS,
        "methods",
    )

    split_seeds = _normalize_selection(
        split_seeds,
        SPLIT_SEEDS,
        "split seeds",
    )

    model_seeds = _normalize_selection(
        model_seeds,
        MODEL_SEEDS,
        "model seeds",
    )

    if pair_loss_weights is not None and not any(method in PAIRING_STRATEGIES for method in methods):
        raise ValueError("Pair loss weights require at least one pair-aware method.")
    weights = tuple(validate_pair_loss_weight(value) for value in (
        (PAIR_LOSS_WEIGHT,) if pair_loss_weights is None else pair_loss_weights
    ))
    if not weights or len(set(weights)) != len(weights):
        raise ValueError("Pair loss weights must be nonempty and cannot contain duplicates.")

    return [
        {
            "method": method,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "pair_loss_weight": weight,
            "evaluation_scope": evaluation_scope,
        }
        for method in methods
        for weight in ((None,) if method == "instance_level" else weights)
        for split_seed in split_seeds
        for model_seed in model_seeds
    ]


def execute_grid_run(
    method,
    split_seed,
    model_seed,
    force=False,
    pair_loss_weight=None,
    evaluation_scope="full",
):
    evaluation_scope = validate_evaluation_scope(evaluation_scope)
    pair_loss_weight = resolve_pair_loss_weight(method, pair_loss_weight)
    if method == "instance_level":
        return run_instance_level(
            split_seed=split_seed,
            model_seed=model_seed,
            force=force,
            evaluation_scope=evaluation_scope,
        )

    if method in PAIRING_STRATEGIES:
        return run_pair_aware(
            method=method,
            split_seed=split_seed,
            model_seed=model_seed,
            force=force,
            pair_loss_weight=pair_loss_weight,
            evaluation_scope=evaluation_scope,
        )

    raise ValueError(
        f"Invalid experiment method: {method}."
    )


def _build_result_record(
    task,
    result,
    duration_seconds,
    retry_count=0,
):
    status = result.get(
        "status"
    )

    if status not in VALID_RUN_STATUSES:
        raise ValueError(
            f"Invalid run status: {status}."
        )

    record = {
        **task,
        "status": status,
        "duration_seconds": float(
            duration_seconds
        ),
        "retry_count": int(
            retry_count
        ),
    }

    for key in (
        "run_dir",
        "experiment_id",
        "attempt",
    ):
        if key in result:
            record[key] = result[key]

    return record


def execute_grid_task_with_retries(
    task,
    force=False,
    max_retries=DEFAULT_MAX_RETRIES,
):
    validate_max_retries(
        max_retries
    )

    task_start_time = (
        perf_counter()
    )

    retry_count = 0

    while True:
        try:
            result = execute_grid_run(
                method=task["method"],
                split_seed=task[
                    "split_seed"
                ],
                model_seed=task[
                    "model_seed"
                ],
                force=force,
                pair_loss_weight=task["pair_loss_weight"],
                evaluation_scope=task["evaluation_scope"],
            )

            duration_seconds = float(
                perf_counter()
                - task_start_time
            )

            return (
                _build_result_record(
                    task=task,
                    result=result,
                    duration_seconds=(
                        duration_seconds
                    ),
                    retry_count=retry_count,
                ),
                None,
            )

        except Exception as error:
            if retry_count < max_retries:
                retry_count += 1

                print(
                    f"Retrying "
                    f"{format_grid_task(task)} "
                    f"retry={retry_count}/{max_retries} "
                    f"after={type(error).__name__}: "
                    f"{error}",
                    flush=True,
                )

                continue

            duration_seconds = float(
                perf_counter()
                - task_start_time
            )

            return (
                {
                    **task,
                    "status": "failed",
                    "duration_seconds": (
                        duration_seconds
                    ),
                    "retry_count": int(
                        retry_count
                    ),
                    "error_type": type(
                        error
                    ).__name__,
                    "error_message": str(
                        error
                    ),
                },
                error,
            )

        
def run_grid(
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
    model_seeds=MODEL_SEEDS,
    force=False,
    fail_fast=False,
    dry_run=False,
    max_retries=DEFAULT_MAX_RETRIES,
    pair_loss_weights=None,
    evaluation_scope="full",
):
    tasks = build_grid(
        methods=methods,
        split_seeds=split_seeds,
        model_seeds=model_seeds,
        pair_loss_weights=pair_loss_weights,
        evaluation_scope=evaluation_scope,
    )

    validate_max_retries(
        max_retries
    )

    if dry_run:
        return {
            "planned_runs": len(
                tasks
            ),
            "completed_runs": 0,
            "skipped_runs": 0,
            "failed_runs": 0,
            "retried_runs": 0,
            "retry_attempts": 0,
            "duration_seconds": 0.0,
            "tasks": tasks,
            "results": [],
        }

    grid_start_time = (
        perf_counter()
    )

    results = []
    total_runs = len(
        tasks
    )

    for index, task in enumerate(
        tasks,
        start=1,
    ):
        print(
            f"[{index}/{total_runs}] "
            f"Starting "
            f"{format_grid_task(task)}",
            flush=True,
        )

        (
            record,
            final_error,
        ) = execute_grid_task_with_retries(
            task=task,
            force=force,
            max_retries=max_retries,
        )

        results.append(
            record
        )

        if record["status"] == "failed":
            print(
                f"[{index}/{total_runs}] "
                f"Failed "
                f"{format_grid_task(task)} "
                f"retries={record['retry_count']} "
                f"error={record['error_type']}: "
                f"{record['error_message']}",
                flush=True,
            )

            if fail_fast:
                raise final_error

            continue

        print(
            f"[{index}/{total_runs}] "
            f"Finished "
            f"status={record['status']} "
            f"retries={record['retry_count']} "
            f"duration={record['duration_seconds']:.2f}s",
            flush=True,
        )

    duration_seconds = float(
        perf_counter()
        - grid_start_time
    )

    completed_runs = sum(
        result["status"]
        == "completed"
        for result in results
    )

    skipped_runs = sum(
        result["status"]
        == "skipped"
        for result in results
    )

    failed_runs = sum(
        result["status"]
        == "failed"
        for result in results
    )

    retried_runs = sum(
        result.get(
            "retry_count",
            0,
        )
        > 0
        for result in results
    )

    retry_attempts = sum(
        int(
            result.get(
                "retry_count",
                0,
            )
        )
        for result in results
    )

    return {
        "planned_runs": total_runs,
        "completed_runs": int(
            completed_runs
        ),
        "skipped_runs": int(
            skipped_runs
        ),
        "failed_runs": int(
            failed_runs
        ),
        "retried_runs": int(
            retried_runs
        ),
        "retry_attempts": int(
            retry_attempts
        ),
        "duration_seconds": (
            duration_seconds
        ),
        "tasks": tasks,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pair-loss-weights", nargs="+", type=float)
    parser.add_argument("--evaluation-scope", choices=EVALUATION_SCOPES, default="full")

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
    )

    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        choices=SPLIT_SEEDS,
        default=SPLIT_SEEDS,
    )

    parser.add_argument(
        "--model-seeds",
        nargs="+",
        type=int,
        choices=MODEL_SEEDS,
        default=MODEL_SEEDS,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = run_grid(
        methods=args.methods,
        split_seeds=args.split_seeds,
        model_seeds=args.model_seeds,
        force=args.force,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        pair_loss_weights=args.pair_loss_weights,
        evaluation_scope=args.evaluation_scope,
    )

    print_grid_result(result, dry_run=args.dry_run)


def print_grid_result(result, dry_run=False):
    print(
        f"Planned runs: "
        f"{result['planned_runs']}"
    )

    if dry_run:
        for index, task in enumerate(
            result["tasks"],
            start=1,
        ):
            print(
                f"[{index}/"
                f"{result['planned_runs']}] "
                f"{format_grid_task(task)}"
            )

        return

    print(
        f"Completed runs: "
        f"{result['completed_runs']}"
    )

    print(
        f"Skipped runs: "
        f"{result['skipped_runs']}"
    )

    print(
        f"Failed runs: "
        f"{result['failed_runs']}"
    )

    print(
        f"Retried runs: "
        f"{result['retried_runs']}"
    )

    print(
        f"Retry attempts: "
        f"{result['retry_attempts']}"
    )

    print(
        f"Duration seconds: "
        f"{result['duration_seconds']:.2f}"
    )

    if result[
        "failed_runs"
    ]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
