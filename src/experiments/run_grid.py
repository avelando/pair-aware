import argparse
from time import perf_counter

from src.config import (
    DEFAULT_MAX_PARALLEL,
    METHODS,
    MODEL_SEEDS,
    PAIRING_STRATEGIES,
    SPLIT_SEEDS,
)
from src.experiments.gpu_scheduler import (
    DEFAULT_GPU_POLL_SECONDS,
    DEFAULT_GPU_VRAM_SAFETY_MARGIN_GB,
    run_parallel_grid,
)
from src.experiments.run_instance_level import run_instance_level
from src.experiments.run_pair_aware import run_pair_aware


VALID_RUN_STATUSES = {
    "completed",
    "skipped",
}


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
):
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

    return [
        {
            "method": method,
            "split_seed": split_seed,
            "model_seed": model_seed,
        }
        for method in methods
        for split_seed in split_seeds
        for model_seed in model_seeds
    ]


def execute_grid_run(
    method,
    split_seed,
    model_seed,
    force=False,
):
    if method == "instance_level":
        return run_instance_level(
            split_seed=split_seed,
            model_seed=model_seed,
            force=force,
        )

    if method in PAIRING_STRATEGIES:
        return run_pair_aware(
            method=method,
            split_seed=split_seed,
            model_seed=model_seed,
            force=force,
        )

    raise ValueError(
        f"Invalid experiment method: {method}."
    )


def _build_result_record(
    task,
    result,
    duration_seconds,
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
    }

    for key in (
        "run_dir",
        "experiment_id",
        "attempt",
    ):
        if key in result:
            record[key] = result[key]

    return record


def run_grid(
    methods=METHODS,
    split_seeds=SPLIT_SEEDS,
    model_seeds=MODEL_SEEDS,
    force=False,
    fail_fast=False,
    dry_run=False,
    max_parallel=DEFAULT_MAX_PARALLEL,
    vram_per_run_gb=None,
    vram_safety_margin_gb=DEFAULT_GPU_VRAM_SAFETY_MARGIN_GB,
    poll_seconds=DEFAULT_GPU_POLL_SECONDS,
):
    tasks = build_grid(
        methods=methods,
        split_seeds=split_seeds,
        model_seeds=model_seeds,
    )

    if dry_run:
        return {
            "planned_runs": len(
                tasks
            ),
            "completed_runs": 0,
            "skipped_runs": 0,
            "failed_runs": 0,
            "duration_seconds": 0.0,
            "tasks": tasks,
            "results": [],
        }

    if max_parallel > 1:
        return run_parallel_grid(
            tasks=tasks,
            force=force,
            fail_fast=fail_fast,
            max_parallel=max_parallel,
            vram_per_run_gb=(
                vram_per_run_gb
            ),
            vram_safety_margin_gb=(
                vram_safety_margin_gb
            ),
            poll_seconds=poll_seconds,
        )

    if max_parallel != 1:
        raise ValueError(
            "max_parallel must be greater than zero."
        )

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
        method = task["method"]
        split_seed = task[
            "split_seed"
        ]
        model_seed = task[
            "model_seed"
        ]

        print(
            f"[{index}/{total_runs}] "
            f"Starting "
            f"method={method} "
            f"split_seed={split_seed} "
            f"model_seed={model_seed}",
            flush=True,
        )

        run_start_time = (
            perf_counter()
        )

        try:
            result = execute_grid_run(
                method=method,
                split_seed=split_seed,
                model_seed=model_seed,
                force=force,
            )

            duration_seconds = float(
                perf_counter()
                - run_start_time
            )

            record = _build_result_record(
                task,
                result,
                duration_seconds,
            )

        except Exception as error:
            duration_seconds = float(
                perf_counter()
                - run_start_time
            )

            record = {
                **task,
                "status": "failed",
                "duration_seconds": (
                    duration_seconds
                ),
                "error_type": type(
                    error
                ).__name__,
                "error_message": str(
                    error
                ),
            }

            results.append(
                record
            )

            print(
                f"[{index}/{total_runs}] "
                f"Failed "
                f"method={method} "
                f"split_seed={split_seed} "
                f"model_seed={model_seed} "
                f"error={type(error).__name__}: "
                f"{error}",
                flush=True,
            )

            if fail_fast:
                raise

            continue

        results.append(
            record
        )

        print(
            f"[{index}/{total_runs}] "
            f"Finished "
            f"status={record['status']} "
            f"duration={duration_seconds:.2f}s",
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
        "duration_seconds": (
            duration_seconds
        ),
        "tasks": tasks,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser()

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
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
    )

    parser.add_argument(
        "--vram-per-run-gb",
        type=float,
    )

    parser.add_argument(
        "--vram-safety-margin-gb",
        type=float,
        default=DEFAULT_GPU_VRAM_SAFETY_MARGIN_GB,
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_GPU_POLL_SECONDS,
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
        max_parallel=args.max_parallel,
        vram_per_run_gb=(
            args.vram_per_run_gb
        ),
        vram_safety_margin_gb=(
            args.vram_safety_margin_gb
        ),
        poll_seconds=args.poll_seconds,
    )
    
    print(
        f"Planned runs: "
        f"{result['planned_runs']}"
    )

    if args.dry_run:
        for index, task in enumerate(
            result["tasks"],
            start=1,
        ):
            print(
                f"[{index}/"
                f"{result['planned_runs']}] "
                f"method={task['method']} "
                f"split_seed="
                f"{task['split_seed']} "
                f"model_seed="
                f"{task['model_seed']}"
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
        f"Duration seconds: "
        f"{result['duration_seconds']:.2f}"
    )

    if result[
        "failed_runs"
    ]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()