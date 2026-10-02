import os
import signal
import subprocess
import sys
from math import ceil
from pathlib import Path
from time import perf_counter, sleep

from src.experiments.fingerprint import build_experiment_fingerprint
from src.experiments.progress import (
    format_duration,
    format_progress_line,
    read_progress,
)
from src.experiments.retry import (
    DEFAULT_MAX_RETRIES,
    validate_max_retries,
)
from src.paths import (
    PROJECT_ROOT,
    get_log_path,
    get_run_dir,
)
from src.results.io import is_run_completed


DEFAULT_GPU_VRAM_SAFETY_MARGIN_GB = 2.0
DEFAULT_GPU_POLL_SECONDS = 5.0
DEFAULT_PROGRESS_REPORT_SECONDS = 30.0


def query_gpu_memory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    if not lines:
        raise RuntimeError(
            "nvidia-smi returned no GPU memory information."
        )

    values = [
        value.strip()
        for value in lines[0].split(",")
    ]

    if len(values) != 2:
        raise RuntimeError(
            "Unexpected nvidia-smi GPU memory output."
        )

    total_mib = float(
        values[0]
    )

    free_mib = float(
        values[1]
    )

    return {
        "total_gb": (
            total_mib / 1024.0
        ),
        "free_gb": (
            free_mib / 1024.0
        ),
    }


def validate_parallel_settings(
    max_parallel,
    vram_per_run_gb,
    vram_safety_margin_gb,
    poll_seconds,
):
    if max_parallel <= 1:
        raise ValueError(
            "max_parallel must be greater than one for parallel scheduling."
        )

    if vram_per_run_gb is None:
        raise ValueError(
            "vram_per_run_gb is required for parallel scheduling."
        )

    if vram_per_run_gb <= 0:
        raise ValueError(
            "vram_per_run_gb must be greater than zero."
        )

    if vram_safety_margin_gb < 0:
        raise ValueError(
            "vram_safety_margin_gb cannot be negative."
        )

    if poll_seconds <= 0:
        raise ValueError(
            "poll_seconds must be greater than zero."
        )


def has_vram_capacity(
    memory,
    vram_per_run_gb,
    vram_safety_margin_gb,
    reserved_by_scheduler_gb=0.0,
    baseline_free_gb=None,
):
    observed_free_gb = float(
        memory["free_gb"]
    )

    if baseline_free_gb is None:
        baseline_free_gb = observed_free_gb

    logical_free_gb = max(
        0.0,
        float(
            baseline_free_gb
        )
        - float(
            reserved_by_scheduler_gb
        ),
    )

    effective_free_gb = min(
        observed_free_gb,
        logical_free_gb,
    )

    required_free_gb = (
        float(
            vram_per_run_gb
        )
        + float(
            vram_safety_margin_gb
        )
    )

    return (
        effective_free_gb
        >= required_free_gb
    )


def build_run_command(
    task,
    force=False,
):
    method = task["method"]
    split_seed = task[
        "split_seed"
    ]
    model_seed = task[
        "model_seed"
    ]

    if method == "instance_level":
        command = [
            sys.executable,
            "-m",
            "src.experiments.run_instance_level",
            "--split-seed",
            str(
                split_seed
            ),
            "--model-seed",
            str(
                model_seed
            ),
        ]
    elif method in {
        "true_pair",
        "shuffled_pair",
    }:
        command = [
            sys.executable,
            "-m",
            "src.experiments.run_pair_aware",
            "--method",
            method,
            "--split-seed",
            str(
                split_seed
            ),
            "--model-seed",
            str(
                model_seed
            ),
        ]
    else:
        raise ValueError(
            f"Invalid experiment method: {method}."
        )

    if force:
        command.append(
            "--force"
        )

    return command


def build_task_context(
    task,
):
    experiment_fingerprint = (
        build_experiment_fingerprint(
            task["method"],
            task["split_seed"],
            task["model_seed"],
        )
    )

    return {
        **task,
        "run_dir": get_run_dir(
            task["method"],
            task["split_seed"],
            task["model_seed"],
        ),
        "log_path": get_log_path(
            task["method"],
            task["split_seed"],
            task["model_seed"],
        ),
        "experiment_id": (
            experiment_fingerprint[
                "experiment_id"
            ]
        ),
    }


def build_skipped_record(
    context,
    retry_count=0,
    duration_seconds=0.0,
):
    return {
        "method": context[
            "method"
        ],
        "split_seed": context[
            "split_seed"
        ],
        "model_seed": context[
            "model_seed"
        ],
        "status": "skipped",
        "duration_seconds": float(
            duration_seconds
        ),
        "retry_count": int(
            retry_count
        ),
        "run_dir": str(
            context[
                "run_dir"
            ]
        ),
        "experiment_id": context[
            "experiment_id"
        ],
        "log_path": str(
            context[
                "log_path"
            ]
        ),
    }


def launch_run_process(
    context,
    force=False,
):
    log_path = Path(
        context[
            "log_path"
        ]
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = log_path.open(
        "a",
        encoding="utf-8",
    )

    command = build_run_command(
        context,
        force=force,
    )

    environment = dict(
        os.environ
    )

    environment[
        "PYTHONUNBUFFERED"
    ] = "1"

    try:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=PROJECT_ROOT,
        )
    except BaseException:
        log_file.close()
        raise

    return {
        "process": process,
        "log_file": log_file,
        "context": context,
        "started_at": (
            perf_counter()
        ),
    }


def finalize_run_process(
    running_entry,
):
    process = running_entry[
        "process"
    ]

    context = running_entry[
        "context"
    ]

    log_file = running_entry[
        "log_file"
    ]

    if not log_file.closed:
        log_file.close()

    duration_seconds = float(
        perf_counter()
        - running_entry[
            "started_at"
        ]
    )

    returncode = (
        process.returncode
    )

    if (
        returncode == 0
        and is_run_completed(
            context["run_dir"],
            expected_experiment_id=(
                context[
                    "experiment_id"
                ]
            ),
        )
    ):
        return {
            "method": context[
                "method"
            ],
            "split_seed": context[
                "split_seed"
            ],
            "model_seed": context[
                "model_seed"
            ],
            "status": "completed",
            "duration_seconds": (
                duration_seconds
            ),
            "run_dir": str(
                context[
                    "run_dir"
                ]
            ),
            "experiment_id": context[
                "experiment_id"
            ],
            "log_path": str(
                context[
                    "log_path"
                ]
            ),
            "returncode": 0,
        }

    if returncode == 0:
        error_type = (
            "IncompleteRunError"
        )

        error_message = (
            "Training process exited successfully, "
            "but completed artifacts are missing or invalid."
        )
    else:
        error_type = (
            "SubprocessError"
        )

        error_message = (
            "Training process exited with "
            f"code {returncode}."
        )

    return {
        "method": context[
            "method"
        ],
        "split_seed": context[
            "split_seed"
        ],
        "model_seed": context[
            "model_seed"
        ],
        "status": "failed",
        "duration_seconds": (
            duration_seconds
        ),
        "run_dir": str(
            context[
                "run_dir"
            ]
        ),
        "experiment_id": context[
            "experiment_id"
        ],
        "log_path": str(
            context[
                "log_path"
            ]
        ),
        "returncode": returncode,
        "error_type": error_type,
        "error_message": (
            error_message
        ),
    }


def stop_run_process(
    running_entry,
    timeout_seconds=10.0,
):
    process = running_entry[
        "process"
    ]

    if process.poll() is None:
        process.send_signal(
            signal.SIGINT
        )

        try:
            process.wait(
                timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired:
            process.terminate()

            try:
                process.wait(
                    timeout=timeout_seconds
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    log_file = running_entry[
        "log_file"
    ]

    if not log_file.closed:
        log_file.close()


def summarize_parallel_results(
    tasks,
    results,
    duration_seconds,
):
    results = sorted(
        results,
        key=lambda result: result[
            "task_index"
        ],
    )

    for result in results:
        result.pop(
            "task_index",
            None,
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
        "planned_runs": len(
            tasks
        ),
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
        "duration_seconds": float(
            duration_seconds
        ),
        "tasks": tasks,
        "results": results,
    }


def build_running_progress_line(
    running_entry,
):
    context = running_entry[
        "context"
    ]

    progress = read_progress(
        context[
            "run_dir"
        ]
    )

    if progress is not None:
        return format_progress_line(
            progress
        )

    elapsed_seconds = float(
        perf_counter()
        - running_entry[
            "started_at"
        ]
    )

    return (
        f"method={context['method']} "
        f"split_seed={context['split_seed']} "
        f"model_seed={context['model_seed']} "
        f"phase=starting "
        f"epoch=0/? "
        f"elapsed="
        f"{format_duration(elapsed_seconds)} "
        f"eta=--:--:--"
    )


def report_running_progress(
    running,
):
    for entry in running:
        print(
            "[ACTIVE] "
            + build_running_progress_line(
                entry
            ),
            flush=True,
        )


def run_parallel_grid(
    tasks,
    force,
    fail_fast,
    max_parallel,
    vram_per_run_gb,
    vram_safety_margin_gb=DEFAULT_GPU_VRAM_SAFETY_MARGIN_GB,
    poll_seconds=DEFAULT_GPU_POLL_SECONDS,
    progress_report_seconds=DEFAULT_PROGRESS_REPORT_SECONDS,
    max_retries=DEFAULT_MAX_RETRIES,
):
    validate_parallel_settings(
        max_parallel=max_parallel,
        vram_per_run_gb=(
            vram_per_run_gb
        ),
        vram_safety_margin_gb=(
            vram_safety_margin_gb
        ),
        poll_seconds=poll_seconds,
    )

    validate_max_retries(
        max_retries
    )

    if progress_report_seconds <= 0:
        raise ValueError(
            "progress_report_seconds must be greater than zero."
        )

    report_every_polls = max(
        1,
        int(
            ceil(
                progress_report_seconds
                / poll_seconds
            )
        ),
    )

    pending = [
        {
            "task_index": index,
            "context": build_task_context(
                task
            ),
            "retry_count": 0,
            "accumulated_duration_seconds": 0.0,
        }
        for index, task in enumerate(
            tasks
        )
    ]

    running = []
    results = []
    baseline_free_gb = None
    poll_count = 0

    grid_start_time = (
        perf_counter()
    )

    try:
        while pending or running:
            remaining_running = []

            for entry in running:
                returncode = entry[
                    "process"
                ].poll()

                if returncode is None:
                    remaining_running.append(
                        entry
                    )

                    continue

                entry[
                    "process"
                ].returncode = (
                    returncode
                )

                result = (
                    finalize_run_process(
                        entry
                    )
                )

                retry_count = int(
                    entry.get(
                        "retry_count",
                        0,
                    )
                )

                accumulated_duration_seconds = float(
                    entry.get(
                        "accumulated_duration_seconds",
                        0.0,
                    )
                )

                result[
                    "duration_seconds"
                ] = float(
                    accumulated_duration_seconds
                    + result[
                        "duration_seconds"
                    ]
                )

                result[
                    "retry_count"
                ] = retry_count

                context = entry[
                    "context"
                ]

                if (
                    result[
                        "status"
                    ]
                    == "failed"
                    and retry_count
                    < max_retries
                ):
                    next_retry_count = (
                        retry_count
                        + 1
                    )

                    pending.append(
                        {
                            "task_index": entry[
                                "task_index"
                            ],
                            "context": context,
                            "retry_count": (
                                next_retry_count
                            ),
                            "accumulated_duration_seconds": (
                                result[
                                    "duration_seconds"
                                ]
                            ),
                        }
                    )

                    print(
                        f"[{entry['task_index'] + 1}/{len(tasks)}] "
                        f"Retrying "
                        f"method={context['method']} "
                        f"split_seed={context['split_seed']} "
                        f"model_seed={context['model_seed']} "
                        f"retry={next_retry_count}/{max_retries} "
                        f"after={result['error_type']}: "
                        f"{result['error_message']}",
                        flush=True,
                    )

                    continue

                result[
                    "task_index"
                ] = entry[
                    "task_index"
                ]

                results.append(
                    result
                )

                print(
                    f"[{entry['task_index'] + 1}/{len(tasks)}] "
                    f"Finished "
                    f"method={context['method']} "
                    f"split_seed={context['split_seed']} "
                    f"model_seed={context['model_seed']} "
                    f"status={result['status']}",
                    flush=True,
                )

                if (
                    fail_fast
                    and result[
                        "status"
                    ]
                    == "failed"
                ):
                    raise RuntimeError(
                        "Parallel grid stopped after failure: "
                        f"{result['method']} "
                        f"split_seed={result['split_seed']} "
                        f"model_seed={result['model_seed']}: "
                        f"{result['error_message']}"
                    )

            running = (
                remaining_running
            )

            if (
                pending
                and len(running)
                < max_parallel
            ):
                pending_entry = (
                    pending[0]
                )

                context = (
                    pending_entry[
                        "context"
                    ]
                )

                if (
                    not force
                    and is_run_completed(
                        context[
                            "run_dir"
                        ],
                        expected_experiment_id=(
                            context[
                                "experiment_id"
                            ]
                        ),
                    )
                ):
                    result = (
                        build_skipped_record(
                            context=context,
                            retry_count=(
                                pending_entry[
                                    "retry_count"
                                ]
                            ),
                            duration_seconds=(
                                pending_entry[
                                    "accumulated_duration_seconds"
                                ]
                            ),
                        )
                    )

                    result[
                        "task_index"
                    ] = pending_entry[
                        "task_index"
                    ]

                    results.append(
                        result
                    )

                    pending.pop(0)

                    print(
                        f"[{pending_entry['task_index'] + 1}/{len(tasks)}] "
                        f"Skipped "
                        f"method={context['method']} "
                        f"split_seed={context['split_seed']} "
                        f"model_seed={context['model_seed']}",
                        flush=True,
                    )

                    continue

                memory = (
                    query_gpu_memory()
                )

                if (
                    baseline_free_gb is None
                    or not running
                ):
                    baseline_free_gb = float(
                        memory["free_gb"]
                    )

                required_free_gb = (
                    float(
                        vram_per_run_gb
                    )
                    + float(
                        vram_safety_margin_gb
                    )
                )

                if (
                    required_free_gb
                    > memory[
                        "total_gb"
                    ]
                ):
                    raise ValueError(
                        "Requested VRAM reservation exceeds total GPU memory."
                    )

                reserved_by_scheduler_gb = (
                    len(running)
                    * float(
                        vram_per_run_gb
                    )
                )

                if has_vram_capacity(
                    memory=memory,
                    vram_per_run_gb=(
                        vram_per_run_gb
                    ),
                    vram_safety_margin_gb=(
                        vram_safety_margin_gb
                    ),
                    reserved_by_scheduler_gb=(
                        reserved_by_scheduler_gb
                    ),
                    baseline_free_gb=(
                        baseline_free_gb
                    ),
                ):
                    running_entry = (
                        launch_run_process(
                            context,
                            force=force,
                        )
                    )

                    running_entry[
                        "retry_count"
                    ] = pending_entry[
                        "retry_count"
                    ]

                    running_entry[
                        "accumulated_duration_seconds"
                    ] = pending_entry[
                        "accumulated_duration_seconds"
                    ]

                    running.append(
                        running_entry
                    )

                    pending.pop(0)

                    print(
                        f"[{running_entry['task_index'] + 1}/{len(tasks)}] "
                        f"Started "
                        f"method={context['method']} "
                        f"split_seed={context['split_seed']} "
                        f"model_seed={context['model_seed']} "
                        f"retry={pending_entry['retry_count']}/{max_retries} "
                        f"free_vram={memory['free_gb']:.2f}GB "
                        f"reserved="
                        f"{reserved_by_scheduler_gb + float(vram_per_run_gb):.2f}GB",
                        flush=True,
                    )

            if pending or running:
                sleep(
                    poll_seconds
                )

                poll_count += 1

                if (
                    running
                    and poll_count
                    % report_every_polls
                    == 0
                ):
                    report_running_progress(
                        running
                    )

    except BaseException:
        for entry in running:
            stop_run_process(
                entry
            )

        raise

    duration_seconds = float(
        perf_counter()
        - grid_start_time
    )

    return (
        summarize_parallel_results(
            tasks=tasks,
            results=results,
            duration_seconds=(
                duration_seconds
            ),
        )
    )