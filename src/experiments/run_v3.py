import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.config import PAIR_LOSS_WEIGHT
from src.experiments.lifecycle import run_lock
from src.experiments.retry import DEFAULT_MAX_RETRIES, validate_max_retries
from src.experiments.run_confirmation import build_confirmation_grid, run_confirmation
from src.experiments.run_grid import print_grid_result
from src.experiments.run_references import build_reference_grid, run_references
from src.experiments.run_screening import build_screening_grid, run_screening
from src.paths import RESULTS_ROOT
from src.results.collection import collect_v3_results
from src.results.confirmation import DEFAULT_SELECTION_PATH, load_screening_selection
from src.results.io import write_json
from src.results.export import export_v3_results
from src.results.selection import _hash_file, _read_json


PIPELINE_ROOT = RESULTS_ROOT / "pipeline"
PIPELINE_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _check_phase(result, tasks):
    if (
        result.get("failed_runs") != 0 or result.get("tasks") != tasks
        or result.get("planned_runs") != len(tasks)
        or result.get("completed_runs", 0) + result.get("skipped_runs", 0) != len(tasks)
    ):
        raise RuntimeError("The phase did not complete its entire experiment grid.")


def _same_selection(first, second):
    return all(first.get(key) == second.get(key) for key in ("selected_pair_loss_weight", "selection_sha256"))


def _count_unique_runs(*grids):
    return len({tuple(sorted(task.items())) for grid in grids for task in grid})


def run_v3(
    dry_run=False, preview_weight=None, max_retries=DEFAULT_MAX_RETRIES,
    output_root=PIPELINE_ROOT, include_references=False,
):
    validate_max_retries(max_retries)
    if preview_weight is not None and not dry_run:
        raise ValueError("A preview weight can only be used with --dry-run.")
    screening_tasks = build_screening_grid()
    reference_tasks = build_reference_grid() if include_references else []
    phase_count = 3 if include_references else 2
    if dry_run:
        weight = PAIR_LOSS_WEIGHT if preview_weight is None else preview_weight
        confirmation_tasks = build_confirmation_grid(weight)
        return {
            "status": "preview", "execution_mode": "sequential", "planned_runs": len(screening_tasks) + len(confirmation_tasks) + len(reference_tasks),
            "include_references": include_references,
            "unique_planned_runs": _count_unique_runs(screening_tasks, confirmation_tasks, reference_tasks),
            "references": {"planned_runs": len(reference_tasks), "tasks": reference_tasks},
            "preview_pair_loss_weight": confirmation_tasks[0]["pair_loss_weight"],
            "screening": {"planned_runs": len(screening_tasks), "tasks": screening_tasks},
            "confirmation": {"planned_runs": len(confirmation_tasks), "tasks": confirmation_tasks},
        }
    output_root = Path(output_root)
    status_path = output_root / "pipeline_status.json"
    with run_lock(output_root, experiment_id="v3_sequential_pipeline"):
        previous = _read_json(status_path) if status_path.is_file() else {}
        if previous and (previous.get("pipeline_version") != PIPELINE_VERSION or previous.get("execution_mode") != "sequential"):
            raise RuntimeError("The existing pipeline status has an incompatible format.")
        frozen = previous.get("selection")
        if frozen is not None and not isinstance(frozen, dict):
            raise ValueError("Invalid frozen selection in the pipeline status.")
        state = {
            "pipeline_version": PIPELINE_VERSION, "execution_mode": "sequential", "status": "running",
            "phase": "screening", "started_at_utc": _utc_now(), "max_retries": max_retries,
            "include_references": include_references,
            "pipeline_source_sha256": _hash_file(__file__), "phases": {},
        }
        if frozen is not None:
            state["selection"] = frozen
        start = perf_counter()
        write_json(state, status_path)
        try:
            phase_start = perf_counter()
            print(f"Phase 1/{phase_count}: validation-only screening", flush=True)
            if frozen is not None and not DEFAULT_SELECTION_PATH.is_file():
                raise RuntimeError("The frozen selection file is missing; restore it before resuming.")
            if DEFAULT_SELECTION_PATH.is_file():
                decision = load_screening_selection(DEFAULT_SELECTION_PATH)
                if frozen is not None and not _same_selection(frozen, decision):
                    raise RuntimeError("The saved selection changed after it was frozen for this pipeline.")
                screening = {
                    "planned_runs": len(screening_tasks), "tasks": screening_tasks,
                    "completed_runs": 0, "skipped_runs": len(screening_tasks), "failed_runs": 0,
                }
            else:
                screening = run_screening(force=False, fail_fast=True, dry_run=False, max_retries=max_retries)
                _check_phase(screening, screening_tasks)
                decision = load_screening_selection(DEFAULT_SELECTION_PATH)
            state["selection"] = decision
            state["phases"]["screening"] = {
                "status": "completed", "planned_runs": len(screening_tasks),
                "completed_runs": screening["completed_runs"], "skipped_runs": screening["skipped_runs"],
                "duration_seconds": perf_counter() - phase_start,
            }
            state["phase"] = "confirmation"
            write_json(state, status_path)
            print(f"Selected pair loss weight: {decision['selected_pair_loss_weight']}", flush=True)
            print(f"Phase 2/{phase_count}: true-pair and shuffled-pair confirmation", flush=True)
            phase_start = perf_counter()
            confirmation_tasks = build_confirmation_grid(decision["selected_pair_loss_weight"])
            confirmation = run_confirmation(
                selection_path=DEFAULT_SELECTION_PATH, force=False, fail_fast=True,
                dry_run=False, max_retries=max_retries,
            )
            _check_phase(confirmation, confirmation_tasks)
            if not _same_selection(decision, confirmation.get("selection", {})) or _hash_file(DEFAULT_SELECTION_PATH) != decision["selection_sha256"]:
                raise RuntimeError("The confirmation does not match the frozen selection.")
            state["phases"]["confirmation"] = {
                "status": "completed", "planned_runs": len(confirmation_tasks),
                "completed_runs": confirmation["completed_runs"], "skipped_runs": confirmation["skipped_runs"],
                "duration_seconds": perf_counter() - phase_start,
                "result_path": confirmation["confirmation_result_path"],
            }
            if include_references:
                state["phase"] = "references"
                write_json(state, status_path)
                print("Phase 3/3: instance-level and lambda-one reference collection", flush=True)
                phase_start = perf_counter()
                references = run_references(
                    selection_path=DEFAULT_SELECTION_PATH, dry_run=False,
                    fail_fast=True, max_retries=max_retries,
                )
                _check_phase(references, reference_tasks)
                if (
                    not _same_selection(decision, references.get("selection", {}))
                    or _hash_file(DEFAULT_SELECTION_PATH) != decision["selection_sha256"]
                ):
                    raise RuntimeError("The references do not match the frozen selection.")
                state["phases"]["references"] = {
                    "status": "completed", "planned_runs": len(reference_tasks),
                    "completed_runs": references["completed_runs"], "skipped_runs": references["skipped_runs"],
                    "duration_seconds": perf_counter() - phase_start,
                    "result_path": references["reference_result_path"],
                }
            state["phase"] = "collection"
            write_json(state, status_path)
            print("Collecting verified per-run results", flush=True)
            collection = collect_v3_results(selection_path=DEFAULT_SELECTION_PATH, include_references=include_references)
            if (
                collection.get("status") != "completed" or not _same_selection(decision, collection.get("selection", {}))
                or collection.get("row_count") != _count_unique_runs(screening_tasks, confirmation_tasks, reference_tasks)
                or _hash_file(DEFAULT_SELECTION_PATH) != decision["selection_sha256"]
            ):
                raise RuntimeError("The result collection does not match the complete frozen experiment grid.")
            state["collection"] = collection
            state["phase"] = "export"
            write_json(state, status_path)
            print("Exporting verified result archive", flush=True)
            exported = export_v3_results(collection_manifest=collection["manifest_path"])
            if (
                exported.get("status") != "completed" or not _same_selection(decision, exported.get("selection", {}))
                or exported.get("row_count") != collection["row_count"]
                or _hash_file(DEFAULT_SELECTION_PATH) != decision["selection_sha256"]
            ):
                raise RuntimeError("The result export does not match the frozen experiment collection.")
            state["export"] = exported
            state.update({
                "unique_planned_runs": _count_unique_runs(screening_tasks, confirmation_tasks, reference_tasks),
                "status": "completed", "phase": "completed", "completed_at_utc": _utc_now(),
                "duration_seconds": perf_counter() - start, "planned_runs": len(screening_tasks) + len(confirmation_tasks) + len(reference_tasks),
                "completed_runs": sum(phase["completed_runs"] for phase in state["phases"].values()),
                "skipped_runs": sum(phase["skipped_runs"] for phase in state["phases"].values()),
            })
            write_json(state, status_path)
        except (Exception, KeyboardInterrupt) as error:
            state.update({
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "stopped_at_utc": _utc_now(), "duration_seconds": perf_counter() - start,
                "error_type": type(error).__name__, "error_message": str(error),
            })
            write_json(state, status_path)
            raise
    return {**state, "status_path": str(status_path)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--preview-weight", type=float)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()
    if args.preview_weight is not None and not args.dry_run:
        parser.error("--preview-weight requires --dry-run.")
    return args


def main():
    args = parse_args()
    try:
        result = run_v3(
            dry_run=args.dry_run, preview_weight=args.preview_weight,
            max_retries=args.max_retries, include_references=args.include_references,
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    print(f"Execution mode: {result['execution_mode']}")
    if args.dry_run:
        print("Confirmation weight will be selected from validation; the following weight is a preview.")
        print(f"Preview pair loss weight: {result['preview_pair_loss_weight']}")
        print("Screening:")
        print_grid_result(result["screening"], dry_run=True)
        print("Confirmation:")
        print_grid_result(result["confirmation"], dry_run=True)
        if args.include_references:
            print("References:")
            print_grid_result(result["references"], dry_run=True)
        print(f"Total planned runs: {result['planned_runs']}")
        print(f"Unique planned runs: {result['unique_planned_runs']}")
    else:
        print(f"Status: {result['status']}")
        print(f"Completed runs: {result['completed_runs']}")
        print(f"Reused runs: {result['skipped_runs']}")
        print(f"Pipeline status: {result['status_path']}")
        print(f"Collection manifest: {result['collection']['manifest_path']}")
        print(f"Result archive: {result['export']['archive_path']}")
        print(f"Archive SHA-256: {result['export']['archive_sha256']}")


if __name__ == "__main__":
    main()
