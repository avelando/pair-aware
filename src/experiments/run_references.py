import argparse
from pathlib import Path

from src.config import MODEL_SEEDS, SPLIT_SEEDS
from src.experiments.fingerprint import build_experiment_fingerprint
from src.experiments.lifecycle import run_lock
from src.experiments.retry import DEFAULT_MAX_RETRIES, validate_max_retries
from src.experiments.run_grid import build_grid, print_grid_result, run_grid
from src.paths import RESULTS_ROOT, get_run_dir
from src.results.confirmation import DEFAULT_SELECTION_PATH, load_screening_selection
from src.results.io import is_run_completed, write_json
from src.results.selection import _hash_file


REFERENCE_METHODS = ("instance_level", "true_pair", "shuffled_pair")
REFERENCE_PAIR_LOSS_WEIGHT = 1.0
REFERENCE_ROOT = RESULTS_ROOT / "references"


def build_reference_grid():
    return build_grid(
        methods=REFERENCE_METHODS, split_seeds=SPLIT_SEEDS, model_seeds=MODEL_SEEDS,
        pair_loss_weights=(REFERENCE_PAIR_LOSS_WEIGHT,), evaluation_scope="full",
    )


def audit_reference_runs(tasks):
    records = []
    for task in tasks:
        fingerprint = build_experiment_fingerprint(**task)
        run_dir = get_run_dir(**task)
        if not is_run_completed(run_dir, expected_experiment_id=fingerprint["experiment_id"]):
            raise RuntimeError(f"Reference artifacts are incomplete or incompatible: {run_dir}.")
        records.append({**task, "run_dir": str(run_dir), "experiment_id": fingerprint["experiment_id"]})
    return records


def run_references(
    selection_path=DEFAULT_SELECTION_PATH, dry_run=False, fail_fast=False,
    max_retries=DEFAULT_MAX_RETRIES, output_root=REFERENCE_ROOT,
):
    validate_max_retries(max_retries)
    tasks = build_reference_grid()
    arguments = {
        "methods": REFERENCE_METHODS, "split_seeds": SPLIT_SEEDS, "model_seeds": MODEL_SEEDS,
        "pair_loss_weights": (REFERENCE_PAIR_LOSS_WEIGHT,), "evaluation_scope": "full",
        "force": False, "fail_fast": fail_fast, "dry_run": dry_run, "max_retries": max_retries,
    }
    if dry_run:
        return run_grid(**arguments)
    output_root = Path(output_root)
    result_path = output_root / "reference_result.json"
    with run_lock(output_root, experiment_id="v3_reference_collection"):
        decision = load_screening_selection(selection_path)
        plan_path = output_root / "reference_plan.json"
        write_json({
            "phase": "references", "execution_mode": "sequential", "selection": decision,
            "expected_runs": len(tasks), "tasks": tasks, "source_sha256": _hash_file(__file__),
        }, plan_path)
        report = {"status": "running", "plan": str(plan_path), "selection": decision, "artifacts_verified": False}
        write_json(report, result_path)
        try:
            result = run_grid(**arguments)
            if _hash_file(selection_path) != decision["selection_sha256"]:
                raise RuntimeError("The frozen screening selection changed during reference collection.")
            complete = result["failed_runs"] == 0
            artifacts = []
            if complete:
                if (
                    result["tasks"] != tasks or result["planned_runs"] != len(tasks)
                    or result["completed_runs"] + result["skipped_runs"] != len(tasks)
                ):
                    raise RuntimeError("Reference collection did not complete its entire experiment grid.")
                artifacts = audit_reference_runs(tasks)
            report.update({
                **result, "status": "completed" if complete else "failed",
                "artifacts_verified": complete, "run_artifacts": artifacts,
            })
            write_json(report, result_path)
        except (Exception, KeyboardInterrupt) as error:
            report.update({
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "artifacts_verified": False, "error_type": type(error).__name__, "error_message": str(error),
            })
            write_json(report, result_path)
            raise
    return {**result, "selection": decision, "reference_result_path": str(result_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()
    try:
        result = run_references(
            selection_path=args.selection, dry_run=args.dry_run,
            fail_fast=args.fail_fast, max_retries=args.max_retries,
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    print_grid_result(result, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Reference results: {result['reference_result_path']}")


if __name__ == "__main__":
    main()
