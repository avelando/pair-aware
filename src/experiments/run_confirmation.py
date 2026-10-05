import argparse
import hashlib
from pathlib import Path

from src.config import MODEL_SEEDS, SPLIT_SEEDS
from src.experiments.fingerprint import build_experiment_fingerprint
from src.experiments.retry import DEFAULT_MAX_RETRIES, validate_max_retries
from src.experiments.run_grid import build_grid, print_grid_result, run_grid
from src.pair_loss import format_pair_loss_weight
from src.paths import RESULTS_ROOT, get_run_dir
from src.results.confirmation import DEFAULT_SELECTION_PATH, load_screening_selection, validate_confirmation_weight
from src.results.io import is_run_completed, write_json


CONFIRMATION_METHODS = ("true_pair", "shuffled_pair")
CONFIRMATION_ROOT = RESULTS_ROOT / "confirmation"


def build_confirmation_grid(pair_loss_weight):
    weight = validate_confirmation_weight(pair_loss_weight)
    return build_grid(
        methods=CONFIRMATION_METHODS, split_seeds=SPLIT_SEEDS, model_seeds=MODEL_SEEDS,
        pair_loss_weights=(weight,), evaluation_scope="full",
    )


def audit_confirmation_runs(tasks):
    missing = []
    for task in tasks:
        fingerprint = build_experiment_fingerprint(**task)
        if not is_run_completed(get_run_dir(**task), expected_experiment_id=fingerprint["experiment_id"]):
            missing.append(task)
    if missing:
        raise RuntimeError(f"Confirmation artifacts are incomplete or incompatible: {len(missing)} runs.")


def run_confirmation(
    selection_path=DEFAULT_SELECTION_PATH, force=False, fail_fast=False, dry_run=False,
    max_retries=DEFAULT_MAX_RETRIES, preview_weight=None, output_root=CONFIRMATION_ROOT,
):
    validate_max_retries(max_retries)
    if preview_weight is not None:
        if not dry_run:
            raise ValueError("A preview weight can only be used with --dry-run.")
        decision = {"selected_pair_loss_weight": validate_confirmation_weight(preview_weight), "status": "preview"}
    else:
        decision = {**load_screening_selection(selection_path), "status": "validated"}
    weight = decision["selected_pair_loss_weight"]
    tasks = build_confirmation_grid(weight)
    arguments = {
        "methods": CONFIRMATION_METHODS, "split_seeds": SPLIT_SEEDS, "model_seeds": MODEL_SEEDS,
        "pair_loss_weights": (weight,), "evaluation_scope": "full",
        "force": force, "fail_fast": fail_fast, "dry_run": dry_run, "max_retries": max_retries,
    }
    if dry_run:
        return {**run_grid(**arguments), "selection": decision}
    output_dir = Path(output_root) / f"lambda_{format_pair_loss_weight(weight)}"
    plan_path = output_dir / "confirmation_plan.json"
    result_path = output_dir / "confirmation_result.json"
    plan = {
        "phase": "confirmation", "selection": decision, "expected_runs": len(tasks),
        "methods": list(CONFIRMATION_METHODS), "split_seeds": list(SPLIT_SEEDS),
        "model_seeds": list(MODEL_SEEDS), "evaluation_scope": "full", "tasks": tasks,
        "confirmation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(plan, plan_path)
    write_json({"status": "running", "plan": str(plan_path), "selection": decision}, result_path)
    try:
        result = run_grid(**arguments)
        complete = result["failed_runs"] == 0
        if complete:
            if result["tasks"] != tasks or result["completed_runs"] + result["skipped_runs"] != len(tasks):
                raise RuntimeError("Confirmation did not execute the complete planned grid.")
            audit_confirmation_runs(tasks)
        report = {
            **result, "status": "completed" if complete else "failed", "selection": decision,
            "plan": str(plan_path), "artifacts_verified": complete,
        }
        write_json(report, result_path)
    except Exception as error:
        write_json({
            "status": "failed", "plan": str(plan_path), "selection": decision,
            "artifacts_verified": False, "error_type": type(error).__name__, "error_message": str(error),
        }, result_path)
        raise
    return {**result, "selection": decision, "confirmation_result_path": str(result_path)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--preview-weight", type=float)
    args = parser.parse_args()
    if args.preview_weight is not None and not args.dry_run:
        parser.error("--preview-weight requires --dry-run.")
    return args


def main():
    args = parse_args()
    result = run_confirmation(
        selection_path=args.selection, force=args.force, fail_fast=args.fail_fast,
        dry_run=args.dry_run, max_retries=args.max_retries, preview_weight=args.preview_weight,
    )
    print(f"Selection status: {result['selection']['status']}")
    print(f"Pair loss weight: {result['selection']['selected_pair_loss_weight']}")
    print_grid_result(result, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Confirmation: {result['confirmation_result_path']}")


if __name__ == "__main__":
    main()
