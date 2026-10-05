import argparse

from src.config import PAIR_LOSS_WEIGHTS, SCREENING_MODEL_SEEDS, SPLIT_SEEDS
from src.experiments.retry import DEFAULT_MAX_RETRIES
from src.experiments.run_grid import build_grid, print_grid_result, run_grid
from src.results.selection import generate_screening_selection


def build_screening_grid():
    return build_grid(
        methods=("true_pair",),
        split_seeds=SPLIT_SEEDS,
        model_seeds=SCREENING_MODEL_SEEDS,
        pair_loss_weights=PAIR_LOSS_WEIGHTS,
        evaluation_scope="validation",
    )


def run_screening(force=False, fail_fast=False, dry_run=False, max_retries=DEFAULT_MAX_RETRIES):
    result = run_grid(
        methods=("true_pair",),
        split_seeds=SPLIT_SEEDS,
        model_seeds=SCREENING_MODEL_SEEDS,
        pair_loss_weights=PAIR_LOSS_WEIGHTS,
        evaluation_scope="validation",
        force=force,
        fail_fast=fail_fast,
        dry_run=dry_run,
        max_retries=max_retries,
    )
    if not dry_run and result["failed_runs"] == 0:
        result["selection"] = generate_screening_selection()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--select-only", action="store_true")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()
    if args.select_only and (args.force or args.fail_fast or args.max_retries != DEFAULT_MAX_RETRIES):
        parser.error("Training options cannot be used with --select-only.")
    return args


def main():
    args = parse_args()
    if args.select_only:
        selection = generate_screening_selection()
        print(f"Selected pair loss weight: {selection['selected_pair_loss_weight']}")
        print(f"Selection: {selection['selection_path']}")
        return
    result = run_screening(
        force=args.force,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )
    print_grid_result(result, dry_run=args.dry_run)
    if "selection" in result:
        print(f"Selected pair loss weight: {result['selection']['selected_pair_loss_weight']}")
        print(f"Selection: {result['selection']['selection_path']}")


if __name__ == "__main__":
    main()
