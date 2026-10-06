import argparse
from pathlib import Path

import pandas as pd

from src.config import PAIR_LOSS_WEIGHT
from src.data.loading import read_jsonl
from src.evaluation.scope import get_evaluation_split_names
from src.experiments.fingerprint import build_experiment_fingerprint
from src.experiments.lifecycle import run_lock
from src.experiments.run_confirmation import build_confirmation_grid
from src.experiments.run_references import build_reference_grid
from src.experiments.run_screening import build_screening_grid
from src.paths import RESULTS_ROOT, get_run_dir, get_split_dir
from src.results.confirmation import DEFAULT_SELECTION_PATH, load_screening_selection
from src.results.io import get_run_artifact_paths, is_run_completed, write_dataframe, write_json
from src.results.selection import _canonical, _hash_file, _read_json


COLLECTION_ROOT = RESULTS_ROOT / "collection"
TASK_KEYS = ("method", "split_seed", "model_seed", "pair_loss_weight", "evaluation_scope")


def build_collection_plan(weight, include_references=False):
    grids = {"screening": build_screening_grid(), "confirmation": build_confirmation_grid(weight)}
    if include_references:
        grids["references"] = build_reference_grid()
    unique = {}
    for phase, tasks in grids.items():
        for task in tasks:
            key = tuple(task[name] for name in TASK_KEYS)
            if key not in unique:
                unique[key] = {**task, "phases": []}
            unique[key]["phases"].append(phase)
    return list(unique.values())


def _flatten(mapping, prefix=""):
    values = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            values.update(_flatten(value, name + "_"))
        elif value is None or isinstance(value, (str, int, float, bool)):
            values[name] = value
    return values


def _verify_source_predictions(paths, seed, scope, cache):
    for split in get_evaluation_split_names(scope):
        key = (seed, split)
        if key not in cache:
            source = read_jsonl(get_split_dir(seed) / f"{split}.jsonl")
            if not {"id", "text", "label"}.issubset(source):
                raise ValueError("Source splits are missing required columns.")
            if source["id"].duplicated().any() or not source["id"].map(lambda value: isinstance(value, str) and bool(value)).all():
                raise ValueError("Source split IDs must be unique nonempty strings.")
            if not source["label"].isin((0, 1)).all():
                raise ValueError("Source split labels must be binary.")
            cache[key] = source[["id", "text", "label"]].sort_values("id").reset_index(drop=True)
        artifact = "validation_predictions" if split == "validation" else "predictions"
        saved = pd.read_csv(
            paths[artifact], usecols=["id", "text", "true_label"], dtype={"id": str, "text": str},
            keep_default_na=False, float_precision="round_trip",
        ).rename(columns={"true_label": "label"})
        try:
            pd.testing.assert_frame_equal(
                saved[["id", "text", "label"]].sort_values("id").reset_index(drop=True),
                cache[key], check_dtype=False, check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(f"Prediction IDs, texts or labels do not match seed {seed}/{split}.") from error


def collect_v3_results(selection_path=DEFAULT_SELECTION_PATH, include_references=False, output_root=COLLECTION_ROOT):
    decision = load_screening_selection(selection_path)
    plan = build_collection_plan(decision["selected_pair_loss_weight"], include_references)
    output_root = Path(output_root)
    manifest_path = output_root / "collection_manifest.json"
    with run_lock(output_root, experiment_id="v3_result_collection"):
        report = {
            "collection_version": 1, "status": "running", "selection": decision, "include_references": include_references,
            "expected_runs": len(plan), "collection_source_sha256": _hash_file(__file__),
        }
        write_json(report, manifest_path)
        try:
            rows, artifacts, cache = [], [], {}
            for entry in plan:
                task = {key: entry[key] for key in TASK_KEYS}
                expected = build_experiment_fingerprint(**task)
                run_dir = get_run_dir(**task)
                paths = get_run_artifact_paths(run_dir)
                names = ["metadata", "metrics", "history", "completed", "validation_predictions", "validation_pair_predictions"]
                if task["evaluation_scope"] == "full":
                    names.extend(("predictions", "pair_predictions"))
                snapshots = [{
                    "experiment_id": expected["experiment_id"], "artifact": name,
                    "path": str(paths[name].resolve()), "size_bytes": paths[name].stat().st_size,
                    "sha256": _hash_file(paths[name]),
                } for name in names]
                if not is_run_completed(run_dir, expected_experiment_id=expected["experiment_id"]):
                    raise RuntimeError(f"Incomplete or incompatible result run: {run_dir}.")
                metadata = _read_json(paths["metadata"])
                fingerprint = metadata.get("fingerprint", {})
                if (
                    any(metadata.get(key) != task[key] for key in ("method", "split_seed", "model_seed", "evaluation_scope"))
                    or fingerprint.get("version") != expected["fingerprint_version"]
                    or any(fingerprint.get(key) != expected[key] for key in ("config_hash", "dataset_hash", "source_hash"))
                    or _canonical(fingerprint.get("config")) != _canonical(expected["config"])
                ):
                    raise RuntimeError(f"Run metadata does not match the planned experiment: {run_dir}.")
                _verify_source_predictions(paths, task["split_seed"], task["evaluation_scope"], cache)
                artifacts.extend(snapshots)
                rows.append({
                    **task, "phases": "|".join(entry["phases"]), "experiment_id": expected["experiment_id"],
                    **{key: expected[key] for key in ("config_hash", "dataset_hash", "source_hash")},
                    "run_dir": str(Path(run_dir).resolve()),
                    "checkpoint_available": paths["checkpoint"].is_file() and paths["checkpoint"].stat().st_size > 0,
                    **_flatten(expected["config"], "config_"), **_flatten(metadata.get("code", {}), "code_"),
                    **_flatten(_read_json(paths["metrics"])),
                })
            if _hash_file(selection_path) != decision["selection_sha256"]:
                raise RuntimeError("The frozen selection changed during result collection.")
            if any(_hash_file(row["path"]) != row["sha256"] for row in artifacts):
                raise RuntimeError("Run artifacts changed during result collection.")
            outputs = {}
            for name, records in (("runs", rows), ("artifacts", artifacts)):
                path = output_root / f"{name}.csv"
                write_dataframe(pd.DataFrame(records), path)
                outputs[name] = {"path": str(path.resolve()), "sha256": _hash_file(path), "row_count": len(records)}
            report.update({"status": "completed", "row_count": len(rows), "source_predictions_verified": True, "outputs": outputs})
            write_json(report, manifest_path)
        except (Exception, KeyboardInterrupt) as error:
            report.update({
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__, "error_message": str(error),
            })
            write_json(report, manifest_path)
            raise
    return {**report, "manifest_path": str(manifest_path.resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--output-root", type=Path, default=COLLECTION_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-weight", type=float)
    args = parser.parse_args()
    if args.preview_weight is not None and not args.dry_run:
        parser.error("--preview-weight requires --dry-run.")
    if args.dry_run:
        weight = PAIR_LOSS_WEIGHT if args.preview_weight is None else args.preview_weight
        print(f"Unique result rows planned: {len(build_collection_plan(weight, args.include_references))}")
        return
    try:
        result = collect_v3_results(args.selection, args.include_references, args.output_root)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    print(f"Result rows: {result['row_count']}")
    print(f"Collection manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
