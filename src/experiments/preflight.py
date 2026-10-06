import argparse
import importlib
import platform
import shutil
import sys
import tempfile
import tomllib
from importlib.metadata import version
from pathlib import Path

from src import config
from src.data.loading import load_metadata, load_split_directory
from src.data.validation import validate_input_splits
from src.experiments.fingerprint import get_config_snapshot, get_dataset_fingerprint
from src.experiments.lifecycle import run_lock
from src.paths import PROJECT_ROOT, RESULTS_ROOT, get_run_dir
from src.results.collection import build_collection_plan
from src.results.confirmation import DEFAULT_SELECTION_PATH, load_screening_selection
from src.results.io import get_run_artifact_paths, write_json
from src.results.selection import _canonical, _hash_file, _read_json


V2_ROOT = PROJECT_ROOT / "results_v2" / "bf16"
PREFLIGHT_ROOT = RESULTS_ROOT / "preflight"
CHECKPOINT_BUDGET_BYTES = int(1.5 * 1024 ** 3)
RESERVE_BYTES = 10 * 1024 ** 3


def verify_v2_protocol():
    if config.EXPERIMENT_VERSION != "v3" or config.TRAINING_PRECISION != "bf16" or not config.KEEP_CHECKPOINTS:
        raise RuntimeError("V3 requires BF16, isolated outputs and preserved checkpoints.")
    expected_paths = {
        V2_ROOT / method / f"split_{split_seed}" / f"model_seed_{model_seed}" / "metadata.json"
        for method in config.METHODS for split_seed in config.SPLIT_SEEDS for model_seed in config.MODEL_SEEDS
    }
    if set(V2_ROOT.glob("*/split_*/model_seed_*/metadata.json")) != expected_paths:
        raise RuntimeError("The definitive V2 metadata grid does not match the V3 methods and seeds.")
    records = []
    for path in sorted(expected_paths):
        metadata = _read_json(path)
        method = path.parents[2].name
        split_seed = int(path.parents[1].name.removeprefix("split_"))
        model_seed = int(path.parent.name.removeprefix("model_seed_"))
        expected = get_config_snapshot(method)
        expected.pop("evaluation_scope")
        expected.pop("evaluation_protocol")
        expected["experiment_version"] = "v2"
        if (
            metadata.get("status") != "completed" or metadata.get("method") != method
            or metadata.get("split_seed") != split_seed or metadata.get("model_seed") != model_seed
            or _canonical(metadata.get("fingerprint", {}).get("config")) != _canonical(expected)
        ):
            raise RuntimeError(f"The V3 training protocol differs from the definitive V2 run: {path}.")
        records.append({"path": str(path.resolve()), "sha256": _hash_file(path)})
    return records


def verify_validation_data():
    records = []
    for seed in config.SPLIT_SEEDS:
        metadata = load_metadata(seed)
        if metadata.get("strategy") != "pair_controlled" or metadata.get("seed") != seed or metadata.get("cross_split_pairs") != 0:
            raise ValueError(f"Invalid pair-controlled metadata for split seed {seed}.")
        frames = load_split_directory(seed, split_names=("train", "validation"))
        validated = validate_input_splits(frames, f"seed_{seed}", split_names=("train", "validation"))
        for split, result in validated.items():
            if metadata.get(f"{split}_examples") != result["example_count"] or _canonical(metadata.get("class_distribution", {}).get(split)) != _canonical(result["class_counts"]):
                raise ValueError(f"Split metadata counts do not match seed {seed}/{split}.")
        records.append({
            "split_seed": seed, "train_validation_dataset_hash": get_dataset_fingerprint(seed, split_names=("train", "validation")),
            "splits": {name: {key: value for key, value in row.items() if key not in ("ids", "pair_ids")} for name, row in validated.items()},
        })
    return records


def verify_dependencies():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The frozen environment requires Python 3.12.")
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    records = {}
    for requirement in project["dependencies"]:
        parts = requirement.split("==")
        if len(parts) != 2 or not all(parts):
            raise ValueError("Preflight requires exact dependency pins in pyproject.toml.")
        name, expected = parts
        installed = version(name)
        if installed.split("+", 1)[0] != expected:
            raise RuntimeError(f"Dependency mismatch: {name} requires {expected}, found {installed}.")
        records[name] = installed
    return {"python_version": platform.python_version(), "packages": records, "pyproject_sha256": _hash_file(PROJECT_ROOT / "pyproject.toml"), "uv_lock_sha256": _hash_file(PROJECT_ROOT / "uv.lock")}


def verify_bf16_runtime():
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("A working CUDA build of PyTorch is required.")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 support is required.")
    device = torch.device("cuda")
    with torch.enable_grad():
        left = torch.ones((16, 16), device=device, requires_grad=True)
        right = torch.ones((16, 16), device=device, requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = left @ right
            loss = output.float().mean()
        loss.backward()
        if output.dtype != torch.bfloat16 or not all(torch.isfinite(tensor).all().item() for tensor in (output, left.grad, right.grad)):
            raise RuntimeError("The CUDA BF16 forward/backward probe failed.")
    torch.cuda.synchronize(device)
    return {"device": str(device), "gpu": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda, "torch_version": torch.__version__, "native_bf16": True, "forward_backward_verified": True}


def verify_storage(include_references=False):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    decision = load_screening_selection(DEFAULT_SELECTION_PATH) if DEFAULT_SELECTION_PATH.is_file() else None
    weight = 0.25 if decision is None else decision["selected_pair_loss_weight"]
    plan = build_collection_plan(weight, include_references)
    missing = 0
    for entry in plan:
        task = {key: entry[key] for key in ("method", "split_seed", "model_seed", "pair_loss_weight", "evaluation_scope")}
        checkpoint = get_run_artifact_paths(get_run_dir(**task))["checkpoint"]
        missing += not checkpoint.is_file() or checkpoint.stat().st_size == 0
    free = shutil.disk_usage(RESULTS_ROOT).free
    required = missing * CHECKPOINT_BUDGET_BYTES + RESERVE_BYTES
    result = {
        "results_root": str(RESULTS_ROOT.resolve()), "free_bytes": free, "required_free_bytes_estimate": required, "checkpoint_budget_bytes": CHECKPOINT_BUDGET_BYTES,
        "reserve_bytes": RESERVE_BYTES, "planned_unique_runs": len(plan), "new_checkpoint_count": missing,
        "selection_known": decision is not None, "selection": decision,
    }
    if free < required:
        raise RuntimeError(f"Insufficient estimated result storage: {required / 1024 ** 3:.1f} GiB required, {free / 1024 ** 3:.1f} GiB available.")
    with tempfile.TemporaryFile(dir=RESULTS_ROOT) as stream:
        stream.write(b"pair-aware-preflight")
        stream.flush()
    return result


def run_preflight(include_references=False, output_root=PREFLIGHT_ROOT):
    output_root = Path(output_root)
    report_path = output_root / "preflight_report.json"
    with run_lock(output_root, experiment_id="v3_preflight"):
        report = {"preflight_version": 1, "status": "running", "include_references": include_references, "preflight_source_sha256": _hash_file(__file__)}
        write_json(report, report_path)
        try:
            report["v2_metadata"] = verify_v2_protocol()
            report["validation_data"] = verify_validation_data()
            report["environment"] = verify_dependencies()
            report["storage"] = verify_storage(include_references)
            report["runtime"] = verify_bf16_runtime()
            report["status"] = "completed"
            write_json(report, report_path)
        except (Exception, KeyboardInterrupt) as error:
            report.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error_message": str(error)})
            write_json(report, report_path)
            raise
    return {**report, "report_path": str(report_path.resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--output-root", type=Path, default=PREFLIGHT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("Checks: definitive V2 protocol, train/validation splits, pinned dependencies, result storage and native CUDA BF16 forward/backward.")
        print("Test predictions and test split contents are excluded from preflight.")
        return
    try:
        report = run_preflight(args.include_references, args.output_root)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    print(f"Preflight status: {report['status']}")
    print(f"Preflight report: {report['report_path']}")
    print(f"GPU: {report['runtime']['gpu']}")
    print(f"Estimated free storage required: {report['storage']['required_free_bytes_estimate'] / 1024 ** 3:.1f} GiB")


if __name__ == "__main__":
    main()
