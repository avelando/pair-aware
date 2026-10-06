import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd

from src.evaluation.scope import get_training_split_names
from src.experiments.fingerprint import build_experiment_fingerprint
from src.experiments.lifecycle import run_lock
from src.paths import PROJECT_ROOT, RESULTS_ROOT, get_run_dir, get_split_dir
from src.results.collection import COLLECTION_ROOT, TASK_KEYS, build_collection_plan
from src.results.confirmation import load_screening_selection
from src.results.io import get_run_artifact_paths, write_json
from src.results.selection import _read_json
import src.results.collection as collection_module


EXPORT_ROOT = RESULTS_ROOT / "export"
MANIFEST_NAME = "archive_manifest.json"


def _digest(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _hash(path):
    with Path(path).open("rb") as stream:
        return _digest(stream)


def _safe_name(name):
    path = PurePosixPath(name)
    return bool(path.parts) and "\\" not in name and not path.is_absolute() and ".." not in path.parts and path.as_posix() == name


def verify_result_archive(path, expected_sha256=None):
    if expected_sha256 is not None and _hash(path) != expected_sha256:
        raise ValueError("The archive SHA-256 does not match.")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or any(not member.isfile() or not _safe_name(member.name) for member in members):
            raise ValueError("The archive contains duplicate or unsafe members.")
        with archive.extractfile(MANIFEST_NAME) as stream:
            manifest = json.load(stream)
        if manifest.get("archive_version") != 1 or manifest.get("status") != "completed" or manifest.get("checkpoints_included") is not False:
            raise ValueError("Invalid result archive manifest.")
        records = manifest["files"]
        expected = [row["path"] for row in records]
        if len(expected) != len(set(expected)) or set(names) != {MANIFEST_NAME, *expected} or MANIFEST_NAME in expected:
            raise ValueError("The archive does not match its file inventory.")
        for row in records:
            member = archive.getmember(row["path"])
            with archive.extractfile(member) as stream:
                if member.size != row["size_bytes"] or _digest(stream) != row["sha256"]:
                    raise ValueError(f"Archive member failed verification: {member.name}.")
    return manifest


def _snapshot_files(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    report = _read_json(manifest_path)
    decision = load_screening_selection(report["selection"]["selection_path"])
    plan = build_collection_plan(decision["selected_pair_loss_weight"], report["include_references"])
    if (
        report.get("collection_version") != 1 or report.get("status") != "completed"
        or report.get("source_predictions_verified") is not True
        or report.get("collection_source_sha256") != _hash(collection_module.__file__)
        or report.get("selection") != decision or report.get("row_count") != len(plan)
        or report.get("expected_runs") != len(plan)
    ):
        raise RuntimeError("Export requires a complete collection from the current protocol.")
    files = {}

    def add(source, name, expected_hash=None):
        source = Path(source).resolve()
        if not source.is_file() or not _safe_name(name) or name in files:
            raise ValueError(f"Invalid or duplicate export file: {name}.")
        digest = _hash(source)
        if expected_hash is not None and digest != expected_hash:
            raise RuntimeError(f"An export input has changed: {source}.")
        files[name] = {"path": name, "size_bytes": source.stat().st_size, "sha256": digest, "source": source}

    add(manifest_path, "collection/collection_manifest.json")
    for name in ("runs", "artifacts"):
        entry = report["outputs"][name]
        path = manifest_path.parent / f"{name}.csv"
        add(path, f"collection/{name}.csv", entry["sha256"])
    runs = pd.read_csv(manifest_path.parent / "runs.csv", dtype={"pair_loss_weight": float}, na_values={"pair_loss_weight": [""]}, keep_default_na=False, float_precision="round_trip")
    artifacts = pd.read_csv(manifest_path.parent / "artifacts.csv", keep_default_na=False)
    if len(runs) != len(plan) or any(len(frame) != report["outputs"][name]["row_count"] for name, frame in (("runs", runs), ("artifacts", artifacts))):
        raise RuntimeError("Collection table row counts do not match.")
    selection_path = Path(decision["selection_path"])
    add(selection_path, "screening/selection.json", decision["selection_sha256"])
    selection = _read_json(selection_path)
    for name, filename in (("runs", "screening_runs.csv"), ("by_split", "screening_by_split.csv"), ("by_lambda", "screening_by_lambda.csv")):
        add(selection_path.parent / filename, f"screening/{filename}", selection["outputs"][name]["sha256"])
    seeds = {}
    for entry in plan:
        seeds.setdefault(entry["split_seed"], set()).update(get_training_split_names(entry["evaluation_scope"]))
    for seed, splits in sorted(seeds.items()):
        for filename in [*(f"{split}.jsonl" for split in sorted(splits)), "metadata.json"]:
            add(get_split_dir(seed) / filename, f"data/pair_controlled/seed_{seed}/{filename}")
    for folder in ("src", "tests"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            add(path, path.relative_to(PROJECT_ROOT).as_posix())
    for name in ("pyproject.toml", "uv.lock"):
        add(PROJECT_ROOT / name, name)
    for name in (".python-version", "LICENSE", "LICENSE.md", "LICENSE.txt"):
        if (PROJECT_ROOT / name).is_file():
            add(PROJECT_ROOT / name, name)
    for path in sorted((PROJECT_ROOT / "results_v2" / "bf16").glob("*/split_*/model_seed_*/metadata.json")):
        add(path, path.relative_to(PROJECT_ROOT).as_posix())
    for path in (
        RESULTS_ROOT / "pipeline" / "preflight" / "preflight_report.json",
        RESULTS_ROOT / "preflight" / "preflight_report.json",
    ):
        if path.is_file():
            add(path, path.relative_to(PROJECT_ROOT).as_posix())
    identities = []
    for index, entry in enumerate(plan):
        task = {key: entry[key] for key in TASK_KEYS}
        expected = build_experiment_fingerprint(**task)
        row = runs.iloc[index]
        values = {**task, "phases": "|".join(entry["phases"]), **{key: expected[key] for key in ("experiment_id", "config_hash", "dataset_hash", "source_hash")}}
        if any(not pd.isna(row[key]) if value is None else row[key] != value for key, value in values.items()):
            raise RuntimeError("Collected runs do not match the current experiment fingerprints.")
        paths = get_run_artifact_paths(get_run_dir(**task))
        names = ["metadata", "metrics", "history", "completed", "validation_predictions", "validation_pair_predictions"]
        if task["evaluation_scope"] == "full":
            names.extend(("predictions", "pair_predictions"))
        saved = artifacts.loc[artifacts["experiment_id"] == expected["experiment_id"]]
        if len(saved) != len(names) or set(saved["artifact"]) != set(names):
            raise RuntimeError("The collection has an incomplete run artifact inventory.")
        for artifact in saved.to_dict("records"):
            path = paths[artifact["artifact"]].resolve()
            if artifact["path"] != str(path) or artifact["size_bytes"] != path.stat().st_size:
                raise RuntimeError("The collection artifact path or size does not match.")
            add(path, f"runs/{expected['experiment_id']}/{path.name}", artifact["sha256"])
        identities.append(expected["experiment_id"])
    if set(artifacts["experiment_id"]) != set(identities):
        raise RuntimeError("Unexpected experiments in the artifact inventory.")
    return report, sorted(files.values(), key=lambda row: row["path"])


def export_v3_results(collection_manifest=COLLECTION_ROOT / "collection_manifest.json", output_root=EXPORT_ROOT):
    output_root = Path(output_root).resolve()
    archive_path = output_root / "v3-results.tar.gz"
    status_path = output_root / "export_status.json"
    temporary = output_root / "v3-results.tar.gz.tmp"
    with run_lock(output_root, experiment_id="v3_result_export"), run_lock(Path(collection_manifest).parent, experiment_id="v3_result_collection"):
        status = {"export_version": 1, "status": "running", "export_source_sha256": _hash(__file__)}
        write_json(status, status_path)
        try:
            report, files = _snapshot_files(collection_manifest)
            manifest = {
                "archive_version": 1, "status": "completed", "checkpoints_included": False,
                "selection": report["selection"], "include_references": report["include_references"],
                "row_count": report["row_count"], "export_source_sha256": status["export_source_sha256"],
                "files": [{key: value for key, value in row.items() if key != "source"} for row in files],
            }
            with tarfile.open(temporary, "w:gz") as archive:
                for row in files:
                    info = tarfile.TarInfo(row["path"])
                    info.size, info.mode = row["size_bytes"], 0o644
                    with row["source"].open("rb") as stream:
                        archive.addfile(info, stream)
                content = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
                info = tarfile.TarInfo(MANIFEST_NAME)
                info.size, info.mode = len(content), 0o644
                archive.addfile(info, io.BytesIO(content))
            if any(_hash(row["source"]) != row["sha256"] for row in files):
                raise RuntimeError("Export inputs changed while the archive was being written.")
            verify_result_archive(temporary)
            digest = _hash(temporary)
            temporary.replace(archive_path)
            checksum = output_root / "v3-results.tar.gz.sha256"
            checksum_tmp = checksum.with_suffix(".tmp")
            checksum_tmp.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
            checksum_tmp.replace(checksum)
            status.update({
                "status": "completed", "selection": report["selection"], "row_count": report["row_count"],
                "archive_path": str(archive_path), "archive_sha256": digest, "checksum_path": str(checksum),
                "file_count": len(files), "checkpoints_included": False,
            })
            write_json(status, status_path)
        except (Exception, KeyboardInterrupt) as error:
            temporary.unlink(missing_ok=True)
            status.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error_message": str(error)})
            write_json(status, status_path)
            raise
    return {**status, "status_path": str(status_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-manifest", type=Path, default=COLLECTION_ROOT / "collection_manifest.json")
    parser.add_argument("--output-root", type=Path, default=EXPORT_ROOT)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.sha256 is not None and args.verify is None:
        parser.error("--sha256 requires --verify.")
    if args.verify is not None:
        result = verify_result_archive(args.verify, args.sha256)
        print(f"Verified archive files: {len(result['files'])}")
    elif args.dry_run:
        print(f"Result archive: {args.output_root / 'v3-results.tar.gz'}")
        print("Included: collected results, run artifacts, screening decision, splits, source, tests and dependency lock.")
        print("Checkpoints included: False")
    else:
        try:
            result = export_v3_results(args.collection_manifest, args.output_root)
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        print(f"Result archive: {result['archive_path']}")
        print(f"Archive SHA-256: {result['archive_sha256']}")


if __name__ == "__main__":
    main()
