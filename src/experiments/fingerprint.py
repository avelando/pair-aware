import hashlib
import json
from pathlib import Path

from src.config import (
    AMP_GROWTH_INTERVAL,
    AMP_INIT_SCALE,
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    EXPERIMENT_VERSION,
    ID2LABEL,
    INSTANCE_TRAIN_BATCH_SIZE,
    LABEL2ID,
    LEARNING_RATE,
    LR_SCHEDULER_TYPE,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    METRIC_FOR_BEST_MODEL,
    MODEL_NAME,
    MODEL_REVISION,
    NUM_EPOCHS,
    PAIR_BATCH_SIZE,
    PAIR_LOSS_WEIGHT,
    SPLIT_NAMES,
    TRAINING_PRECISION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.paths import PROJECT_ROOT, get_split_dir


FINGERPRINT_VERSION = 1


def _canonical_json(data):
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _hash_files(paths, root):
    root = Path(root)
    digest = hashlib.sha256()

    for path in sorted(
        (Path(path) for path in paths),
        key=lambda value: value.as_posix(),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Fingerprint file not found: {path}"
            )

        try:
            relative_path = path.relative_to(
                root
            )
        except ValueError:
            relative_path = path

        digest.update(
            relative_path.as_posix().encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def get_config_snapshot(method):
    common = {
        "experiment_version": EXPERIMENT_VERSION,
        "method": method,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "training_precision": TRAINING_PRECISION,
        "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "max_grad_norm": MAX_GRAD_NORM,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "metric_for_best_model": METRIC_FOR_BEST_MODEL,
        "amp_init_scale": AMP_INIT_SCALE,
        "amp_growth_interval": AMP_GROWTH_INTERVAL,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }

    if method == "instance_level":
        common["train_batch_size"] = (
            INSTANCE_TRAIN_BATCH_SIZE
        )
        return common

    if method in {
        "true_pair",
        "shuffled_pair",
    }:
        common["pair_batch_size"] = (
            PAIR_BATCH_SIZE
        )
        common["pair_loss_weight"] = (
            PAIR_LOSS_WEIGHT
        )
        return common

    raise ValueError(
        f"Invalid experiment method: {method}."
    )


def get_config_hash(method):
    return _sha256_text(
        _canonical_json(
            get_config_snapshot(method)
        )
    )


def get_dataset_fingerprint(split_seed):
    split_dir = get_split_dir(
        split_seed
    )

    paths = [
        split_dir / f"{split_name}.jsonl"
        for split_name in SPLIT_NAMES
    ]

    paths.append(
        split_dir / "metadata.json"
    )

    return _hash_files(
        paths,
        split_dir,
    )


def _source_paths(method):
    src_root = PROJECT_ROOT / "src"

    relative_paths = [
        "config.py",
        "data/datasets.py",
        "data/loading.py",
        "data/pairing.py",
        "data/validation.py",
        "evaluation/metrics.py",
        "evaluation/predictions.py",
        "experiments/lifecycle.py",
        "experiments/runner.py",
        "models/factory.py",
        "results/io.py",
        "training/common.py",
        "training/reproducibility.py",
    ]

    if method == "instance_level":
        relative_paths.extend(
            [
                "training/instance_level.py",
                "experiments/run_instance_level.py",
            ]
        )
    elif method in {
        "true_pair",
        "shuffled_pair",
    }:
        relative_paths.extend(
            [
                "training/pair_aware.py",
                "experiments/run_pair_aware.py",
            ]
        )
    else:
        raise ValueError(
            f"Invalid experiment method: {method}."
        )

    return [
        src_root / relative_path
        for relative_path in relative_paths
    ]


def get_source_fingerprint(method):
    return _hash_files(
        _source_paths(method),
        PROJECT_ROOT,
    )


def build_experiment_fingerprint(
    method,
    split_seed,
    model_seed,
):
    config_snapshot = (
        get_config_snapshot(method)
    )

    config_hash = _sha256_text(
        _canonical_json(
            config_snapshot
        )
    )

    dataset_hash = (
        get_dataset_fingerprint(
            split_seed
        )
    )

    source_hash = (
        get_source_fingerprint(
            method
        )
    )

    identity = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "method": method,
        "split_seed": int(split_seed),
        "model_seed": int(model_seed),
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "source_hash": source_hash,
    }

    experiment_id = _sha256_text(
        _canonical_json(
            identity
        )
    )

    return {
        "experiment_id": experiment_id,
        "fingerprint_version": FINGERPRINT_VERSION,
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "source_hash": source_hash,
        "config": config_snapshot,
    }