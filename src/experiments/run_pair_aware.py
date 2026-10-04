import argparse
from functools import partial
from time import perf_counter

import torch

from src.config import (
    AMP_GROWTH_INTERVAL,
    AMP_INIT_SCALE,
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    KEEP_CHECKPOINTS,
    LEARNING_RATE,
    LR_SCHEDULER_TYPE,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    METRIC_FOR_BEST_MODEL,
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_SEEDS,
    NUM_EPOCHS,
    PAIR_BATCH_SIZE,
    PAIR_LOSS_WEIGHT,
    PAIRING_STRATEGIES,
    SPLIT_SEEDS,
    TRAINING_PRECISION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.data.datasets import (
    InstanceDataset,
    PairDataset,
)
from src.data.loading import (
    load_metadata,
    load_split_directory,
)
from src.data.pairing import build_pairs
from src.data.validation import validate_input_splits
from src.evaluation.artifacts import evaluate_split_predictions
from src.evaluation.metrics import (
    evaluate_instance_logits,
    evaluate_pair_predictions,
)
from src.evaluation.predictions import (
    build_instance_predictions,
    build_pair_predictions,
)
from src.experiments.progress import RunProgressTracker
from src.experiments.runner import (
    run_experiment,
    validate_seeds,
)
from src.experiments.provenance import (
    get_environment_metadata,
    get_git_provenance,
    utc_now,
)
from src.models.factory import (
    create_sequence_classifier,
    create_tokenizer,
)
from src.pair_loss import format_pair_loss_weight, validate_pair_loss_weight
from src.results.io import (
    mark_run_completed,
    prepare_run_directory,
    remove_checkpoint,
    write_dataframe,
    write_history,
    write_json,
)
from src.training.common import (
    clear_memory,
    create_dataloader,
    get_device,
    predict_instances,
)
from src.training.pair_aware import (
    train_pair_aware_model,
)
from src.training.reproducibility import set_model_seed


def validate_run_arguments(
    method,
    split_seed,
    model_seed,
):
    if method not in PAIRING_STRATEGIES:
        raise ValueError(
            f"Invalid pair-aware method: {method}."
        )

    validate_seeds(
        split_seed,
        model_seed,
    )


def build_training_pairs(
    train_dataframe,
    method,
    split_seed,
):
    if method == "true_pair":
        return build_pairs(
            train_dataframe,
            "true_pair",
        )

    return build_pairs(
        train_dataframe,
        "shuffled_pair",
        seed=split_seed,
    )


def create_run_metadata(
    method,
    split_seed,
    model_seed,
    run_dir,
    split_metadata,
    train_pair_count,
    experiment_fingerprint,
    attempt,
    device,
    pair_loss_weight=PAIR_LOSS_WEIGHT,
):
    pair_loss_weight = validate_pair_loss_weight(pair_loss_weight)
    pairing_metadata = {
        "strategy": method,
        "train_pair_count": int(
            train_pair_count
        ),
        "validation_uses_true_pairs": True,
        "test_uses_true_pairs": True,
    }

    if method == "shuffled_pair":
        pairing_metadata[
            "shuffle_seed"
        ] = split_seed
        pairing_metadata[
            "shuffle_seed_source"
        ] = "split_seed"
        pairing_metadata[
            "pairing_algorithm"
        ] = "uniform_derangement_by_rejection_sampling"
    else:
        pairing_metadata[
            "shuffle_seed"
        ] = None
        pairing_metadata[
            "shuffle_seed_source"
        ] = None
        pairing_metadata[
            "pairing_algorithm"
        ] = "true_counterfactual_pair"

    return {
        "status": "running",
        "method": method,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "run_dir": str(run_dir),
        "experiment_id": experiment_fingerprint[
            "experiment_id"
        ],
        "attempt": attempt,
        "fingerprint": {
            "version": experiment_fingerprint[
                "fingerprint_version"
            ],
            "config_hash": experiment_fingerprint[
                "config_hash"
            ],
            "dataset_hash": experiment_fingerprint[
                "dataset_hash"
            ],
            "source_hash": experiment_fingerprint[
                "source_hash"
            ],
            "config": experiment_fingerprint[
                "config"
            ],
        },
        "started_at_utc": utc_now(),
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "max_length": MAX_LENGTH,
        },
        "pairing": pairing_metadata,
        "training": {
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "pair_batch_size": PAIR_BATCH_SIZE,
            "texts_per_training_batch": (
                PAIR_BATCH_SIZE * 2
            ),
            "eval_batch_size": EVAL_BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "lr_scheduler_type": LR_SCHEDULER_TYPE,
            "max_grad_norm": MAX_GRAD_NORM,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "metric_for_best_model": METRIC_FOR_BEST_MODEL,
            "precision": TRAINING_PRECISION,
            "pair_loss_weight": pair_loss_weight,
            "amp_init_scale": AMP_INIT_SCALE,
            "amp_growth_interval": AMP_GROWTH_INTERVAL,
        },
        "seed_control": {
            "seed": model_seed,
            "controls": [
                "python_rng",
                "numpy_rng",
                "torch_rng",
                "model_initialization",
                "training_stochasticity",
                "train_dataloader_order",
            ],
        },
        "code": get_git_provenance(),
        "environment": get_environment_metadata(
            device
        ),
        "split_metadata": split_metadata,
    }


def _execute_pair_aware(
    method,
    split_seed,
    model_seed,
    run_dir,
    experiment_fingerprint,
    attempt,
    pair_loss_weight=PAIR_LOSS_WEIGHT,
):
    pair_loss_weight = validate_pair_loss_weight(pair_loss_weight)
    artifact_paths = prepare_run_directory(
        run_dir
    )

    run_name = (
        f"{method}/"
        f"lambda_{format_pair_loss_weight(pair_loss_weight)}/"
        f"split_{split_seed}/"
        f"model_seed_{model_seed}"
    )

    split_data = load_split_directory(
        split_seed
    )

    validate_input_splits(
        split_data,
        run_name,
    )

    split_metadata = load_metadata(
        split_seed
    )

    train_pairs = build_training_pairs(
        train_dataframe=split_data[
            "train"
        ],
        method=method,
        split_seed=split_seed,
    )

    set_model_seed(
        model_seed
    )

    clear_memory()

    device = get_device()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    metadata = create_run_metadata(
        method=method,
        split_seed=split_seed,
        model_seed=model_seed,
        run_dir=run_dir,
        split_metadata=split_metadata,
        train_pair_count=len(
            train_pairs
        ),
        experiment_fingerprint=(
            experiment_fingerprint
        ),
        attempt=attempt,
        device=device,
        pair_loss_weight=pair_loss_weight,
    )

    write_json(
        metadata,
        artifact_paths["metadata"],
    )

    model = None
    progress_tracker = None
    start_time = perf_counter()

    try:
        tokenizer = create_tokenizer()

        train_dataset = PairDataset(
            pair_dataframe=train_pairs,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )

        validation_dataset = (
            InstanceDataset(
                dataframe=split_data[
                    "validation"
                ],
                tokenizer=tokenizer,
                max_length=MAX_LENGTH,
            )
        )

        test_dataset = InstanceDataset(
            dataframe=split_data["test"],
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )

        train_loader = create_dataloader(
            dataset=train_dataset,
            batch_size=PAIR_BATCH_SIZE,
            shuffle=True,
            seed=model_seed,
        )

        validation_loader = (
            create_dataloader(
                dataset=validation_dataset,
                batch_size=EVAL_BATCH_SIZE,
                shuffle=False,
            )
        )

        test_loader = create_dataloader(
            dataset=test_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
        )

        model = (
            create_sequence_classifier()
            .to(device)
        )

        progress_tracker = (
            RunProgressTracker(
                path=artifact_paths[
                    "progress"
                ],
                method=method,
                split_seed=split_seed,
                model_seed=model_seed,
                experiment_id=(
                    experiment_fingerprint[
                        "experiment_id"
                    ]
                ),
                attempt=attempt,
                total_epochs=NUM_EPOCHS,
            )
        )

        progress_tracker.start()

        training_result = (
            train_pair_aware_model(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                checkpoint_path=artifact_paths[
                    "checkpoint"
                ],
                device=device,
                progress_callback=(
                    progress_tracker.update_epoch
                ),
                pair_loss_weight=pair_loss_weight,
            )
        )

        progress_tracker.set_phase(
            "evaluation"
        )

        validation_logits, validation_labels = predict_instances(
            model=model,
            dataloader=validation_loader,
            device=device,
        )

        validation_result = evaluate_split_predictions(
            dataframe=split_data["validation"],
            logits=validation_logits,
            labels=validation_labels,
            split_name="validation",
        )

        (
            test_logits,
            test_labels,
        ) = predict_instances(
            model=model,
            dataloader=test_loader,
            device=device,
        )

        expected_test_labels = (
            split_data["test"][
                "label"
            ]
            .astype(int)
            .tolist()
        )

        returned_test_labels = (
            test_labels
            .astype(int)
            .tolist()
        )

        if (
            returned_test_labels
            != expected_test_labels
        ):
            raise RuntimeError(
                "Test labels returned by the dataloader "
                "do not match the test dataframe order."
            )

        (
            instance_metrics,
            classification_report,
            confusion_matrix,
            _,
        ) = evaluate_instance_logits(
            test_labels,
            test_logits,
        )

        instance_predictions = (
            build_instance_predictions(
                split_data["test"],
                test_logits,
            )
        )

        pair_predictions = (
            build_pair_predictions(
                instance_predictions
            )
        )

        pair_metrics = (
            evaluate_pair_predictions(
                pair_predictions
            )
        )

        peak_allocated_gb = float(
            torch.cuda.max_memory_allocated(
                device
            )
            / 1024**3
        )

        peak_reserved_gb = float(
            torch.cuda.max_memory_reserved(
                device
            )
            / 1024**3
        )

        duration_seconds = float(
            perf_counter()
            - start_time
        )

        metrics = {
            "training": {
                "best_validation_f1_macro": training_result[
                    "best_validation_f1_macro"
                ],
                "best_epoch": training_result[
                    "best_epoch"
                ],
                "epochs_trained": training_result[
                    "epochs_trained"
                ],
                "optimizer_steps": training_result[
                    "training_steps"
                ],
                "skipped_optimizer_steps": training_result[
                    "skipped_training_steps"
                ],
                "planned_training_steps": training_result[
                    "planned_training_steps"
                ],
                "warmup_steps": training_result[
                    "warmup_steps"
                ],
                "pair_loss_weight": pair_loss_weight,
            },
            "validation_instance": validation_result["instance"],
            "validation_pair": validation_result["pair"],
            "validation_confusion_matrix": validation_result["confusion_matrix"],
            "validation_classification_report": validation_result["classification_report"],
            "test_instance": instance_metrics,
            "test_pair": pair_metrics,
            "test_confusion_matrix": (
                confusion_matrix.tolist()
            ),
            "test_classification_report": (
                classification_report
            ),
            "runtime": {
                "duration_seconds": duration_seconds,
                "peak_allocated_gb": peak_allocated_gb,
                "peak_reserved_gb": peak_reserved_gb,
            },
        }

        write_history(
            training_result["history"],
            artifact_paths["history"],
        )

        write_dataframe(
            instance_predictions,
            artifact_paths[
                "predictions"
            ],
        )

        write_dataframe(
            pair_predictions,
            artifact_paths[
                "pair_predictions"
            ],
        )

        write_json(
            metrics,
            artifact_paths["metrics"],
        )

        write_dataframe(
            validation_result["predictions"],
            artifact_paths["validation_predictions"],
        )

        write_dataframe(
            validation_result["pair_predictions"],
            artifact_paths["validation_pair_predictions"],
        )

        if not KEEP_CHECKPOINTS:
            remove_checkpoint(
                run_dir
            )

        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "duration_seconds": duration_seconds,
                "best_epoch": training_result[
                    "best_epoch"
                ],
                "best_validation_f1_macro": training_result[
                    "best_validation_f1_macro"
                ],
            }
        )

        write_json(
            metadata,
            artifact_paths["metadata"],
        )

        mark_run_completed(
            run_dir
        )

        progress_tracker.complete()

        return {
            "status": "completed",
            "method": method,
            "pair_loss_weight": pair_loss_weight,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "run_dir": str(run_dir),
            "experiment_id": experiment_fingerprint[
                "experiment_id"
            ],
            "best_epoch": training_result[
                "best_epoch"
            ],
            "best_validation_f1_macro": training_result[
                "best_validation_f1_macro"
            ],
            "test_accuracy": instance_metrics[
                "accuracy"
            ],
            "test_f1_macro": instance_metrics[
                "f1_macro"
            ],
            "pair_ranking_accuracy": pair_metrics[
                "pair_ranking_accuracy"
            ],
            "pair_exact_match": pair_metrics[
                "pair_exact_match"
            ],
            "duration_seconds": duration_seconds,
        }

    except BaseException as error:
        if progress_tracker is not None:
            progress_tracker.fail(
                error
            )

        duration_seconds = float(
            perf_counter()
            - start_time
        )

        metadata.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "duration_seconds": duration_seconds,
                "error_type": type(
                    error
                ).__name__,
                "error_message": str(
                    error
                ),
            }
        )

        write_json(
            metadata,
            artifact_paths["metadata"],
        )

        if not KEEP_CHECKPOINTS:
            remove_checkpoint(
                run_dir
            )

        raise

    finally:
        if model is not None:
            del model

        clear_memory()


def run_pair_aware(
    method,
    split_seed,
    model_seed,
    force=False,
    pair_loss_weight=PAIR_LOSS_WEIGHT,
):
    validate_run_arguments(
        method,
        split_seed,
        model_seed,
    )

    pair_loss_weight = validate_pair_loss_weight(pair_loss_weight)

    execute = partial(
        _execute_pair_aware,
        method=method,
        pair_loss_weight=pair_loss_weight,
    )

    return run_experiment(
        method=method,
        split_seed=split_seed,
        model_seed=model_seed,
        execute=execute,
        force=force,
        pair_loss_weight=pair_loss_weight,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        required=True,
        choices=PAIRING_STRATEGIES,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        required=True,
        choices=SPLIT_SEEDS,
    )

    parser.add_argument(
        "--model-seed",
        type=int,
        required=True,
        choices=MODEL_SEEDS,
    )

    parser.add_argument(
        "--pair-loss-weight",
        type=validate_pair_loss_weight,
        default=PAIR_LOSS_WEIGHT,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = run_pair_aware(
        method=args.method,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        force=args.force,
        pair_loss_weight=args.pair_loss_weight,
    )

    print(
        f"Status: {result['status']}"
    )

    print(
        f"Method: {result['method']}"
    )

    print(
        f"Run directory: "
        f"{result['run_dir']}"
    )

    if result["status"] == "completed":
        print(
            f"Best epoch: "
            f"{result['best_epoch']}"
        )

        print(
            f"Best validation F1: "
            f"{result['best_validation_f1_macro']:.6f}"
        )

        print(
            f"Test accuracy: "
            f"{result['test_accuracy']:.6f}"
        )

        print(
            f"Test macro F1: "
            f"{result['test_f1_macro']:.6f}"
        )

        print(
            f"Pair ranking accuracy: "
            f"{result['pair_ranking_accuracy']:.6f}"
        )

        print(
            f"Pair exact match: "
            f"{result['pair_exact_match']:.6f}"
        )

        print(
            f"Duration seconds: "
            f"{result['duration_seconds']:.2f}"
        )


if __name__ == "__main__":
    main()
