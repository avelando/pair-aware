from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    EARLY_STOPPING_PATIENCE,
    METRIC_FOR_BEST_MODEL,
    NUM_EPOCHS,
    PAIR_LOSS_WEIGHT,
)
from src.evaluation.metrics import evaluate_instance_logits
from src.training.common import (
    autocast_context,
    backward_and_step,
    create_grad_scaler,
    create_optimizer_and_scheduler,
    load_model_state,
    predict_instances,
    save_model_state,
)


def _move_pair_inputs(
    batch,
    prefix,
    device,
):
    inputs = {}

    for key, value in batch.items():
        if not key.startswith(prefix):
            continue

        input_name = key[
            len(prefix):
        ]

        if input_name == "labels":
            continue

        inputs[input_name] = value.to(
            device,
            non_blocking=True,
        )

    if not inputs:
        raise ValueError(
            f"No inputs found for prefix {prefix}."
        )

    return inputs


def compute_pair_aware_loss(
    model,
    batch,
    device,
):
    h_inputs = _move_pair_inputs(
        batch,
        "h_",
        device,
    )

    n_inputs = _move_pair_inputs(
        batch,
        "n_",
        device,
    )

    if set(h_inputs) != set(n_inputs):
        raise ValueError(
            "H and N inputs must contain the same model fields."
        )

    h_labels = batch[
        "h_labels"
    ].to(
        device,
        non_blocking=True,
    )

    n_labels = batch[
        "n_labels"
    ].to(
        device,
        non_blocking=True,
    )

    if h_labels.shape != n_labels.shape:
        raise ValueError(
            "H and N labels must have the same shape."
        )

    pair_batch_size = int(
        h_labels.shape[0]
    )

    combined_inputs = {
        key: torch.cat(
            [
                h_inputs[key],
                n_inputs[key],
            ],
            dim=0,
        )
        for key in h_inputs
    }

    combined_labels = torch.cat(
        [
            h_labels,
            n_labels,
        ],
        dim=0,
    )

    outputs = model(
        **combined_inputs,
        labels=combined_labels,
    )

    logits = outputs.logits

    if logits.shape[0] != (
        pair_batch_size * 2
    ):
        raise ValueError(
            "Unexpected number of logits for paired batch."
        )

    classification_loss = (
        outputs.loss.float()
    )

    h_logits = logits[
        :pair_batch_size
    ].float()

    n_logits = logits[
        pair_batch_size:
    ].float()

    h_scores = (
        h_logits[:, 1]
        - h_logits[:, 0]
    )

    n_scores = (
        n_logits[:, 1]
        - n_logits[:, 0]
    )

    pair_margins = (
        h_scores
        - n_scores
    )

    pair_loss = F.softplus(
        -pair_margins
    ).mean()

    total_loss = (
        classification_loss
        + PAIR_LOSS_WEIGHT
        * pair_loss
    )

    return {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "pair_loss": pair_loss,
        "mean_pair_margin": pair_margins.mean(),
        "pair_batch_size": pair_batch_size,
    }


def train_pair_aware_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
):
    model.train()

    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    pair_loss_sum = 0.0
    pair_margin_sum = 0.0

    total_pairs = 0
    total_batches = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0

    for batch in dataloader:
        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast_context(device):
            loss_result = (
                compute_pair_aware_loss(
                    model=model,
                    batch=batch,
                    device=device,
                )
            )

        total_loss = loss_result[
            "total_loss"
        ]

        classification_loss = loss_result[
            "classification_loss"
        ]

        pair_loss = loss_result[
            "pair_loss"
        ]

        mean_pair_margin = loss_result[
            "mean_pair_margin"
        ]

        pair_batch_size = loss_result[
            "pair_batch_size"
        ]

        if not torch.isfinite(
            total_loss
        ):
            raise RuntimeError(
                "Non-finite total loss detected during training."
            )

        if not torch.isfinite(
            classification_loss
        ):
            raise RuntimeError(
                "Non-finite classification loss detected during training."
            )

        if not torch.isfinite(
            pair_loss
        ):
            raise RuntimeError(
                "Non-finite pair loss detected during training."
            )

        step_result = backward_and_step(
            loss=total_loss,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        if step_result["step_skipped"]:
            skipped_optimizer_steps += 1
        else:
            optimizer_steps += 1

        total_loss_sum += (
            float(
                total_loss.detach().cpu()
            )
            * pair_batch_size
        )

        classification_loss_sum += (
            float(
                classification_loss.detach().cpu()
            )
            * pair_batch_size
        )

        pair_loss_sum += (
            float(
                pair_loss.detach().cpu()
            )
            * pair_batch_size
        )

        pair_margin_sum += (
            float(
                mean_pair_margin.detach().cpu()
            )
            * pair_batch_size
        )

        total_pairs += pair_batch_size
        total_batches += 1

    if total_pairs == 0:
        raise ValueError(
            "Training dataloader is empty."
        )

    return {
        "total_loss": (
            total_loss_sum
            / total_pairs
        ),
        "classification_loss": (
            classification_loss_sum
            / total_pairs
        ),
        "pair_loss": (
            pair_loss_sum
            / total_pairs
        ),
        "mean_pair_margin": (
            pair_margin_sum
            / total_pairs
        ),
        "pairs": total_pairs,
        "texts": total_pairs * 2,
        "batches": total_batches,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "grad_scale": float(
            scaler.get_scale()
        ),
    }


def train_pair_aware_model(
    model,
    train_loader,
    validation_loader,
    checkpoint_path,
    device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    (
        optimizer,
        scheduler,
        planned_training_steps,
        warmup_steps,
    ) = create_optimizer_and_scheduler(
        model,
        steps_per_epoch=len(
            train_loader
        ),
    )

    scaler = create_grad_scaler(
        device
    )

    best_validation_score = -np.inf
    best_epoch = None
    epochs_without_improvement = 0
    executed_training_steps = 0
    skipped_training_steps = 0
    history = []

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        train_result = (
            train_pair_aware_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )
        )

        executed_training_steps += (
            train_result[
                "optimizer_steps"
            ]
        )

        skipped_training_steps += (
            train_result[
                "skipped_optimizer_steps"
            ]
        )

        (
            validation_logits,
            validation_labels,
        ) = predict_instances(
            model=model,
            dataloader=validation_loader,
            device=device,
        )

        (
            validation_metrics,
            _,
            _,
            _,
        ) = evaluate_instance_logits(
            validation_labels,
            validation_logits,
        )

        validation_score = float(
            validation_metrics[
                METRIC_FOR_BEST_MODEL
            ]
        )

        history.append(
            {
                "epoch": epoch,
                "train_total_loss": float(
                    train_result[
                        "total_loss"
                    ]
                ),
                "train_classification_loss": float(
                    train_result[
                        "classification_loss"
                    ]
                ),
                "train_pair_loss": float(
                    train_result[
                        "pair_loss"
                    ]
                ),
                "train_mean_pair_margin": float(
                    train_result[
                        "mean_pair_margin"
                    ]
                ),
                "validation_accuracy": float(
                    validation_metrics[
                        "accuracy"
                    ]
                ),
                "validation_f1_macro": float(
                    validation_metrics[
                        "f1_macro"
                    ]
                ),
                "learning_rate": float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
                "optimizer_steps": int(
                    train_result[
                        "optimizer_steps"
                    ]
                ),
                "skipped_optimizer_steps": int(
                    train_result[
                        "skipped_optimizer_steps"
                    ]
                ),
                "grad_scale": float(
                    train_result[
                        "grad_scale"
                    ]
                ),
            }
        )

        if (
            validation_score
            > best_validation_score
        ):
            best_validation_score = (
                validation_score
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            save_model_state(
                model=model,
                checkpoint_path=checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_epoch is None:
        raise RuntimeError(
            "No valid checkpoint was produced."
        )

    load_model_state(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return {
        "history": history,
        "best_validation_f1_macro": float(
            best_validation_score
        ),
        "best_epoch": int(
            best_epoch
        ),
        "epochs_trained": int(
            len(history)
        ),
        "training_steps": int(
            executed_training_steps
        ),
        "skipped_training_steps": int(
            skipped_training_steps
        ),
        "planned_training_steps": int(
            planned_training_steps
        ),
        "warmup_steps": int(
            warmup_steps
        ),
    }