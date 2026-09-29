from pathlib import Path

import numpy as np
import torch

from src.config import (
    EARLY_STOPPING_PATIENCE,
    METRIC_FOR_BEST_MODEL,
    NUM_EPOCHS,
)
from src.evaluation.metrics import evaluate_instance_logits
from src.training.common import (
    autocast_context,
    backward_and_step,
    create_grad_scaler,
    create_optimizer_and_scheduler,
    load_model_state,
    move_batch_to_device,
    predict_instances,
    save_model_state,
)


def train_instance_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
):
    model.train()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0

    for batch in dataloader:
        optimizer.zero_grad(
            set_to_none=True
        )

        batch_size = int(
            batch["labels"].shape[0]
        )

        inputs = move_batch_to_device(
            batch,
            device,
        )

        with autocast_context(device):
            outputs = model(
                **inputs
            )

            loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected during training."
            )

        step_result = backward_and_step(
            loss=loss,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        if step_result["step_skipped"]:
            skipped_optimizer_steps += 1
        else:
            optimizer_steps += 1

        total_loss += (
            float(
                loss.detach().cpu()
            )
            * batch_size
        )

        total_examples += batch_size
        total_batches += 1

    if total_examples == 0:
        raise ValueError(
            "Training dataloader is empty."
        )

    return {
        "loss": (
            total_loss
            / total_examples
        ),
        "examples": total_examples,
        "batches": total_batches,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "grad_scale": float(
            scaler.get_scale()
        ),
    }


def train_instance_level_model(
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
        train_result = train_instance_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
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
                "train_loss": float(
                    train_result["loss"]
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
        "planned_training_steps": int(
            planned_training_steps
        ),
        "warmup_steps": int(
            warmup_steps
        ),
        "skipped_training_steps": int(
            skipped_training_steps
        ),
    }