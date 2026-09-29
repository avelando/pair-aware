import gc

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import (
    AMP_GROWTH_INTERVAL,
    AMP_INIT_SCALE,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    NUM_EPOCHS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.training.reproducibility import create_torch_generator


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the planned experiments."
        )

    return torch.device("cuda")


def uses_amp(device):
    return device.type == "cuda"


def create_dataloader(
    dataset,
    batch_size,
    shuffle,
    seed=None,
):
    if shuffle and seed is None:
        raise ValueError(
            "A seed is required when shuffle=True."
        )

    generator = (
        create_torch_generator(seed)
        if shuffle
        else None
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(
    batch,
    device,
    excluded_keys=(),
):
    excluded_keys = set(
        excluded_keys
    )

    return {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
        if key not in excluded_keys
    }


def _create_optimizer_parameter_groups(model):
    no_decay_parameter_ids = set()

    for module in model.modules():
        if isinstance(
            module,
            torch.nn.LayerNorm,
        ):
            for parameter in module.parameters(
                recurse=False
            ):
                no_decay_parameter_ids.add(
                    id(parameter)
                )

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if (
            name.endswith("bias")
            or id(parameter)
            in no_decay_parameter_ids
        ):
            no_decay_parameters.append(
                parameter
            )
        else:
            decay_parameters.append(
                parameter
            )

    return [
        {
            "params": decay_parameters,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]


def create_optimizer_and_scheduler(
    model,
    steps_per_epoch,
):
    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch must be greater than zero."
        )

    optimizer = torch.optim.AdamW(
        _create_optimizer_parameter_groups(
            model
        ),
        lr=LEARNING_RATE,
    )

    total_steps = (
        steps_per_epoch
        * NUM_EPOCHS
    )

    warmup_steps = int(
        total_steps
        * WARMUP_RATIO
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    return (
        optimizer,
        scheduler,
        total_steps,
        warmup_steps,
    )


def create_grad_scaler(device):
    return torch.amp.GradScaler(
        device.type,
        init_scale=AMP_INIT_SCALE,
        growth_interval=AMP_GROWTH_INTERVAL,
        enabled=uses_amp(device),
    )


def autocast_context(device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=uses_amp(device),
    )


def backward_and_step(
    loss,
    model,
    optimizer,
    scheduler,
    scaler,
):
    scale_before = float(
        scaler.get_scale()
    )

    scaler.scale(
        loss
    ).backward()

    scaler.unscale_(
        optimizer
    )

    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )
    )

    scaler.step(
        optimizer
    )

    scaler.update()

    scale_after = float(
        scaler.get_scale()
    )

    step_skipped = (
        scale_after
        < scale_before
    )

    if not step_skipped:
        scheduler.step()

    return {
        "gradient_norm": float(
            gradient_norm.detach().cpu()
        ),
        "scale_before": scale_before,
        "scale_after": scale_after,
        "step_skipped": bool(
            step_skipped
        ),
    }


def predict_instances(
    model,
    dataloader,
    device,
):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]

            inputs = move_batch_to_device(
                batch,
                device,
                excluded_keys=("labels",),
            )

            with autocast_context(device):
                outputs = model(
                    **inputs
                )

            all_logits.append(
                outputs.logits
                .detach()
                .cpu()
            )

            all_labels.append(
                labels.detach().cpu()
            )

    if not all_logits:
        raise ValueError(
            "Prediction dataloader is empty."
        )

    logits = torch.cat(
        all_logits,
        dim=0,
    ).numpy()

    labels = torch.cat(
        all_labels,
        dim=0,
    ).numpy()

    return logits, labels


def save_model_state(
    model,
    checkpoint_path,
):
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )


def load_model_state(
    model,
    checkpoint_path,
    device,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )


def clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()