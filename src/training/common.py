import gc

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import (
    LEARNING_RATE,
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


def create_optimizer_and_scheduler(
    model,
    steps_per_epoch,
):
    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch must be greater than zero."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
        enabled=uses_amp(device),
    )


def autocast_context(device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=uses_amp(device),
    )


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