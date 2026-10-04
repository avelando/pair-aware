import math

from src.config import METHODS, PAIR_LOSS_WEIGHT


def validate_pair_loss_weight(value):
    if isinstance(value, bool):
        raise ValueError("Pair loss weight must be a finite nonnegative number.")

    try:
        weight = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Pair loss weight must be a finite nonnegative number."
        ) from error

    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("Pair loss weight must be a finite nonnegative number.")

    return 0.0 if weight == 0.0 else weight


def resolve_pair_loss_weight(method, pair_loss_weight=None):
    if method not in METHODS:
        raise ValueError(f"Invalid experiment method: {method}.")

    if method == "instance_level":
        if pair_loss_weight is not None:
            raise ValueError("Instance-level experiments cannot use a pair loss weight.")
        return None

    return validate_pair_loss_weight(
        PAIR_LOSS_WEIGHT if pair_loss_weight is None else pair_loss_weight
    )


def format_pair_loss_weight(value):
    return str(validate_pair_loss_weight(value)).replace(".", "p")
