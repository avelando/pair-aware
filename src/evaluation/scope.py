from src.config import SPLIT_NAMES


EVALUATION_SCOPES = ("full", "validation")


def validate_evaluation_scope(value):
    if value not in EVALUATION_SCOPES:
        raise ValueError(f"Invalid evaluation scope: {value}.")
    return value


def get_evaluation_split_names(evaluation_scope):
    evaluation_scope = validate_evaluation_scope(evaluation_scope)
    return ("validation", "test") if evaluation_scope == "full" else ("validation",)


def get_training_split_names(evaluation_scope):
    return ("train", *get_evaluation_split_names(evaluation_scope))


def validate_split_names(split_names):
    split_names = tuple(split_names)
    if not split_names or len(set(split_names)) != len(split_names):
        raise ValueError("Split names must be nonempty and unique.")
    if not set(split_names).issubset(SPLIT_NAMES):
        raise ValueError(f"Invalid split names: {split_names}.")
    return split_names
