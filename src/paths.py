from pathlib import Path

from src.config import EXPERIMENT_VERSION, TRAINING_PRECISION
from src.evaluation.scope import validate_evaluation_scope
from src.pair_loss import format_pair_loss_weight, resolve_pair_loss_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
PAIR_CONTROLLED_DATA_ROOT = DATA_ROOT / "pair_controlled"

RESULTS_ROOT = (
    PROJECT_ROOT
    / f"results_{EXPERIMENT_VERSION}"
    / TRAINING_PRECISION
)
SUMMARY_PATH = RESULTS_ROOT / "summary.csv"
SUMMARY_BY_SPLIT_PATH = RESULTS_ROOT / "summary_by_split.csv"
SUMMARY_BY_METHOD_PATH = RESULTS_ROOT / "summary_by_method.csv"

STATISTICS_DESCRIPTIVE_PATH = (
    RESULTS_ROOT
    / "statistics_descriptive.csv"
)

STATISTICS_OMNIBUS_PATH = (
    RESULTS_ROOT
    / "statistics_omnibus.csv"
)

STATISTICS_PAIRWISE_PATH = (
    RESULTS_ROOT
    / "statistics_pairwise.csv"
)

STATISTICS_METADATA_PATH = (
    RESULTS_ROOT
    / "statistics_metadata.json"
)


def get_split_dir(split_seed):
    return PAIR_CONTROLLED_DATA_ROOT / f"seed_{split_seed}"


def get_run_dir(method, split_seed, model_seed, pair_loss_weight=None, evaluation_scope="full"):
    pair_loss_weight = resolve_pair_loss_weight(method, pair_loss_weight)
    evaluation_scope = validate_evaluation_scope(evaluation_scope)
    root = RESULTS_ROOT if evaluation_scope == "full" else RESULTS_ROOT / "validation_only"
    method_dir = root / method

    if pair_loss_weight is not None:
        method_dir = method_dir / f"lambda_{format_pair_loss_weight(pair_loss_weight)}"

    return (
        method_dir
        / f"split_{split_seed}"
        / f"model_seed_{model_seed}"
    )
