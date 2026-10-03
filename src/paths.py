from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
PAIR_CONTROLLED_DATA_ROOT = DATA_ROOT / "pair_controlled"

RESULTS_ROOT = PROJECT_ROOT / "results"
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

LOGS_ROOT = PROJECT_ROOT / "logs"

NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"
REFERENCE_NOTEBOOKS_ROOT = NOTEBOOKS_ROOT / "reference"


def get_split_dir(split_seed):
    return PAIR_CONTROLLED_DATA_ROOT / f"seed_{split_seed}"


def get_run_dir(method, split_seed, model_seed):
    return (
        RESULTS_ROOT
        / method
        / f"split_{split_seed}"
        / f"model_seed_{model_seed}"
    )


def get_log_path(method, split_seed, model_seed):
    return (
        LOGS_ROOT
        / method
        / f"split_{split_seed}"
        / f"model_seed_{model_seed}.log"
    )