MODEL_NAME = "neuralmind/bert-large-portuguese-cased"

SPLIT_SEEDS = (13, 21, 40, 42, 73, 101)
MODEL_SEEDS = (13, 21, 40, 42, 73, 101)

SPLIT_NAMES = ("train", "validation", "test")

METHODS = (
    "instance_level",
    "true_pair",
    "shuffled_pair",
)

PAIRING_STRATEGIES = (
    "true_pair",
    "shuffled_pair",
)

EXPECTED_SPLIT_COUNTS = {
    "train": 3990,
    "validation": 570,
    "test": 1140,
}

EXPECTED_PAIR_COUNTS = {
    "train": 1995,
    "validation": 285,
    "test": 570,
}

EXPECTED_CLASS_COUNTS = {
    "train": {
        0: 1995,
        1: 1995,
    },
    "validation": {
        0: 285,
        1: 285,
    },
    "test": {
        0: 570,
        1: 570,
    },
}

MAX_LENGTH = 256
NUM_EPOCHS = 6
LEARNING_RATE = 2e-5

INSTANCE_TRAIN_BATCH_SIZE = 8
PAIR_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 2

AMP_INIT_SCALE = 512.0
AMP_GROWTH_INTERVAL = 10000

PAIR_LOSS_WEIGHT = 1.0

KEEP_CHECKPOINTS = False
SKIP_COMPLETED_RUNS = True

DEFAULT_MAX_PARALLEL = 1

METRIC_FOR_BEST_MODEL = "f1_macro"
LR_SCHEDULER_TYPE = "linear"

ID2LABEL = {
    0: "0",
    1: "1",
}

LABEL2ID = {
    "0": 0,
    "1": 1,
}