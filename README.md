# Pair-Aware Pun Detection

Reproducible experiments for **Portuguese pun detection** with **BERTimbau Large** on pair-controlled splits derived from the Puntuguese corpus.

The current version compares three training strategies:

- `instance_level`: conventional binary classification, where each sentence is trained independently;
- `true_pair`: pair-aware training with the true counterfactual pair `(H_i, N_i)`;
- `shuffled_pair`: pair-aware control using a deterministic derangement `(H_i, N_j)`, with `i != j`.

All experiments are executed **sequentially, one training run at a time**.

## Experimental design

The complete grid contains:

- 3 methods;
- 6 split seeds: `13, 21, 40, 42, 73, 101`;
- 6 model seeds: `13, 21, 40, 42, 73, 101`.

```text
3 methods x 6 split seeds x 6 model seeds = 108 runs
```

The **split seed** determines the controlled train/validation/test organization. The **model seed** controls Python, NumPy, PyTorch, CUDA, model initialization, stochastic training behavior, and training DataLoader order.

For `shuffled_pair`, the pair derangement is determined by the split seed, so all model seeds evaluated on the same split use the same shuffled pair structure.

## Model

```text
Model: neuralmind/bert-large-portuguese-cased
Revision: aa302f6ea73b759f7df9cad58bd272127b67ec28
Task: binary sequence classification
Labels: 0 = non-pun, 1 = pun
```

`Revision` is the exact **Hugging Face repository commit SHA** used to load the model. It is not an authentication token. Pinning the revision ensures that the same model files are used in future executions even if the upstream repository changes.

BERTimbau Large is public, so no Hugging Face authentication token is required for normal execution.

## Training configuration

| Parameter | Value |
|---|---:|
| Maximum sequence length | 256 |
| Maximum epochs | 6 |
| Learning rate | `2e-5` |
| Instance-level train batch size | 8 texts |
| Pair-aware train batch size | 4 pairs = 8 texts |
| Evaluation batch size | 8 |
| Weight decay | 0.01 |
| Warmup ratio | 0.10 |
| LR scheduler | Linear |
| Maximum gradient norm | 1.0 |
| Early-stopping patience | 2 epochs |
| Best-checkpoint metric | Validation macro-F1 |
| Pair-loss weight | 1.0 |
| Mixed precision | FP16 on CUDA |
| Keep final checkpoint | No |

### Objectives

`instance_level` optimizes the standard classification cross-entropy loss.

`true_pair` and `shuffled_pair` optimize:

```text
L = L_classification + lambda * L_pair
```

with `lambda = 1.0`.

For each instance:

```text
score(x) = logit_pun(x) - logit_non_pun(x)
```

For a pair:

```text
margin = score(H) - score(N)
L_pair = softplus(-margin)
```

The pair-aware objective therefore encourages the pun instance to receive a larger pun score than its paired non-pun counterpart.

## Dataset

The controlled data are stored under:

```text
data/
└── pair_controlled/
    ├── seed_13/
    ├── seed_21/
    ├── seed_40/
    ├── seed_42/
    ├── seed_73/
    ├── seed_101/
    └── summary.csv
```

Each seed directory contains:

```text
train.jsonl
validation.jsonl
test.jsonl
metadata.json
inspection.csv
pair_matrix.csv
```

Expected split sizes:

| Split | Instances | Pairs | Pun | Non-pun |
|---|---:|---:|---:|---:|
| Train | 3,990 | 1,995 | 1,995 | 1,995 |
| Validation | 570 | 285 | 285 | 285 |
| Test | 1,140 | 570 | 570 | 570 |

Before training, the code validates split sizes, class counts, IDs, H/N suffixes, token labels, complete pairs, and the absence of record or pair overlap across train, validation, and test.

## Execution environment

The experiments are being executed on the following environment:

```text
Operating system: Ubuntu 26.04 LTS (Resolute Raccoon)
Kernel: Linux 7.0.0-31-generic
Architecture: x86_64

CPU: AMD Ryzen 9 9900X 12-Core Processor
Physical cores: 12
Logical CPUs: 24

RAM: 125 GiB
Swap: 8 GiB

Workspace filesystem: 3.6 TiB
Available workspace observed before training: ~3.1 TiB

GPU: NVIDIA GeForce RTX 5090
VRAM: 32,607 MiB (31.36 GiB reported by PyTorch)
NVIDIA driver: 595.84
CUDA compatibility reported by NVIDIA driver: 13.2
PyTorch CUDA runtime: 13.0

Python: 3.12.14
PyTorch: 2.13.0+cu130
Transformers: 5.15.0
Accelerate: 1.14.0
NumPy: 2.5.3
pandas: 2.3.2
scikit-learn: 1.9.0
SciPy: 1.17.0
```

The execution container reports no CPU or memory cgroup quota and exposes all 24 logical CPUs.

CUDA is required by the current training implementation.

There is **no fixed VRAM reservation or memory cap configured by this project**. PyTorch allocates CUDA memory dynamically according to the needs of each run. Runtime fields such as `peak_allocated_gb` and `peak_reserved_gb` only report observed memory usage; they do not limit the amount of VRAM available to the model.

Only one training run is active at a time.

## Dependencies

The project requires Python `>=3.12,<3.13`.

Pinned direct dependencies:

| Library | Version |
|---|---:|
| accelerate | 1.14.0 |
| ipykernel | 6.30.1 |
| ipython | 9.5.0 |
| numpy | 2.5.3 |
| pandas | 2.3.2 |
| scikit-learn | 1.9.0 |
| scipy | 1.17.0 |
| torch | 2.13.0 |
| transformers | 5.15.0 |

The complete reproducible environment is defined by `pyproject.toml` and `uv.lock`.

## Installation

Clone the repository and switch to the experiment branch:

```bash
git clone https://github.com/avelando/pair-aware.git
cd pair-aware
git switch refactor/python-scripts
```

Install `uv` if necessary:

```bash
python3 -m pip install --user uv
```

Create the environment from the lock file:

```bash
uv sync --frozen
```

Optional activation:

```bash
source .venv/bin/activate
```

## Validation before training

Compile the source:

```bash
python -m compileall -q src tests
```

Run the test suite:

```bash
uv run python -m unittest discover -s tests -v
```

For the current sequential BERTimbau-only version:

```text
Ran 185 tests

OK
```

Inspect the complete grid without training:

```bash
uv run python -m src.experiments.run_grid --dry-run
```

Expected:

```text
Planned runs: 108
```

## Running experiments

### Complete grid

```bash
uv run python -m src.experiments.run_grid     --max-retries 1
```

The default configuration already covers all 108 combinations. Runs are executed sequentially.

### One method

Instance-level only:

```bash
uv run python -m src.experiments.run_grid     --methods instance_level     --max-retries 1
```

True-pair only:

```bash
uv run python -m src.experiments.run_grid     --methods true_pair     --max-retries 1
```

Shuffled-pair only:

```bash
uv run python -m src.experiments.run_grid     --methods shuffled_pair     --max-retries 1
```

### Selected seeds

```bash
uv run python -m src.experiments.run_grid     --methods instance_level     --split-seeds 13 21     --model-seeds 13 40     --max-retries 1
```

### Single run

Instance-level:

```bash
uv run python -m src.experiments.run_instance_level     --split-seed 13     --model-seed 40
```

Pair-aware:

```bash
uv run python -m src.experiments.run_pair_aware     --method true_pair     --split-seed 13     --model-seed 40
```

Replace `true_pair` with `shuffled_pair` when needed.

## Long-running execution

For remote execution, a persistent terminal session such as `tmux` is recommended:

```bash
tmux new -s pair-aware
```

Inside the session:

```bash
cd /workspace/pair-aware
source .venv/bin/activate
set -o pipefail

uv run python -m src.experiments.run_grid     --max-retries 1     2>&1 | tee experiment.log
```

Detach with `Ctrl+B`, then `D`.

Reconnect with:

```bash
tmux attach -t pair-aware
```

## Resume and retries

Completed runs are identified through their artifacts and experiment fingerprint.

If execution is interrupted, run the same command again:

```bash
uv run python -m src.experiments.run_grid     --max-retries 1
```

Runs already completed with the same fingerprint are returned as `skipped`. Only interrupted, failed, or missing runs are executed.

`--max-retries 1` retries a failed run once before proceeding to the next task.

Do **not** use `--force` when resuming a normal interrupted experiment. `--force` intentionally retrains runs that are already complete.

Resume is implemented at the **run level**, not at the epoch level. If a run is interrupted during training, that individual run restarts from epoch 1, while previously completed runs remain untouched.

## Experiment fingerprint

Each run receives an `experiment_id` based on:

- scientific configuration;
- controlled dataset files;
- relevant source files;
- split seed;
- model seed.

This prevents results generated by different versions of the experiment from being silently mixed.

After starting the definitive grid, avoid modifying fingerprinted scientific configuration or training code unless a new experimental version is intentionally being created.

## Outputs

Run artifacts are stored under:

```text
results/
└── METHOD/
    └── split_SPLIT_SEED/
        └── model_seed_MODEL_SEED/
```

A completed run contains:

```text
metadata.json
metrics.json
history.csv
predictions.csv
pair_predictions.csv
progress.json
completed
```

The temporary best checkpoint is removed after a successful run. The `results/` directory is ignored by Git.

## Metrics

Instance-level metrics include:

- accuracy;
- macro and weighted precision, recall, and F1;
- per-class precision, recall, and F1;
- TP, TN, FP, and FN.

Pair-oriented metrics include:

- pair ranking accuracy;
- pair exact match;
- pair ties;
- mean pair margin;
- median pair margin;
- pair-margin standard deviation.

Runtime metadata also records execution duration and observed CUDA peak memory usage.

## Result summaries

Generate summaries:

```bash
uv run python -m src.results.summaries
```

For the final analysis, require the complete grid:

```bash
uv run python -m src.results.summaries --require-complete
```

Outputs:

```text
results/summary.csv
results/summary_by_split.csv
results/summary_by_method.csv
```

## Statistical analysis

Run after the complete grid is available:

```bash
uv run python -m src.results.statistics
```

Default inferential metrics:

```text
f1_macro
pair_ranking_accuracy
pair_exact_match
```

The inferential unit is the **split seed**. Model-seed results are first averaged within each split seed.

The pipeline performs:

- Friedman omnibus test;
- Kendall's W effect size;
- paired two-sided Wilcoxon signed-rank tests;
- Holm correction within each metric;
- paired rank-biserial correlation.

Outputs:

```text
results/statistics_descriptive.csv
results/statistics_omnibus.csv
results/statistics_pairwise.csv
results/statistics_metadata.json
```

The default significance level is `0.05`.

## Reproducibility

Each run records provenance including method, seeds, experiment fingerprint, model name and revision, training configuration, Git state when available, Python/library versions, GPU name, and CUDA runtime.

The training setup seeds Python, NumPy, PyTorch, CUDA, and the training DataLoader. cuDNN deterministic behavior is enabled and benchmarking is disabled.

Exact bitwise equivalence across different GPU architectures, driver versions, CUDA stacks, or library builds should not be assumed.

## Project structure

```text
pair-aware/
├── data/
│   └── pair_controlled/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── models/
│   ├── results/
│   └── training/
├── tests/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## Current scope

This repository version contains the **BERTimbau Large experimental pipeline**. The planned traditional ensemble extension is not included in this version yet.
