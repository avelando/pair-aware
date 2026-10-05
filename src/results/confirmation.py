from pathlib import Path

import pandas as pd

import src.results.selection as screening
from src.config import PAIR_LOSS_WEIGHTS, SCREENING_MODEL_SEEDS, SPLIT_SEEDS, VALIDATION_F1_TOLERANCE
from src.pair_loss import validate_pair_loss_weight


DEFAULT_SELECTION_PATH = screening.SCREENING_REPORT_DIR / "selection.json"


def validate_confirmation_weight(value):
    weight = validate_pair_loss_weight(value)
    if weight not in PAIR_LOSS_WEIGHTS:
        raise ValueError("Confirmation requires a weight from the screening protocol.")
    return weight


def load_screening_selection(selection_path=DEFAULT_SELECTION_PATH):
    selection_path = Path(selection_path)
    selection = screening._read_json(selection_path)
    rule = {
        "selection_split": "validation", "classification_decision": "default",
        "constraint_metric": "f1_macro", "f1_tolerance": VALIDATION_F1_TOLERANCE,
        "objective": "pair_ranking_accuracy",
        "tie_break": ["higher_f1_macro", "lower_pair_loss_weight"],
        "numerical_tolerance": screening.NUMERICAL_TOLERANCE,
        "aggregation": "mean_over_model_seeds_then_equal_mean_over_split_seeds",
    }
    required = {
        "status": "selected", "expected_runs": len(screening._screening_tasks()),
        "completed_runs": len(screening._screening_tasks()),
        "split_seeds": list(SPLIT_SEEDS), "model_seeds": list(SCREENING_MODEL_SEEDS),
        "pair_loss_weights": list(PAIR_LOSS_WEIGHTS), "rule": rule,
        "selection_source_sha256": screening._hash_file(screening.__file__),
    }
    if any(screening._canonical(selection.get(key)) != screening._canonical(value) for key, value in required.items()):
        raise RuntimeError("The selection manifest does not match the current screening protocol.")
    weight = validate_confirmation_weight(selection.get("selected_pair_loss_weight"))
    records = screening.collect_screening_results()
    by_split, by_weight = screening.build_screening_summaries(records)
    expected_weight, by_weight = screening.select_pair_loss_weight(by_weight)
    expected_decision = {
        "selected_pair_loss_weight": expected_weight,
        "best_validation_f1_macro": float(by_weight["f1_macro_mean"].max()),
        "selected_validation_metrics": by_weight.loc[by_weight["selected"]].iloc[0].to_dict(),
        "eligible_pair_loss_weights": by_weight.loc[by_weight["eligible"], "pair_loss_weight"].tolist(),
    }
    if any(screening._canonical(selection.get(key)) != screening._canonical(value) for key, value in expected_decision.items()):
        raise RuntimeError("The saved decision does not match the audited validation results.")
    saved_inputs = selection.get("inputs")
    if not isinstance(saved_inputs, list) or not all(isinstance(row, dict) for row in saved_inputs):
        raise ValueError("The selection manifest must contain its complete input records.")
    expected_inputs = records.drop(columns=list(screening.METRIC_FIELDS)).to_dict("records")
    def normalize(rows):
        return [{key: value for key, value in row.items() if key != "run_dir"} for row in rows]

    if screening._canonical(normalize(saved_inputs)) != screening._canonical(normalize(expected_inputs)):
        raise RuntimeError("Screening inputs have changed since the selection was saved.")
    tables = {
        "runs": (records, "screening_runs.csv"),
        "by_split": (by_split, "screening_by_split.csv"),
        "by_lambda": (by_weight, "screening_by_lambda.csv"),
    }
    outputs = selection.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(tables):
        raise ValueError("The selection manifest must contain all screening output tables.")
    for name, (expected_table, filename) in tables.items():
        entry = outputs[name]
        if not isinstance(entry, dict) or Path(entry.get("path", "")).name != filename:
            raise ValueError(f"Invalid screening output entry: {name}.")
        path = selection_path.parent / filename
        if not path.is_file() or screening._hash_file(path) != entry.get("sha256"):
            raise RuntimeError(f"Screening output is missing or has changed: {path}.")
        saved = pd.read_csv(path)
        if "run_dir" in expected_table:
            expected_table = expected_table.drop(columns="run_dir")
            saved = saved.drop(columns="run_dir")
        try:
            pd.testing.assert_frame_equal(
                saved, expected_table.reset_index(drop=True), check_dtype=False,
                check_exact=False, rtol=1e-10, atol=1e-12,
            )
        except AssertionError as error:
            raise RuntimeError(f"Screening output does not match the audited validation results: {path}.") from error
    return {
        "selected_pair_loss_weight": weight, "selection_path": str(selection_path.resolve()),
        "selection_sha256": screening._hash_file(selection_path),
    }
