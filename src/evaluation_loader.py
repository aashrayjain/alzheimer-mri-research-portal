"""Loader and parser for saved validation metrics, reports, and confusion matrices."""
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.config import (
    VAL_COMPARISON_CSV_PATH,
    VAL_EVAL_DETAILS_JSON_PATH,
    load_json_file,
    DEFAULT_CLASSES
)


def load_model_comparison_df() -> pd.DataFrame:
    """Loads validation comparison CSV and cleans column names."""
    if not VAL_COMPARISON_CSV_PATH.is_file():
        raise FileNotFoundError(f"Missing validation comparison CSV: {VAL_COMPARISON_CSV_PATH}")
    df = pd.read_csv(VAL_COMPARISON_CSV_PATH)
    return df


def load_evaluation_details() -> Dict[str, Any]:
    """Loads detailed validation metrics and per-class reports."""
    return load_json_file(VAL_EVAL_DETAILS_JSON_PATH)


def get_confusion_matrix_for_model(
    model_name: str,
    normalize: bool = False
) -> np.ndarray:
    """Retrieves and optionally normalizes 4x4 confusion matrix."""
    details = load_evaluation_details()
    if model_name not in details:
        raise KeyError(f"No evaluation details found for: {model_name}")

    raw_cm = np.array(details[model_name]["confusion_matrix"], dtype=float)
    if normalize:
        row_sums = raw_cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        return np.round((raw_cm / row_sums) * 100, 2)
    return raw_cm.astype(int)


def get_classification_report_df(model_name: str) -> pd.DataFrame:
    """Returns classification report formatted as a pandas DataFrame."""
    details = load_evaluation_details()
    if model_name not in details:
        raise KeyError(f"No classification report found for: {model_name}")

    report = details[model_name]["classification_report"]
    rows = []
    for key, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append({
                "Class / Metric": key,
                "Precision": round(metrics.get("precision", 0) * 100, 2),
                "Recall": round(metrics.get("recall", 0) * 100, 2),
                "F1-Score": round(metrics.get("f1-score", 0) * 100, 2),
                "Support": int(metrics.get("support", 0)),
            })
    return pd.DataFrame(rows)

