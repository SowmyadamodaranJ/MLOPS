"""
constants.py
------------
Shared constants and type aliases used across the Smart Factory PdM project.

Centralizes values that were previously duplicated across multiple modules
(data_cleaner, feature_engineering, eda, model_evaluator, mlflow_tracker).

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

from typing import Dict, List

# ── Sensor Columns ────────────────────────────────────────────────────────────
# The four telemetry sensor readings from the Azure PdM dataset.
SENSOR_COLS: List[str] = ["volt", "rotate", "pressure", "vibration"]

# ── Target Column ─────────────────────────────────────────────────────────────
TARGET_COL: str = "failure_label"

# ── Non-Feature Columns ──────────────────────────────────────────────────────
# Columns excluded from the feature matrix during model training.
NON_FEATURE_COLS: List[str] = ["datetime", "machineID", TARGET_COL]

# ── Metric Names ──────────────────────────────────────────────────────────────
METRIC_NAMES: List[str] = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

# ── Classification Labels ────────────────────────────────────────────────────
CLASS_LABELS: List[str] = ["No Failure", "Failure"]
CLASS_LABEL_MAP: Dict[int, str] = {0: "No Failure", 1: "Failure"}

# ── Chart Styling ─────────────────────────────────────────────────────────────
CHART_PALETTE: Dict[str, str] = {
    "no_failure": "#4CAF50",
    "failure": "#F44336",
}
CHART_COLOURS: List[str] = ["#4361EE", "#F72585", "#7209B7", "#3A0CA3", "#4CC9F0"]
FIG_DPI: int = 150

# ── Expected Schema per Raw File ─────────────────────────────────────────────
EXPECTED_COLUMNS: Dict[str, List[str]] = {
    "telemetry": ["datetime", "machineID", "volt", "rotate", "pressure", "vibration"],
    "errors":    ["datetime", "machineID", "errorID"],
    "failures":  ["datetime", "machineID", "failure"],
    "maint":     ["datetime", "machineID", "comp"],
    "machines":  ["machineID", "model", "age"],
}

# ── Project Metadata ─────────────────────────────────────────────────────────
PROJECT_NAME: str = "Smart Factory Predictive Maintenance"
PROJECT_VERSION: str = "2.0.0"

# ── MLflow Tags ───────────────────────────────────────────────────────────────
MLFLOW_PROJECT_TAGS: Dict[str, str] = {
    "Project": PROJECT_NAME,
    "Version": PROJECT_VERSION,
    "Stage": "Development",
    "Dataset": "Microsoft Azure Predictive Maintenance Dataset",
}
