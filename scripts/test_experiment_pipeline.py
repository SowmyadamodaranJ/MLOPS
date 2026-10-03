"""
test_experiment_pipeline.py
Tests the execution of the 4-model training workflow on authentic data.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

print("1. Checking raw data files...")
telemetry_path = RAW_DIR / "PdM_telemetry.csv"
machines_path = RAW_DIR / "PdM_machines.csv"
failures_path = RAW_DIR / "PdM_failures.csv"

print(f"Telemetry exists: {telemetry_path.exists()} ({telemetry_path.stat().st_size / 1e6:.1f} MB)")
print(f"Machines exists: {machines_path.exists()}")
print(f"Failures exists: {failures_path.exists()}")
