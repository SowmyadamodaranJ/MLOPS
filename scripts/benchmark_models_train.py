"""
benchmark_models_train.py
Benchmarks live training of Logistic Regression, Decision Tree, Random Forest, and XGBoost
on authentic data features with chronological split.
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
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
print(f"Loaded {len(feature_names)} features.")

# Let's inspect test_data.csv and latest_machine_features.csv
test_csv = PROJECT_ROOT / "data" / "processed" / "test_data.csv"
mini_cache = PROJECT_ROOT / "data" / "processed" / "latest_machine_features.csv"
print(f"Test CSV rows: {len(pd.read_csv(test_csv))}")
print(f"Mini cache rows: {len(pd.read_csv(mini_cache))}")
