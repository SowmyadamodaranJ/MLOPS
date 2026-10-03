"""
test_train_all.py
Tests training all 4 models on authentic features_engineered.csv with chronological split.
"""

import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
features_csv = PROJECT_ROOT / "data" / "processed" / "features_engineered.csv"
feature_names = joblib.load(PROJECT_ROOT / "models" / "saved_models" / "feature_names.pkl")

print(f"Loading features from {features_csv.name}...")
t0 = time.time()
# Read 60,000 rows sampled or chronological
df = pd.read_csv(features_csv, nrows=60000)
print(f"Loaded {len(df)} rows in {time.time() - t0:.2f}s")

# Ensure datetime sorting for chronological split
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# 70/30 chronological split
split_idx = int(len(df) * 0.70)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[feature_names]
y_train = train_df["failure_label"]
X_test = test_df[feature_names]
y_test = test_df["failure_label"]

print(f"Train rows: {len(X_train)} (Failures: {y_train.sum()})")
print(f"Test rows:  {len(X_test)} (Failures: {y_test.sum()})")
print(f"Train max dt: {train_df['datetime'].max()} | Test min dt: {test_df['datetime'].min()}")

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced", random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=60, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "XGBoost": Pipeline([
        ("clf", XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.1, scale_pos_weight=10, random_state=42, eval_metric="logloss", n_jobs=-1))
    ])
}

results = []
for name, pipe in models.items():
    print(f"Training {name}...")
    t_start = time.time()
    pipe.fit(X_train, y_train)
    t_train = time.time() - t_start

    # Evaluate
    t_inf_start = time.time()
    preds = pipe.predict(X_test)
    t_inf = (time.time() - t_inf_start) / len(X_test) * 1000.0  # ms per sample

    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X_test)[:, 1]
    else:
        probs = preds

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    try:
        roc = float(roc_auc_score(y_test, probs))
    except Exception:
        roc = 0.5

    results.append({
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4),
        "training_time_s": round(t_train, 2),
        "inference_latency_ms": round(t_inf, 4)
    })
    print(f"  {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC={roc:.4f}, TrainTime={t_train:.2f}s")

res_df = pd.DataFrame(results)
print("\n=== MODEL COMPARISON TABLE ===")
print(res_df.to_string(index=False))
best_idx = res_df["f1_score"].idxmax()
print(f"\nChampion candidate: {res_df.loc[best_idx, 'model']} with F1={res_df.loc[best_idx, 'f1_score']}")
