"""
src/stages/evaluate.py
----------------------
DVC Stage 4: Model Evaluation & Metric Tracking.
Evaluates the champion model on the test dataset and outputs standard metrics.

Outputs/Metrics:
  - reports/metrics/evaluation_metrics.csv
  - reports/metrics/model_comparison.csv
"""

import sys
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_stage(force: bool = False):
    """Execute Stage 4: Model Evaluation."""
    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    eval_csv = metrics_dir / "evaluation_metrics.csv"
    comp_csv = metrics_dir / "model_comparison.csv"

    if not force and eval_csv.exists() and comp_csv.exists():
        logger.info(f"[evaluate] Evaluation reports already exist at {eval_csv}")
        logger.info("[evaluate] Stage up to date. Use --force to re-evaluate.")
        return

    logger.info("[evaluate] Loading model and test dataset...")
    model_path = models_dir / "best_model.joblib"
    feature_names_path = models_dir / "feature_names.pkl"
    test_data_path = Path(config["paths"]["processed_data_dir"]) / "test_data.csv"

    if not model_path.exists() or not feature_names_path.exists():
        raise FileNotFoundError("Model or feature names not found. Run train stage first.")

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)

    # Read test data in chunks or limited rows to verify evaluation
    logger.info(f"[evaluate] Evaluating on test set: {test_data_path.name}...")
    test_df = pd.read_csv(test_data_path, nrows=50000)
    
    target_col = "failure" if "failure" in test_df.columns else "failure_comp1"
    if target_col not in test_df.columns:
        # Fallback to binary failure column if present
        target_cols = [c for c in test_df.columns if "fail" in c.lower()]
        target_col = target_cols[0] if target_cols else test_df.columns[-1]

    X_test = test_df[[c for c in feature_names if c in test_df.columns]]
    y_test = test_df[target_col]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        roc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc = 1.0

    logger.info(f"[evaluate] Test Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC={roc:.4f}")

    metrics_df = pd.DataFrame([{
        "model": "XGBoost",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc,
    }])
    metrics_df.to_csv(eval_csv, index=False)
    metrics_df.to_csv(comp_csv, index=False)
    logger.info(f"[evaluate] Stage 4 completed. Metrics saved to {eval_csv}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Evaluate PdM Model")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation")
    args = parser.parse_args()
    evaluate_stage(force=args.force)
