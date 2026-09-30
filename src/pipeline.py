"""
pipeline.py
-----------
Top-level orchestrator — Phase 1 + Phase 2 ML Pipeline.

Stages
------
  Stage 1  → Data Loading & Validation    (DataLoader + DataValidator)
  Stage 2  → Dataset Merging              (DataMerger)
  Stage 3  → Data Cleaning               (DataCleaner)
  Stage 4  → Feature Engineering         (FeatureEngineer)
  Stage 5  → Exploratory Data Analysis   (EDAAnalyzer)
  Stage 6  → Model Training & Tuning     (ModelTrainer + HyperparameterTuner)
  Stage 7  → Enhanced Evaluation         (EnhancedEvaluator)
  Stage 8  → Model Comparison & Select   (ModelComparison)
  Stage 9  → SHAP Explainability         (SHAPExplainer)
  Stage 10 → Feature Importance Analysis (FeatureAnalyzer)
  Stage 11 → HTML Training Report        (ReportGenerator)
  Stage 12 → Artifact Validation         (ArtifactManager)
  Stage 13 → MLflow Tracking & Registry  (MLflowTracker)  [Phase 3]

Run from the project root with:
    python run_pipeline.py

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import sys
import traceback
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Phase 1 — Data & Feature stages (unchanged)
from src.data.data_loader import load_datasets
from src.data.data_merger import merge_datasets
from src.data.data_cleaner import clean_data
from src.features.feature_engineering import engineer_features
from src.features.eda import run_eda

# Phase 2 — ML enhancement stages (new)
from src.models.model_trainer import train_models
from src.evaluation.enhanced_evaluator import EnhancedEvaluator
from src.evaluation.model_comparison import ModelComparison
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.evaluation.shap_explainer import SHAPExplainer
from src.evaluation.feature_analysis import FeatureAnalyzer
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.artifact_manager import ArtifactManager
from src.utils.constants import NON_FEATURE_COLS

# MLflow (Phase 3 — wrapped so pipeline does not fail if DB issue arises)
from src.mlflow_tracking.mlflow_tracker import track_experiments

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


import time
from src.data.data_validator import DataValidator

def run_phase1_pipeline() -> None:
    """
    Execute the complete Phase 1 + Phase 2 ML pipeline end-to-end.

    Raises
    ------
    SystemExit
        On any unrecoverable error, logs the traceback and exits with code 1.
    """
    pipeline_start = time.time()
    logger.info("╔═══════════════════════════════════════════════════════════════╗")
    logger.info("║  SMART FACTORY PREDICTIVE MAINTENANCE – PHASE 2 ML PIPELINE  ║")
    logger.info("╚═══════════════════════════════════════════════════════════════╝")

    try:
        # ── Load config ───────────────────────────────────────────────────────
        config = load_config()
        logger.info(
            "Project: %s  |  Version: %s  |  Phase: %s",
            config["project"]["name"],
            config["project"]["version"],
            config["project"].get("phase", "Phase 2"),
        )

        # ── Stage 1 : Load & Validate ──────────────────────────────────────────
        telemetry, errors, failures, maint, machines = load_datasets(config)

        # ── Stage 2 : Merge ────────────────────────────────────────────────────
        merged_df = merge_datasets(telemetry, errors, failures, maint, machines, config=config)

        # ── Stage 3 : Clean ────────────────────────────────────────────────────
        cleaned_df = clean_data(merged_df, config=config)

        # ── Stage 4 : Feature Engineering ─────────────────────────────────────
        features_df = engineer_features(cleaned_df, config=config)

        # ── Stage 5 : EDA ──────────────────────────────────────────────────────
        run_eda(cleaned_df, features_df, config=config)

        # ── Stage 6 : Model Training & Tuning ─────────────────────────────────
        train_start = time.time()
        models, X_train, X_test, y_train, y_test, tuner = train_models(
            features_df, config=config
        )
        training_duration = time.time() - train_start

        # ── Stage 7 : Enhanced evaluation (12 metrics + 6 plot types) ─────────
        eval_start = time.time()
        evaluator = EnhancedEvaluator(config=config, training_times=tuner.training_times)
        metrics_df, _ = evaluator.evaluate(models, X_train, X_test, y_train, y_test)
        eval_duration = time.time() - eval_start

        # ── Stage 8a : Threshold Optimization (PR-curve, F-beta) ──────────────
        thr_optimizer = ThresholdOptimizer(config=config)
        optimal_thresholds = thr_optimizer.optimize_all(models, X_test, y_test)

        # ── Stage 8b : Model comparison & best model selection ──────────────
        feature_names = [c for c in X_train.columns if c not in NON_FEATURE_COLS]
        comparator = ModelComparison(config=config)
        best_name, best_pipe, comparison_df = comparator.run(
            models, metrics_df, feature_names, optimal_thresholds
        )

        # ── Stage 9 : SHAP Explainability ──────────────────────────────────────
        shap_explainer = SHAPExplainer(config=config)
        local_explanations = shap_explainer.explain(models, X_train, X_test, y_test)

        # ── Stage 10 : Feature Importance Analysis ─────────────────────────────
        feat_analyzer = FeatureAnalyzer(config=config)
        feat_analyzer.analyze(models, X_train, X_test, y_test, best_name)

        # ── Stage 11 : HTML Training Report ───────────────────────────────────
        dataset_info = {
            "Total Records": f"{len(features_df):,}",
            "Machines": int(features_df["machineID"].nunique()) if "machineID" in features_df.columns else "N/A",
            "Features": len(feature_names),
            "Training Rows": f"{len(X_train):,}",
            "Test Rows": f"{len(X_test):,}",
            "Failure Rate": f"{float(y_train.mean()):.4%}",
            "Models Trained": len(models),
        }

        reporter = ReportGenerator(config=config)
        reporter.generate(
            dataset_info=dataset_info,
            feature_names=feature_names,
            best_params=tuner.best_params,
            comparison_df=comparison_df,
            metrics_df=metrics_df,
            best_name=best_name,
            local_explanations=local_explanations,
        )

        # ── Stage 12 : Artifact Validation ─────────────────────────────────────
        artifact_mgr = ArtifactManager(config=config)
        artifact_mgr.validate_and_log()

        # ── Stage 13 : MLflow Tracking & Registry (Phase 3) ──────────────────
        mlflow_run_id = "N/A"
        try:
            tracker = track_experiments(
                models=models,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                tuner=tuner,
                best_name=best_name,
                metrics_df=metrics_df,
                config=config,
            )
            mlflow_run_id = tracker._run_ids.get(best_name, "N/A")
        except Exception as mlflow_exc:
            logger.warning(
                "MLflow tracking failed (non-fatal): %s — pipeline continues.", mlflow_exc
            )

        # ── Comprehensive Validation Suite ─────────────────────────────────────
        models_dir = Path(config["paths"]["models_dir"])
        reports_dir = Path(config["paths"]["reports_dir"])
        validator = DataValidator(config=config)
        validator.run_comprehensive_pipeline_validation(
            features_df=features_df,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            models_dir=models_dir,
            reports_dir=reports_dir,
        )

        # Extract best model metrics for summary
        best_row = metrics_df[metrics_df["model"] == best_name].iloc[0] if not metrics_df.empty else {}
        acc = best_row.get("accuracy", 0.0)
        prec = best_row.get("precision", 0.0)
        rec = best_row.get("recall", 0.0)
        f1 = best_row.get("f1_score", 0.0)
        roc = best_row.get("roc_auc", 0.0)
        opt_thr = optimal_thresholds.get(best_name, 0.5)

        plots_dir = reports_dir / "plots"
        n_plots = len(list(plots_dir.glob("*.png"))) if plots_dir.exists() else 0
        n_reports = len(list(reports_dir.glob("*.html"))) + len(list((reports_dir / "metrics").glob("*.csv"))) if (reports_dir / "metrics").exists() else 1

        artifacts_saved = [
            f.name for f in models_dir.glob("*") if f.is_file()
        ]

        total_duration = time.time() - pipeline_start

        # ── Issue 5 Required Summary Block ─────────────────────────────────────
        logger.info("")
        logger.info("════════════════════════════════════════════════════════════")
        logger.info("               PIPELINE EXECUTION SUMMARY                   ")
        logger.info("════════════════════════════════════════════════════════════")
        logger.info("  Pipeline Status       : SUCCESS")
        logger.info("  Training Time         : %.2f s", training_duration)
        logger.info("  Evaluation Time       : %.2f s", eval_duration)
        logger.info("  Best Model            : %s", best_name)
        logger.info("  Accuracy              : %.4f (%.2f%%)", acc, acc * 100)
        logger.info("  Precision             : %.4f (%.2f%%)", prec, prec * 100)
        logger.info("  Recall                : %.4f (%.2f%%)", rec, rec * 100)
        logger.info("  F1 Score              : %.4f (%.2f%%)", f1, f1 * 100)
        logger.info("  ROC-AUC               : %.4f", roc)
        logger.info("  Threshold             : %.2f", opt_thr)
        logger.info("  Number of Features    : %d", len(feature_names))
        logger.info("  Training Samples      : %d", len(X_train))
        logger.info("  Testing Samples       : %d", len(X_test))
        logger.info("  MLflow Run ID         : %s", mlflow_run_id)
        logger.info("  Saved Artifacts       : %s", ", ".join(artifacts_saved))
        logger.info("  Total Generated Reports: %d", n_reports)
        logger.info("  Total Generated Plots  : %d", n_plots)
        logger.info("  Execution Time        : %.2f s", total_duration)
        logger.info("════════════════════════════════════════════════════════════")

    except FileNotFoundError as exc:
        logger.error("FILE NOT FOUND: %s", exc)
        logger.error("Hint: Make sure all five CSV files are in 'data/raw/'")
        sys.exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("UNEXPECTED ERROR: %s", exc)
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    run_phase1_pipeline()
