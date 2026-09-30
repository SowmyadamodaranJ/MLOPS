"""
mlflow_tracker.py
-----------------
Modular MLflow experiment tracking and model registry integration for
the Smart Factory Predictive Maintenance pipeline.

Responsibilities
----------------
- Create / set the "Smart_Factory_Predictive_Maintenance" experiment.
- Log every training run with:
    • Model name (tag)
    • Hyperparameters
    • Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, etc.)
    • Confusion matrix and curve plots
- Log additional artifacts for the best model:
    • best_model.pkl
    • preprocessor.pkl (if scaler exists)
    • evaluation_report.csv
    • feature_importance.png (if available)
- Save trained sklearn pipelines as MLflow model artifacts.
- Register the best-performing model in the MLflow Model Registry.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    Manages MLflow experiment tracking and model registration.

    Parameters
    ----------
    config : dict, optional
        Project configuration. Loaded automatically if not provided.
    """

    def __init__(self, config: dict = None) -> None:
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is not installed. Run: pip install mlflow"
            )
        self.config = config or load_config()
        self.mlflow_cfg = self.config.get("mlflow", {})
        self.experiment_name: str = self.mlflow_cfg.get(
            "experiment_name", "Smart_Factory_Predictive_Maintenance"
        )
        self.tracking_uri: str = self.mlflow_cfg.get(
            "tracking_uri", "mlruns"
        )
        self.registry_model_name: str = "PredictiveMaintenanceModel"
        self.figures_dir = Path(self.config["paths"]["figures_dir"])
        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])
        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.plots_dir = self.figures_dir.parent / "plots"

        self._run_ids: Dict[str, str] = {}
        self._metrics: Dict[str, Dict[str, float]] = {}

        self._setup_mlflow()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking URI and experiment."""
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info("MLflow tracking URI  : %s", self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(
                "Created MLflow experiment: '%s' (id=%s)",
                self.experiment_name, experiment_id,
            )
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                "Using existing MLflow experiment: '%s' (id=%s)",
                self.experiment_name, experiment_id,
            )
        mlflow.set_experiment(self.experiment_name)

    # ── Tag helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_run_tags(model_name: str) -> Dict[str, str]:
        """
        Build the full tag dictionary for one MLflow run.

        Includes project-level tags plus the per-run Algorithm tag.
        """
        return {
            "Project": "Smart Factory Predictive Maintenance",
            "Version": "2.0.0",
            "Framework": "Scikit-learn / XGBoost",
            "Dataset": "Microsoft Azure Predictive Maintenance Dataset",
            "SplitStrategy": "Chronological (70:30)",
            "CVStrategy": "TimeSeriesSplit (5 Folds)",
            "Environment": "Development",
            "Algorithm": model_name,
        }

    # ── Hyperparameter extraction ─────────────────────────────────────────────

    @staticmethod
    def _extract_hyperparameters(pipe: Pipeline) -> Dict[str, Any]:
        """
        Extract hyperparameters from the estimator step of a sklearn Pipeline.

        Returns
        -------
        dict
            Flat dictionary of hyperparameter name-value pairs.
        """
        clf = pipe.named_steps.get("clf")
        if clf is None:
            return {}
        params = clf.get_params()
        # Filter out non-serialisable / internal params
        clean = {}
        for k, v in params.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                clean[k] = v
        return clean

    # ── Artifact helpers ──────────────────────────────────────────────────────

    def _log_run_artifacts(self, model_name: str, is_best: bool) -> None:
        """
        Log specific artifacts for the current run, and best model artifacts if is_best.
        """
        safe_name = model_name.lower().replace(" ", "_")
        search_dirs = [self.plots_dir, self.figures_dir]

        # Log combined curves/plots
        combined_plots = [
            ("roc_curves.png", "ROC Curve"),
            ("pr_curves.png", "Precision Recall Curve"),
            ("confusion_matrices.png", "Confusion Matrix"),
        ]
        for plot_file, label in combined_plots:
            for s_dir in search_dirs:
                path = s_dir / plot_file
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path="plots")
                    logger.info("    Logged combined plot: %s as %s", plot_file, label)
                    break

        # Log model-specific feature importance plots
        for s_dir in search_dirs:
            if s_dir.exists():
                for path in s_dir.glob("*.png"):
                    if safe_name in path.name.lower() and "feature_importance" in path.name.lower():
                        mlflow.log_artifact(str(path), artifact_path="plots")
                        logger.info("    Logged feature importance plot: %s", path.name)

        # Log SHAP plots for XGBoost/Random Forest
        if model_name in ["XGBoost", "Random Forest"]:
            shap_types = ["summary", "bar", "waterfall", "dependence"]
            for s_type in shap_types:
                for s_dir in search_dirs:
                    if s_dir.exists():
                        for path in s_dir.glob(f"*shap_{s_type}*_{safe_name}.png"):
                            mlflow.log_artifact(str(path), artifact_path="plots/shap")
                            logger.info("    Logged SHAP %s plot: %s", s_type, path.name)
                        for path in s_dir.glob(f"shap_{s_type}_{safe_name}.png"):
                            mlflow.log_artifact(str(path), artifact_path="plots/shap")
                            logger.info("    Logged SHAP %s plot: %s", s_type, path.name)

        if is_best:
            logger.info("    '%s' is best model — logging extra artifacts …", model_name)
            
            reports_dir = Path(self.config["paths"]["reports_dir"])
            global_artifacts = [
                (self.models_dir / "best_model.pkl", "models"),
                (self.models_dir / "preprocessor.pkl", "models"),
                (self.models_dir / "feature_names.pkl", "models"),
                (self.models_dir / "label_encoder.pkl", "models"),
                (reports_dir / "training_report.html", "reports"),
                (self.metrics_dir / "evaluation_metrics.csv", "reports/metrics"),
                (self.metrics_dir / "classification_report.csv", "reports/metrics"),
                (reports_dir / "hyperparameter_results.csv", "reports"),
                (self.metrics_dir / "feature_importance.csv", "reports/metrics"),
            ]
            
            for path, dest_dir in global_artifacts:
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path=dest_dir)
                    logger.info("    Logged global artifact: %s to %s", path.name, dest_dir)
                else:
                    logger.warning("    Global artifact missing: %s", path)

    # ── Single-model tracking ─────────────────────────────────────────────────

    def track_model(
        self,
        model_name: str,
        pipe: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        tuner: Any,
        is_best: bool = False,
        metrics_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Log one training run to MLflow.

        Parameters
        ----------
        model_name : str
            Human-readable model name (e.g. "Random Forest").
        pipe : Pipeline
            Fitted sklearn Pipeline.
        X_train : pd.DataFrame
            Training feature matrix.
        X_test : pd.DataFrame
            Test feature matrix.
        y_train : pd.Series
            Training labels.
        y_test : pd.Series
            Test labels.
        tuner : HyperparameterTuner
            Fitted hyperparameter tuner containing training details.
        is_best : bool
            If True, log additional best-model artifacts.
        metrics_df : pd.DataFrame, optional
            Full model comparison table (logged as evaluation_report.csv
            if is_best is True).

        Returns
        -------
        str
            The MLflow run_id for this model.
        """
        logger.info("  MLflow: logging run for '%s' …", model_name)

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            # ── Tags ──────────────────────────────────────────────────────────
            tags = self._get_run_tags(model_name)
            mlflow.set_tags(tags)

            # ── Parameters ────────────────────────────────────────────────────
            # 1. Model Name
            mlflow.log_param("Model Name", model_name)
            
            # 2. Hyperparameters (JSON string and individual parameters)
            hyperparams = self._extract_hyperparameters(pipe)
            mlflow.log_param("Hyperparameters", json.dumps(hyperparams, default=str))
            mlflow.log_params(hyperparams)
            
            # 3. Random Seed
            clf = pipe.named_steps.get("clf")
            random_seed = getattr(clf, "random_state", None)
            if random_seed is None:
                random_seed = self.config.get("preprocessing", {}).get("random_state", 42)
            mlflow.log_param("Random Seed", random_seed)
            
            # 4. Feature Count
            mlflow.log_param("Feature Count", X_train.shape[1])
            
            # 5. Training Samples
            mlflow.log_param("Training Samples", X_train.shape[0])
            
            # 6. Testing Samples
            mlflow.log_param("Testing Samples", X_test.shape[0])
            
            # 7. Cross Validation Score
            cv_score = 0.0
            if tuner is not None:
                if hasattr(tuner, "best_scores"):
                    cv_score = tuner.best_scores.get(model_name, 0.0)
                elif hasattr(tuner, "_best_scores"):
                    cv_score = tuner._best_scores.get(model_name, 0.0)
            mlflow.log_param("Cross Validation Score", cv_score)
            
            # 8. Training Time
            train_time = 0.0
            if tuner is not None and hasattr(tuner, "training_times"):
                train_time = tuner.training_times.get(model_name, 0.0)
            mlflow.log_param("Training Time", train_time)
            
            # 9. Inference Time
            infer_time = 0.0
            if metrics_df is not None and not metrics_df.empty:
                model_row = metrics_df[metrics_df["model"] == model_name]
                if not model_row.empty:
                    infer_time = model_row.iloc[0].get("inference_time_ms", 0.0)
            mlflow.log_param("Inference Time", infer_time)
            
            # 10. Optimal Threshold
            optimal_thr = 0.5
            if metrics_df is not None and not metrics_df.empty:
                model_row = metrics_df[metrics_df["model"] == model_name]
                if not model_row.empty:
                    optimal_thr = model_row.iloc[0].get("optimal_threshold", 0.5)
            mlflow.log_param("Optimal Threshold", optimal_thr)

            # 11. Dataset Version
            mlflow.log_param("Dataset Version", "2.0.0 (Chronological)")

            logger.info("    Logged parameters for %s.", model_name)

            # ── Metrics ───────────────────────────────────────────────────────
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1_score_val = 0.0
            roc_auc_val = float("nan")
            balanced_acc = 0.0
            specificity_val = 0.0
            mcc_val = 0.0
            kappa_val = 0.0
            log_loss_val = float("nan")

            if metrics_df is not None and not metrics_df.empty:
                model_row = metrics_df[metrics_df["model"] == model_name]
                if not model_row.empty:
                    row_dict = model_row.iloc[0].to_dict()
                    accuracy = row_dict.get("accuracy", 0.0)
                    precision = row_dict.get("precision", 0.0)
                    recall = row_dict.get("recall", 0.0)
                    f1_score_val = row_dict.get("f1_score", 0.0)
                    roc_auc_val = row_dict.get("roc_auc", float("nan"))
                    balanced_acc = row_dict.get("balanced_accuracy", 0.0)
                    specificity_val = row_dict.get("specificity", 0.0)
                    mcc_val = row_dict.get("mcc", 0.0)
                    kappa_val = row_dict.get("cohen_kappa", 0.0)
                    log_loss_val = row_dict.get("log_loss", float("nan"))

            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1_score_val)
            if pd.notna(roc_auc_val) and not np.isnan(roc_auc_val):
                mlflow.log_metric("ROC AUC", roc_auc_val)
            mlflow.log_metric("Balanced Accuracy", balanced_acc)
            mlflow.log_metric("Specificity", specificity_val)
            mlflow.log_metric("Matthews Correlation Coefficient", mcc_val)
            mlflow.log_metric("Cohen Kappa", kappa_val)
            if pd.notna(log_loss_val) and not np.isnan(log_loss_val):
                mlflow.log_metric("Log Loss", log_loss_val)

            logger.info(
                "    Logged metrics: acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
                accuracy, precision, recall, f1_score_val,
            )

            # ── Trained model artifact ────────────────────────────────────────
            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="cloudpickle")
            logger.info("    Model artifact saved (run %s).", run_id)

            # ── Artifacts and Plots ───────────────────────────────────────────
            self._log_run_artifacts(model_name, is_best)

        self._run_ids[model_name] = run_id
        # Keep metrics for model registry version tagging
        self._metrics[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score_val,
        }
        return run_id

    # ── Track all models ──────────────────────────────────────────────────────

    def track_all_models(
        self,
        models: Dict[str, Pipeline],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        tuner: Any,
        best_name: str,
        metrics_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """
        Log every trained model as a separate MLflow run.

        Parameters
        ----------
        models : dict
            Trained sklearn Pipeline objects keyed by model name.
        X_train : pd.DataFrame
            Training features.
        X_test : pd.DataFrame
            Test feature matrix.
        y_train : pd.Series
            Training labels.
        y_test : pd.Series
            Test labels.
        tuner : HyperparameterTuner
            Tuning orchestrator.
        best_name : str
            Name of the best model (gets additional artifacts).
        metrics_df : pd.DataFrame, optional
            Full model comparison table from the evaluator.

        Returns
        -------
        dict
            Mapping of model_name → MLflow run_id.
        """
        logger.info("=" * 60)
        logger.info("STEP 13: MLflow Experiment Tracking")
        logger.info("=" * 60)

        for name, pipe in models.items():
            try:
                is_best = (name == best_name)
                self.track_model(
                    model_name=name,
                    pipe=pipe,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    tuner=tuner,
                    is_best=is_best,
                    metrics_df=metrics_df,
                )
            except Exception as exc:
                logger.error(
                    "  MLflow: failed to log '%s': %s", name, exc,
                    exc_info=True
                )

        logger.info(
            "  MLflow: tracked %d / %d models.",
            len(self._run_ids), len(models),
        )
        return self._run_ids

    # ── Best model registration ───────────────────────────────────────────────

    def register_best_model(
        self,
        best_name: str,
    ) -> Optional[str]:
        """
        Register the best model in the MLflow Model Registry.

        Parameters
        ----------
        best_name : str
            Name of the best model (must have been tracked already).

        Returns
        -------
        str or None
            The registered model version, or None on failure.
        """
        if best_name not in self._run_ids:
            logger.error(
                "  MLflow: '%s' was not tracked; cannot register.", best_name,
            )
            return None

        run_id = self._run_ids[best_name]
        model_uri = f"runs:/{run_id}/model"

        try:
            logger.info("  MLflow: registering '%s' in Model Registry …", best_name)
            result = mlflow.register_model(
                model_uri=model_uri,
                name=self.registry_model_name,
            )
            version = result.version
            logger.info(
                "  ✔ Registered: %s  version=%s",
                self.registry_model_name, version,
            )

            # ── Tag the registered version with metadata ──────────────────────
            client = MlflowClient()
            client.set_model_version_tag(
                name=self.registry_model_name,
                version=version,
                key="best_model_name",
                value=best_name,
            )
            if best_name in self._metrics:
                for metric_name, metric_val in self._metrics[best_name].items():
                    client.set_model_version_tag(
                        name=self.registry_model_name,
                        version=version,
                        key=metric_name,
                        value=str(metric_val),
                    )

            return version

        except Exception as exc:
            logger.error(
                "  MLflow: model registration failed: %s", exc,
            )
            return None

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Log a summary table of all tracked runs."""
        logger.info("")
        logger.info("═" * 60)
        logger.info("  MLflow Tracking Summary")
        logger.info("═" * 60)
        logger.info("  Experiment : %s", self.experiment_name)
        logger.info(
            "  %-22s %-36s", "Model", "Run ID",
        )
        logger.info("  " + "-" * 58)
        for name, rid in self._run_ids.items():
            logger.info("  %-22s %s", name, rid)
        logger.info("═" * 60)


# ── Convenience wrapper ──────────────────────────────────────────────────────

def track_experiments(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tuner: Any,
    best_name: str,
    metrics_df: Optional[pd.DataFrame] = None,
    config: dict = None,
) -> MLflowTracker:
    """
    Convenience function to track all models and register the best one.
    """
    tracker = MLflowTracker(config=config)
    tracker.track_all_models(
        models=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tuner=tuner,
        best_name=best_name,
        metrics_df=metrics_df,
    )
    tracker.register_best_model(best_name)
    tracker.print_summary()
    return tracker

