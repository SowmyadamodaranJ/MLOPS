"""
model_comparison.py
-------------------
Task 5 – Phase 2: Model Comparison Engine

Automatically selects the best-performing model based on the primary metric,
saves best_model.pkl, preprocessor.pkl, feature_names.pkl, and
label_encoder.pkl.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_COMPARISON_COLS = [
    "model", "accuracy", "precision", "recall", "f1_score",
    "roc_auc", "balanced_accuracy", "mcc", "training_time_s",
    "inference_time_ms", "optimal_threshold",
]


class ModelComparison:
    """
    Ranks all trained models, selects the best, and persists artifacts.

    Parameters
    ----------
    config : dict, optional
        Project config. Auto-loaded if not provided.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_config()
        self.eval_cfg = self.config["evaluation"]
        self.primary_metric: str = self.eval_cfg.get("primary_metric", "f1_score")
        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Selection ──────────────────────────────────────────────────────────────

    def select_best(self, metrics_df: pd.DataFrame) -> str:
        """
        Return the model name with the highest primary metric value.

        Falls back to f1_score if configured metric not present.
        """
        metric = self.primary_metric
        if metric not in metrics_df.columns:
            metric = "f1_score"

        best_row = metrics_df.loc[metrics_df[metric].idxmax()]
        best_name = best_row["model"]
        best_score = best_row[metric]
        logger.info(
            "  [Comparison] Best model: %s  (%s=%.4f)",
            best_name, metric, best_score,
        )
        return best_name

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_best_model(
        self,
        best_name: str,
        models: Dict[str, Pipeline],
        feature_names: List[str],
        optimal_thresholds: Optional[Dict[str, float]] = None,
        selected_metric_value: Optional[float] = None,
    ) -> None:
        """
        Persist the best pipeline and its components.

        Saved artifacts
        ---------------
        saved_models/best_model.pkl
        saved_models/preprocessor.pkl   (if scaler step exists)
        saved_models/feature_names.pkl
        saved_models/label_encoder.pkl  (binary target label map)
        reports/metrics/best_model_meta.json  (includes optimal_threshold and selected_metric_value)
        models/saved_models/model_manifest.json (updated with optimal_threshold)
        """
        pipe = models[best_name]

        # best_model.pkl (full pipeline)
        best_path = self.models_dir / "best_model.pkl"
        joblib.dump(pipe, best_path)
        logger.info("  [Comparison] best_model.pkl → %s", best_path)

        # preprocessor.pkl (scaler step if present)
        scaler = pipe.named_steps.get("scaler")
        if scaler is not None:
            prep_path = self.models_dir / "preprocessor.pkl"
            joblib.dump(scaler, prep_path)
            logger.info("  [Comparison] preprocessor.pkl → %s", prep_path)

        # feature_names.pkl
        fn_path = self.models_dir / "feature_names.pkl"
        joblib.dump(feature_names, fn_path)
        logger.info("  [Comparison] feature_names.pkl → %s", fn_path)

        # label_encoder.pkl (binary target, no sklearn LabelEncoder; save mapping)
        le_path = self.models_dir / "label_encoder.pkl"
        joblib.dump({0: "No Failure", 1: "Failure"}, le_path)
        logger.info("  [Comparison] label_encoder.pkl → %s", le_path)

        # Also keep backward-compatible best_model.joblib
        joblib.dump(pipe, self.models_dir / "best_model.joblib")

        # Retrieve the optimal threshold for the best model
        optimal_thr = 0.5  # default
        if optimal_thresholds and best_name in optimal_thresholds:
            optimal_thr = float(optimal_thresholds[best_name])

        # best_model_meta.json — includes optimal_threshold for API use
        meta = {
            "model_name":            best_name,
            "feature_names":         feature_names,
            "primary_metric":        self.primary_metric,
            "selected_metric_value": round(float(selected_metric_value), 4) if selected_metric_value is not None else None,
            "optimal_threshold":     round(float(optimal_thr), 4),
            "split_strategy":        "chronological_70_30",
        }
        meta_path = self.metrics_dir / "best_model_meta.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("  [Comparison] best_model_meta.json → %s", meta_path)

        # Sync with model_manifest.json if present
        manifest_path = self.models_dir / "model_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
                manifest["optimal_threshold"] = round(float(optimal_thr), 4)
                if selected_metric_value is not None:
                    manifest["selected_metric_value"] = round(float(selected_metric_value), 4)
                with open(manifest_path, "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2)
                logger.info("  [Comparison] model_manifest.json synced with optimal threshold: %.4f", optimal_thr)
            except Exception as e:
                logger.warning("Could not update model_manifest.json: %s", e)

        logger.info(
            "  [Comparison] Best model: %s | Metric: %.4f | Optimal threshold: %.4f",
            best_name, selected_metric_value or 0.0, optimal_thr,
        )

    def save_all_models(self, models: Dict[str, Pipeline]) -> None:
        """Persist every pipeline for posterity."""
        for name, pipe in models.items():
            safe = name.lower().replace(" ", "_")
            path = self.models_dir / f"{safe}_pipeline.joblib"
            joblib.dump(pipe, path)
            logger.info("  [Comparison] Saved pipeline → %s", path)

    # ── Comparison table ───────────────────────────────────────────────────────

    def build_comparison_table(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Return a trimmed comparison table with the most relevant columns."""
        available = [c for c in _COMPARISON_COLS if c in metrics_df.columns]
        return metrics_df[available].sort_values(
            self.primary_metric
            if self.primary_metric in metrics_df.columns
            else "f1_score",
            ascending=False,
        ).reset_index(drop=True)

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        models: Dict[str, Pipeline],
        metrics_df: pd.DataFrame,
        feature_names: List[str],
        optimal_thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, Pipeline, pd.DataFrame]:
        """
        Select best model, persist all artifacts, return summary.

        Parameters
        ----------
        models : dict
            Trained sklearn Pipeline objects.
        metrics_df : pd.DataFrame
            Evaluation metrics for all models.
        feature_names : list
            Feature names used during training.
        optimal_thresholds : dict, optional
            Per-model optimal thresholds from ThresholdOptimizer.

        Returns
        -------
        tuple
            (best_model_name, best_pipeline, comparison_table_df)
        """
        logger.info("=" * 60)
        logger.info("STEP 8: Model Comparison & Selection")
        logger.info("=" * 60)

        # Attach optimal thresholds to the comparison table if available
        if optimal_thresholds:
            metrics_df = metrics_df.copy()
            metrics_df["optimal_threshold"] = metrics_df["model"].map(
                lambda m: optimal_thresholds.get(m, 0.5)
            )

        comparison_df = self.build_comparison_table(metrics_df)
        logger.info("\n%s\n", comparison_df.to_string(index=False))

        best_name = self.select_best(metrics_df)
        metric = self.primary_metric if self.primary_metric in metrics_df.columns else "f1_score"
        best_score = float(metrics_df.loc[metrics_df["model"] == best_name, metric].values[0]) if not metrics_df.empty else None
        self.save_best_model(best_name, models, feature_names, optimal_thresholds, selected_metric_value=best_score)
        self.save_all_models(models)

        return best_name, models[best_name], comparison_df
