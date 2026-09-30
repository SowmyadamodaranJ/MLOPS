"""
model_evaluator.py
------------------
Evaluates trained PdM models and selects the best one.

Metrics computed per model
--------------------------
- Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- ROC-AUC
- Confusion Matrix

Outputs
-------
- Console / log summary table
- metrics/model_comparison.csv
- reports/figures/11_model_comparison_bar.png
- reports/figures/12_roc_curves.png
- reports/figures/13_confusion_matrices.png (2x2 grid)
- reports/figures/14_feature_importance_*.png (tree-based models)
- models/saved_models/best_model.joblib
- models/saved_models/preprocessing_pipeline.joblib

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.utils.config_loader import load_config
from src.utils.constants import CHART_COLOURS as COLOURS, CLASS_LABELS, FIG_DPI
from src.utils.logger import get_logger
from src.utils.metrics import (
    compute_pipeline_metrics,
    get_classification_report,
    compute_roc_data,
    compute_pr_data,
)

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates, compares, and persists trained PdM classification models.

    Parameters
    ----------
    config : dict, optional
        Project configuration. Loaded automatically if not provided.
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or load_config()
        self.eval_cfg = self.config["evaluation"]
        self.primary_metric: str = self.eval_cfg["primary_metric"]
        self.threshold: float = float(self.eval_cfg["threshold"])
        self.figures_dir = Path(self.config["paths"]["figures_dir"])
        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])
        self.models_dir = Path(self.config["paths"]["models_dir"])
        for d in [self.figures_dir, self.metrics_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Metric helpers ────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        name: str,
        pipe: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics for one model.

        Returns
        -------
        dict
            Keys: model, accuracy, precision, recall, f1_score, roc_auc
        """
        metrics, _ = compute_pipeline_metrics(pipe, X_test, y_test)
        metrics["model"] = name
        logger.info(
            "  %-22s  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
            name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["roc_auc"],
        )
        return metrics

    # ── Chart helpers ─────────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.figures_dir / filename
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Chart saved → %s", path)

    def _plot_model_comparison(self, metrics_df: pd.DataFrame) -> None:
        """Grouped bar chart comparing all metrics across models."""
        logger.info("  Plotting model comparison bar chart …")
        metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        models = metrics_df["model"].tolist()
        x = np.arange(len(models))
        width = 0.15

        fig, ax = plt.subplots(figsize=(16, 7))
        for i, metric in enumerate(metric_cols):
            vals = metrics_df[metric].tolist()
            bars = ax.bar(
                x + i * width, vals, width,
                label=metric.upper(), color=COLOURS[i], edgecolor="white"
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison – All Evaluation Metrics", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        fig.tight_layout()
        self._save(fig, "11_model_comparison_bar.png")

    def _plot_roc_curves(
        self,
        models: Dict[str, Pipeline],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Overlay ROC curves for all models."""
        logger.info("  Plotting ROC curves …")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")

        for idx, (name, pipe) in enumerate(models.items()):
            if not hasattr(pipe, "predict_proba"):
                continue
            y_proba = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, linewidth=2, color=COLOURS[idx % len(COLOURS)],
                label=f"{name} (AUC = {roc_auc:.4f})"
            )

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves – All Models", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        self._save(fig, "12_roc_curves.png")

    def _plot_confusion_matrices(
        self,
        models: Dict[str, Pipeline],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """2×n grid of confusion matrices."""
        logger.info("  Plotting confusion matrices …")
        n = len(models)
        ncols = min(2, n)
        nrows = (n + 1) // 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
        axes = np.array(axes).flatten()

        for idx, (name, pipe) in enumerate(models.items()):
            y_pred = pipe.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

            annot = np.array(
                [[f"{cm[r][c]}\n({cm_pct[r][c]:.1f}%)" for c in range(cm.shape[1])]
                 for r in range(cm.shape[0])]
            )

            sns.heatmap(
                cm, ax=axes[idx], annot=annot, fmt="", cmap="Blues",
                cbar=True, linewidths=0.5, linecolor="white",
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"],
            )
            axes[idx].set_title(f"Confusion Matrix – {name}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Predicted", fontsize=10)
            axes[idx].set_ylabel("Actual", fontsize=10)

        # Hide empty subplots
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Confusion Matrices – All Models", fontsize=15, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "13_confusion_matrices.png")

    def _plot_feature_importance(
        self,
        models: Dict[str, Pipeline],
        feature_names: List[str],
    ) -> None:
        """Top-20 feature importance for tree-based models."""
        logger.info("  Plotting feature importances …")
        tree_models = {k: v for k, v in models.items()
                       if k in ("Random Forest", "Decision Tree", "XGBoost")}

        for name, pipe in tree_models.items():
            clf = pipe.named_steps["clf"]
            if not hasattr(clf, "feature_importances_"):
                continue
            importances = clf.feature_importances_
            fi_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(20)
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(
                fi_df["feature"][::-1], fi_df["importance"][::-1],
                color=COLOURS[0], edgecolor="white"
            )
            ax.set_xlabel("Importance Score", fontsize=12)
            ax.set_title(
                f"Top 20 Feature Importances – {name}", fontsize=13, fontweight="bold"
            )
            fig.tight_layout()
            safe_name = name.lower().replace(" ", "_")
            self._save(fig, f"14_feature_importance_{safe_name}.png")

    # ── Best model selection & persistence ───────────────────────────────────

    def _select_best(self, metrics_df: pd.DataFrame) -> str:
        """Return the model name with the highest primary metric score."""
        metric_col = self.primary_metric
        if metric_col not in metrics_df.columns:
            metric_col = "f1_score"
        best_row = metrics_df.loc[metrics_df[metric_col].idxmax()]
        best_name = best_row["model"]
        best_score = best_row[metric_col]
        logger.info(
            "  Best model: %-22s  %s=%.4f",
            best_name, metric_col, best_score,
        )
        return best_name

    def _save_model(self, pipe: Pipeline, name: str, feature_names: List[str]) -> None:
        """Persist the best model pipeline and metadata using joblib."""
        model_path = self.models_dir / "best_model.joblib"
        meta_path = self.metrics_dir / "best_model_meta.json"

        joblib.dump(pipe, model_path)
        logger.info("  Best model saved  → %s", model_path)

        metadata = {
            "model_name": name,
            "feature_names": feature_names,
            "primary_metric": self.primary_metric,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("  Model metadata    → %s", meta_path)

    def _save_all_models(self, models: Dict[str, Pipeline]) -> None:
        """Persist every trained pipeline for later use."""
        for name, pipe in models.items():
            safe = name.lower().replace(" ", "_")
            path = self.models_dir / f"{safe}_pipeline.joblib"
            joblib.dump(pipe, path)
            logger.info("  Saved pipeline    → %s", path)

    # ── Main evaluation runner ────────────────────────────────────────────────

    def evaluate(
        self,
        models: Dict[str, Pipeline],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[str, Pipeline, pd.DataFrame]:
        """
        Evaluate all models, generate comparison charts, and save the best model.

        Parameters
        ----------
        models : dict
            Trained sklearn Pipeline objects keyed by model name.
        X_train, X_test : pd.DataFrame
            Feature matrices.
        y_train, y_test : pd.Series
            Target labels.

        Returns
        -------
        tuple
            (best_model_name, best_pipeline, metrics_dataframe)
        """
        logger.info("=" * 60)
        logger.info("STEP 7: Model Evaluation & Selection")
        logger.info("=" * 60)

        feature_names = list(X_train.columns)

        # ── Compute metrics ─────────────────────────────────────────────────
        all_metrics: List[Dict] = []
        for name, pipe in models.items():
            m = self._compute_metrics(name, pipe, X_test, y_test)
            all_metrics.append(m)

            # Detailed classification report
            y_pred = pipe.predict(X_test)
            report = classification_report(
                y_test, y_pred,
                target_names=["No Failure", "Failure"],
                zero_division=0,
            )
            logger.info("\n  Classification Report – %s:\n%s", name, report)

        metrics_df = pd.DataFrame(all_metrics)

        # ── Save metrics CSV ────────────────────────────────────────────────
        metrics_path = self.metrics_dir / "model_comparison.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("  Model comparison CSV → %s", metrics_path)

        # ── Render charts ───────────────────────────────────────────────────
        self._plot_model_comparison(metrics_df)
        self._plot_roc_curves(models, X_test, y_test)
        self._plot_confusion_matrices(models, X_test, y_test)
        self._plot_feature_importance(models, feature_names)

        # ── Select & save best model ─────────────────────────────────────────
        best_name = self._select_best(metrics_df)
        best_pipe = models[best_name]
        self._save_model(best_pipe, best_name, feature_names)
        self._save_all_models(models)

        logger.info("=" * 60)
        logger.info("PHASE 1 COMPLETE")
        logger.info("  Best Model : %s", best_name)
        logger.info("  Saved to   : %s", self.models_dir / "best_model.joblib")
        logger.info("=" * 60)

        return best_name, best_pipe, metrics_df


def evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict = None,
) -> Tuple[str, Pipeline, pd.DataFrame]:
    """
    Convenience wrapper around ModelEvaluator.evaluate().

    Returns
    -------
    tuple
        (best_model_name, best_pipeline, metrics_dataframe)
    """
    evaluator = ModelEvaluator(config=config)
    return evaluator.evaluate(models, X_train, X_test, y_train, y_test)
