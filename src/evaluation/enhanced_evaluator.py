"""
enhanced_evaluator.py
---------------------
Task 2 – Phase 2: Professional Model Evaluation

Metrics computed per model
--------------------------
  Accuracy, Precision, Recall, F1, ROC AUC, Balanced Accuracy,
  Specificity, MCC, Cohen Kappa, Log Loss, Training Time, Inference Time

Plots saved to reports/plots/
------------------------------
  roc_curves.png
  pr_curves.png
  confusion_matrices.png
  calibration_curves.png
  feature_importance_top10.png
  model_comparison_bar.png

Reports saved to reports/
--------------------------
  evaluation_metrics.csv
  classification_report.csv

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.pipeline import Pipeline

from src.utils.config_loader import load_config
from src.utils.constants import CHART_COLOURS as COLOURS, CLASS_LABELS, FIG_DPI
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PLOTS_DIR_NAME = "plots"


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Negative Rate = TN / (TN + FP)."""
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


class EnhancedEvaluator:
    """
    Evaluates and compares trained PdM classification models with a rich
    metric suite and professional plots.

    Parameters
    ----------
    config : dict, optional
        Project config. Auto-loaded if not provided.
    training_times : dict, optional
        Per-model training time in seconds (from HyperparameterTuner).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        training_times: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config = config or load_config()
        self.eval_cfg = self.config["evaluation"]
        self.primary_metric: str = self.eval_cfg["primary_metric"]
        self.threshold: float = float(self.eval_cfg.get("threshold", 0.5))
        self.training_times: Dict[str, float] = training_times or {}

        figures_dir = Path(self.config["paths"]["figures_dir"])
        self.plots_dir = figures_dir.parent / _PLOTS_DIR_NAME
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Metric computation ─────────────────────────────────────────────────────

    def _compute_all_metrics(
        self,
        name: str,
        pipe: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Compute all 12 evaluation metrics for one model."""
        y_pred = pipe.predict(X_test)

        has_proba = hasattr(pipe, "predict_proba")
        y_proba: Optional[np.ndarray] = None
        if has_proba:
            y_proba = pipe.predict_proba(X_test)[:, 1]

        # Inference time (ms per sample)
        t0 = time.perf_counter()
        _ = pipe.predict(X_test[:100] if len(X_test) >= 100 else X_test)
        n_samples = min(100, len(X_test))
        infer_ms = (time.perf_counter() - t0) / n_samples * 1000

        metrics: Dict[str, float] = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(
                precision_score(y_test, y_pred, average="binary", zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_test, y_pred, average="binary", zero_division=0), 4
            ),
            "f1_score": round(
                f1_score(y_test, y_pred, average="binary", zero_division=0), 4
            ),
            "roc_auc": round(
                roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan"),
                4,
            ),
            "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
            "specificity": round(_specificity(np.array(y_test), np.array(y_pred)), 4),
            "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
            "cohen_kappa": round(cohen_kappa_score(y_test, y_pred), 4),
            "log_loss": round(
                log_loss(y_test, y_proba) if y_proba is not None else float("nan"), 4
            ),
            "training_time_s": round(self.training_times.get(name, 0.0), 3),
            "inference_time_ms": round(infer_ms, 4),
        }

        logger.info(
            "  %-22s  acc=%.4f  f1=%.4f  auc=%.4f  bal_acc=%.4f  mcc=%.4f",
            name,
            metrics["accuracy"],
            metrics["f1_score"],
            metrics["roc_auc"],
            metrics["balanced_accuracy"],
            metrics["mcc"],
        )
        return metrics

    # ── Plot helpers ───────────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.plots_dir / filename
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Plot saved → %s", path)

    def _plot_roc(
        self, models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
        for idx, (name, pipe) in enumerate(models.items()):
            if not hasattr(pipe, "predict_proba"):
                continue
            y_prob = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, lw=2.5,
                color=COLOURS[idx % len(COLOURS)],
                label=f"{name}  AUC={roc_auc:.4f}",
            )
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        self._save(fig, "roc_curves.png")

    def _plot_pr(
        self, models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 7))
        baseline = y_test.mean()
        ax.axhline(baseline, color="k", linestyle="--", lw=1,
                   label=f"Random baseline (P={baseline:.3f})")
        for idx, (name, pipe) in enumerate(models.items()):
            if not hasattr(pipe, "predict_proba"):
                continue
            y_prob = pipe.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(rec, prec)
            ax.plot(
                rec, prec, lw=2.5,
                color=COLOURS[idx % len(COLOURS)],
                label=f"{name}  AUC={pr_auc:.4f}",
            )
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision–Recall Curves — All Models", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        fig.tight_layout()
        self._save(fig, "pr_curves.png")

    def _plot_confusion(
        self, models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
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
                [[f"{cm[r][c]}\n({cm_pct[r][c]:.1f}%)" for c in range(2)]
                 for r in range(2)]
            )
            sns.heatmap(
                cm, ax=axes[idx], annot=annot, fmt="", cmap="Blues",
                cbar=True, linewidths=0.5, linecolor="white",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            )
            axes[idx].set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Predicted", fontsize=10)
            axes[idx].set_ylabel("Actual", fontsize=10)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Confusion Matrices — All Models", fontsize=15, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "confusion_matrices.png")

    def _plot_calibration(
        self, models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
        for idx, (name, pipe) in enumerate(models.items()):
            if not hasattr(pipe, "predict_proba"):
                continue
            y_prob = pipe.predict_proba(X_test)[:, 1]
            try:
                prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
                ax.plot(
                    prob_pred, prob_true, marker="o", lw=2,
                    color=COLOURS[idx % len(COLOURS)],
                    label=name,
                )
            except Exception:
                pass
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title("Calibration Curves — All Models", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        fig.tight_layout()
        self._save(fig, "calibration_curves.png")

    def _plot_feature_importance(
        self,
        models: Dict[str, Pipeline],
        feature_names: List[str],
        top_n: int = 10,
    ) -> None:
        tree_names = {"Random Forest", "Decision Tree", "XGBoost"}
        plotted = False
        for name, pipe in models.items():
            if name not in tree_names:
                continue
            clf = pipe.named_steps.get("clf")
            if clf is None or not hasattr(clf, "feature_importances_"):
                continue
            importances = clf.feature_importances_
            fi_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(top_n)
            )
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                fi_df["feature"][::-1], fi_df["importance"][::-1],
                color=COLOURS[0], edgecolor="white",
            )
            ax.set_xlabel("Importance Score", fontsize=12)
            ax.set_title(
                f"Top {top_n} Feature Importances — {name}", fontsize=13, fontweight="bold"
            )
            fig.tight_layout()
            safe = name.lower().replace(" ", "_")
            self._save(fig, f"feature_importance_top{top_n}_{safe}.png")
            plotted = True
            break  # plot for the first tree model found (champion)

        if not plotted:
            logger.info("  No tree-based model available for built-in feature importance plot.")

    def _plot_model_comparison(self, metrics_df: pd.DataFrame) -> None:
        metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc",
                       "balanced_accuracy"]
        available = [m for m in metric_cols if m in metrics_df.columns]
        models = metrics_df["model"].tolist()
        x = np.arange(len(models))
        width = 0.13

        fig, ax = plt.subplots(figsize=(16, 7))
        for i, metric in enumerate(available):
            vals = metrics_df[metric].tolist()
            bars = ax.bar(
                x + i * width, vals, width,
                label=metric.replace("_", " ").title(),
                color=COLOURS[i % len(COLOURS)],
                edgecolor="white",
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=45,
                )

        ax.set_xticks(x + width * len(available) / 2)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison — All Evaluation Metrics", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        self._save(fig, "model_comparison_bar.png")

    # ── Reports ────────────────────────────────────────────────────────────────

    def _save_classification_reports(
        self,
        models: Dict[str, Pipeline],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        rows = []
        for name, pipe in models.items():
            y_pred = pipe.predict(X_test)
            report_str = classification_report(
                y_test, y_pred, target_names=CLASS_LABELS,
                output_dict=False, zero_division=0,
            )
            logger.info("\n  Classification Report — %s:\n%s", name, report_str)
            report_dict = classification_report(
                y_test, y_pred, target_names=CLASS_LABELS,
                output_dict=True, zero_division=0,
            )
            for label, stats in report_dict.items():
                if isinstance(stats, dict):
                    rows.append({"model": name, "class": label, **stats})

        if rows:
            clf_report_path = self.metrics_dir / "classification_report.csv"
            pd.DataFrame(rows).to_csv(clf_report_path, index=False)
            logger.info("  Classification report saved → %s", clf_report_path)

    # ── Main entry point ───────────────────────────────────────────────────────

    def evaluate(
        self,
        models: Dict[str, Pipeline],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """
        Evaluate all models, generate plots and CSV reports.

        Returns
        -------
        tuple
            (metrics_df, raw_metrics_dict)
        """
        logger.info("=" * 60)
        logger.info("STEP 7: Enhanced Model Evaluation")
        logger.info("=" * 60)

        feature_names = list(X_train.columns)
        all_metrics: List[Dict] = []

        for name, pipe in models.items():
            m = self._compute_all_metrics(name, pipe, X_test, y_test)
            all_metrics.append(m)

        metrics_df = pd.DataFrame(all_metrics)

        # Save evaluation_metrics.csv
        metrics_path = self.metrics_dir / "evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("  Evaluation metrics saved → %s", metrics_path)

        # Also keep backward-compatible model_comparison.csv
        compat_cols = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
        compat_df = metrics_df[[c for c in compat_cols if c in metrics_df.columns]]
        compat_df.to_csv(self.metrics_dir / "model_comparison.csv", index=False)

        # Classification reports
        self._save_classification_reports(models, X_test, y_test)

        # Plots
        self._plot_roc(models, X_test, y_test)
        self._plot_pr(models, X_test, y_test)
        self._plot_confusion(models, X_test, y_test)
        self._plot_calibration(models, X_test, y_test)
        self._plot_feature_importance(models, feature_names, top_n=10)
        self._plot_model_comparison(metrics_df)

        raw_dict = {row["model"]: row for row in all_metrics}
        logger.info("  Enhanced evaluation complete.")
        return metrics_df, raw_dict
