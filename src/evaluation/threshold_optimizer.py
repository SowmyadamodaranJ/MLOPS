"""
threshold_optimizer.py
----------------------
Task: Optimal Classification Threshold Selection via Precision-Recall Analysis.

Problem
-------
The default threshold of 0.5 is suboptimal for imbalanced PdM datasets where:
  - False Negatives (missed failures) are far more costly than False Positives
  - Failure rate is typically < 5%, making P(1) < 0.5 for most samples

Solution
--------
For each trained model, sweep thresholds on the validation/test set and select
the threshold that maximizes the F-beta score (beta=2 by default, weighting
recall twice as much as precision).

F_beta = (1 + beta²) * precision * recall / (beta² * precision + recall)

With beta=2:
  - A missed failure (FN) is 4× worse than a false alarm (FP)
  - This is the standard trade-off assumption for predictive maintenance

Outputs
-------
  reports/metrics/optimal_thresholds.json — {model_name: float} mapping
  reports/plots/threshold_analysis_{model}.png — PR curve + optimal point

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThresholdOptimizer:
    """
    Finds the optimal classification threshold for each trained model
    by maximizing F-beta score on the test set.

    Parameters
    ----------
    config : dict, optional
        Project configuration. Loaded automatically if not provided.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_config()
        eval_cfg = self.config.get("evaluation", {})

        self.optimize: bool = eval_cfg.get("optimize_threshold", True)
        self.beta: float = float(eval_cfg.get("threshold_beta", 2.0))
        self.default_threshold: float = float(eval_cfg.get("threshold", 0.5))

        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])
        self.plots_dir   = Path(self.config["paths"].get("plots_dir", "reports/plots"))
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._optimal_thresholds: Dict[str, float] = {}

    # ── Core optimization ─────────────────────────────────────────────────────

    def _find_optimal_threshold(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        model_name: str,
    ) -> float:
        """
        Find the threshold that maximizes F-beta score.

        Parameters
        ----------
        y_true : pd.Series
            True binary labels.
        y_proba : np.ndarray
            Predicted probabilities for the positive class.
        model_name : str
            Name for logging and plot file names.

        Returns
        -------
        float
            Optimal threshold in [0, 1].
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # thresholds has one fewer element than precisions/recalls
        # (sklearn convention: last precision=1, recall=0 has no threshold)
        f_beta_scores = []
        for p, r in zip(precisions[:-1], recalls[:-1]):
            denom = self.beta ** 2 * p + r
            if denom > 0:
                fb = (1 + self.beta ** 2) * p * r / denom
            else:
                fb = 0.0
            f_beta_scores.append(fb)

        if not f_beta_scores:
            logger.warning(
                "  [ThresholdOpt] Could not compute F-beta for %s; "
                "using default %.2f", model_name, self.default_threshold
            )
            return self.default_threshold

        best_idx  = int(np.argmax(f_beta_scores))
        best_thr  = float(thresholds[best_idx])
        best_fb   = f_beta_scores[best_idx]

        logger.info(
            "  [ThresholdOpt] %-22s → optimal threshold=%.4f  "
            "F%.0f=%.4f  precision=%.4f  recall=%.4f",
            model_name, best_thr, self.beta, best_fb,
            precisions[best_idx], recalls[best_idx],
        )

        # Plot PR curve with optimal point
        self._plot_threshold_analysis(
            precisions=precisions,
            recalls=recalls,
            thresholds=thresholds,
            f_beta_scores=f_beta_scores,
            best_idx=best_idx,
            best_thr=best_thr,
            model_name=model_name,
        )

        return best_thr

    def _plot_threshold_analysis(
        self,
        precisions: np.ndarray,
        recalls: np.ndarray,
        thresholds: np.ndarray,
        f_beta_scores: list,
        best_idx: int,
        best_thr: float,
        model_name: str,
    ) -> None:
        """Save a PR curve plot with the optimal threshold highlighted."""
        safe_name = model_name.lower().replace(" ", "_")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Precision-Recall curve
        ax = axes[0]
        ax.plot(recalls[:-1], precisions[:-1], linewidth=2, color="#4361EE",
                label="PR curve")
        ax.scatter(
            recalls[best_idx], precisions[best_idx],
            color="#F72585", s=120, zorder=5,
            label=f"Optimal (thr={best_thr:.3f})"
        )
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Right: F-beta score vs threshold
        ax2 = axes[1]
        ax2.plot(thresholds, f_beta_scores, linewidth=2, color="#7209B7",
                 label=f"F{self.beta:.0f} score")
        ax2.axvline(best_thr, color="#F72585", linestyle="--", linewidth=1.5,
                    label=f"Optimal threshold = {best_thr:.3f}")
        ax2.set_xlabel("Threshold", fontsize=12)
        ax2.set_ylabel(f"F{self.beta:.0f} Score", fontsize=12)
        ax2.set_title(f"F{self.beta:.0f} Score vs Threshold — {model_name}",
                      fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        out_path = self.plots_dir / f"threshold_analysis_{safe_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  [ThresholdOpt] Saved threshold analysis → %s", out_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize_all(
        self,
        models: Dict[str, Pipeline],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Find optimal thresholds for all trained models.

        Parameters
        ----------
        models : dict
            Trained sklearn Pipeline objects keyed by model name.
        X_test : pd.DataFrame
            Test feature matrix.
        y_test : pd.Series
            True test labels.

        Returns
        -------
        dict
            Mapping of model_name → optimal threshold.
        """
        logger.info("=" * 60)
        logger.info("STEP 8b: Threshold Optimization (F-beta, beta=%.1f)", self.beta)
        logger.info("=" * 60)

        if not self.optimize:
            logger.info(
                "  Threshold optimization disabled in config. "
                "Using default threshold=%.2f for all models.",
                self.default_threshold,
            )
            self._optimal_thresholds = {
                name: self.default_threshold for name in models
            }
            return self._optimal_thresholds

        for name, pipe in models.items():
            if not hasattr(pipe, "predict_proba"):
                logger.warning(
                    "  [ThresholdOpt] %s has no predict_proba; "
                    "using default threshold.", name
                )
                self._optimal_thresholds[name] = self.default_threshold
                continue

            y_proba = pipe.predict_proba(X_test)[:, 1]
            thr = self._find_optimal_threshold(y_proba=y_proba,
                                               y_true=y_test,
                                               model_name=name)
            self._optimal_thresholds[name] = thr

        self._save_thresholds()
        return dict(self._optimal_thresholds)

    def apply_threshold(
        self,
        pipe: Pipeline,
        X: pd.DataFrame,
        threshold: float,
    ) -> np.ndarray:
        """
        Generate predictions using a custom threshold.

        Parameters
        ----------
        pipe : Pipeline
            Fitted sklearn Pipeline.
        X : pd.DataFrame
            Feature matrix.
        threshold : float
            Classification threshold.

        Returns
        -------
        np.ndarray
            Binary predictions using the given threshold.
        """
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X)[:, 1]
            return (y_proba >= threshold).astype(int)
        return pipe.predict(X)

    def get_threshold(self, model_name: str) -> float:
        """Return the optimal threshold for a named model."""
        return self._optimal_thresholds.get(model_name, self.default_threshold)

    def _save_thresholds(self) -> None:
        """Persist optimal thresholds to JSON for later use by the API."""
        path = self.metrics_dir / "optimal_thresholds.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._optimal_thresholds, fh, indent=2)
        logger.info(
            "  [ThresholdOpt] Saved optimal thresholds → %s", path
        )
        logger.info("  Thresholds: %s", self._optimal_thresholds)

    @property
    def optimal_thresholds(self) -> Dict[str, float]:
        """Return the per-model optimal threshold mapping."""
        return dict(self._optimal_thresholds)
