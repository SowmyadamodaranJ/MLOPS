"""
feature_analysis.py
-------------------
Task 4 – Phase 2: Feature Importance Module

Supports three importance methods:
  1. Built-in (tree feature_importances_)
  2. Permutation importance (model-agnostic)
  3. SHAP importance (mean |SHAP value|)

Outputs saved to reports/
--------------------------
  feature_importance.csv

Plots saved to reports/plots/
------------------------------
  feature_importance_top10_<method>.png
  feature_importance_top20_<method>.png

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.utils.config_loader import load_config
from src.utils.constants import CHART_COLOURS as COLOURS, FIG_DPI, NON_FEATURE_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

_PLOTS_DIR_NAME = "plots"
_TREE_NAMES = {"Random Forest", "Decision Tree", "XGBoost"}


class FeatureAnalyzer:
    """
    Computes and visualizes feature importance using multiple methods.

    Parameters
    ----------
    config : dict, optional
        Project config. Auto-loaded if not provided.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_config()
        figures_dir = Path(self.config["paths"]["figures_dir"])
        self.plots_dir = figures_dir.parent / _PLOTS_DIR_NAME
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Save helper ────────────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.plots_dir / filename
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Feature importance plot saved → %s", path)

    # ── Method 1: Built-in ─────────────────────────────────────────────────────

    def builtin_importance(
        self,
        pipe: Pipeline,
        feature_names: List[str],
        model_name: str,
    ) -> Optional[pd.DataFrame]:
        """Extract built-in feature_importances_ from tree-based models."""
        clf = pipe.named_steps.get("clf")
        if clf is None or not hasattr(clf, "feature_importances_"):
            logger.info("  [FeatureAnalyzer] Built-in importance N/A for %s.", model_name)
            return None

        fi_df = (
            pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        fi_df.insert(0, "method", "builtin")
        fi_df.insert(0, "model", model_name)
        fi_df["rank"] = fi_df.index + 1
        logger.info("  [FeatureAnalyzer] Built-in importance computed for %s.", model_name)
        return fi_df

    # ── Method 2: Permutation ──────────────────────────────────────────────────

    def permutation_importance_df(
        self,
        pipe: Pipeline,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: List[str],
        model_name: str,
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute permutation importance (model-agnostic)."""
        logger.info("  [FeatureAnalyzer] Permutation importance for %s …", model_name)
        # Use a subsample for speed on large datasets
        if len(X_val) > 5000:
            idx = np.random.default_rng(random_state).choice(len(X_val), 5000, replace=False)
            X_sub = X_val.iloc[idx]
            y_sub = y_val.iloc[idx]
        else:
            X_sub, y_sub = X_val, y_val

        result = permutation_importance(
            pipe, X_sub, y_sub,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        fi_df = (
            pd.DataFrame({
                "feature": feature_names,
                "importance": result.importances_mean,
                "std": result.importances_std,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        fi_df.insert(0, "method", "permutation")
        fi_df.insert(0, "model", model_name)
        fi_df["rank"] = fi_df.index + 1
        logger.info("  [FeatureAnalyzer] Permutation importance done for %s.", model_name)
        return fi_df

    # ── Method 3: SHAP ─────────────────────────────────────────────────────────

    def shap_importance(
        self,
        pipe: Pipeline,
        X_test: pd.DataFrame,
        feature_names: List[str],
        model_name: str,
    ) -> Optional[pd.DataFrame]:
        """Compute SHAP-based mean absolute importance."""
        if not SHAP_AVAILABLE:
            logger.info("  [FeatureAnalyzer] SHAP unavailable — skipping SHAP importance.")
            return None

        clf = pipe.named_steps.get("clf")
        if clf is None:
            return None

        try:
            # Transform features
            X_trans = X_test[feature_names].copy()
            for _, step in pipe.steps[:-1]:
                X_trans = step.transform(X_trans)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()

            sample_size = min(500, len(X_trans))
            X_sample = np.array(X_trans)[:sample_size]

            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_sample)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.array(sv)

            fi_df = (
                pd.DataFrame({
                    "feature": feature_names,
                    "importance": np.abs(sv).mean(axis=0),
                })
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            fi_df.insert(0, "method", "shap")
            fi_df.insert(0, "model", model_name)
            fi_df["rank"] = fi_df.index + 1
            logger.info("  [FeatureAnalyzer] SHAP importance computed for %s.", model_name)
            return fi_df
        except Exception as exc:
            logger.warning("  [FeatureAnalyzer] SHAP importance failed for %s: %s", model_name, exc)
            return None

    # ── Plot helpers ───────────────────────────────────────────────────────────

    def _plot_top_n(
        self, fi_df: pd.DataFrame, top_n: int, method: str, model_name: str
    ) -> None:
        top = fi_df.head(top_n)
        safe_model = model_name.lower().replace(" ", "_")
        fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color=COLOURS[0], edgecolor="white")
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(
            f"Top {top_n} Features — {method.title()} Importance ({model_name})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        self._save(fig, f"feature_importance_top{top_n}_{method}_{safe_model}.png")

    # ── Main entry point ───────────────────────────────────────────────────────

    def analyze(
        self,
        models: Dict[str, Pipeline],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        best_name: str,
    ) -> pd.DataFrame:
        """
        Compute built-in, permutation, and SHAP importance for the best model.
        Also compute built-in importance for all tree models.

        Returns
        -------
        pd.DataFrame
            Combined feature importance table (all methods).
        """
        logger.info("=" * 60)
        logger.info("STEP 10: Feature Importance Analysis")
        logger.info("=" * 60)

        feature_names = [c for c in X_test.columns if c not in NON_FEATURE_COLS]
        all_results: List[pd.DataFrame] = []

        # Built-in for all tree models
        for name, pipe in models.items():
            if name in _TREE_NAMES:
                df_bi = self.builtin_importance(pipe, feature_names, name)
                if df_bi is not None:
                    all_results.append(df_bi)
                    self._plot_top_n(df_bi, 10, "builtin", name)
                    self._plot_top_n(df_bi, 20, "builtin", name)

        # Permutation importance for best model
        if best_name in models:
            df_perm = self.permutation_importance_df(
                models[best_name], X_test[feature_names], y_test,
                feature_names, best_name,
            )
            all_results.append(df_perm)
            self._plot_top_n(df_perm, 10, "permutation", best_name)
            self._plot_top_n(df_perm, 20, "permutation", best_name)

        # SHAP importance for supported models
        for name in ["XGBoost", "Random Forest"]:
            if name in models:
                df_shap = self.shap_importance(models[name], X_test, feature_names, name)
                if df_shap is not None:
                    all_results.append(df_shap)
                    self._plot_top_n(df_shap, 10, "shap", name)
                    self._plot_top_n(df_shap, 20, "shap", name)
                break  # one SHAP run is sufficient for the report

        combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

        if not combined.empty:
            fi_path = self.metrics_dir / "feature_importance.csv"
            combined.to_csv(fi_path, index=False)
            logger.info("  Feature importance saved → %s", fi_path)

        return combined
