"""
shap_explainer.py
-----------------
Task 3 – Phase 2: Explainable AI with SHAP

Supports Random Forest and XGBoost pipelines.

Plots saved to reports/plots/
------------------------------
  shap_summary_<model>.png
  shap_bar_<model>.png
  shap_waterfall_<model>.png
  shap_dependence_<model>.png

Also generates one local prediction explanation dict containing:
  machine_id, prediction, failure_probability, confidence_score,
  top_5_features (name, shap_value)

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.constants import FIG_DPI, NON_FEATURE_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed. SHAP explanations will be skipped.")

_SUPPORTED_MODELS = {"Random Forest", "XGBoost"}
_PLOTS_DIR_NAME = "plots"


class SHAPExplainer:
    """
    Generates SHAP explanations and plots for tree-based models.

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

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.plots_dir / filename
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("  SHAP plot saved → %s", path)

    def _get_explainer(self, clf, X_sample: pd.DataFrame):
        """Return appropriate SHAP explainer for the classifier type."""
        clf_type = type(clf).__name__
        if "XGB" in clf_type:
            return shap.TreeExplainer(clf)
        elif "Forest" in clf_type or "Tree" in clf_type:
            return shap.TreeExplainer(clf)
        else:
            return shap.KernelExplainer(clf.predict_proba, shap.sample(X_sample, 50))

    def _transform_features(self, pipe, X: pd.DataFrame) -> np.ndarray:
        """Apply all pipeline steps except the final estimator."""
        X_trans = X.copy()
        for name, step in pipe.steps[:-1]:
            X_trans = step.transform(X_trans)
        return X_trans

    # ── Plot generators ────────────────────────────────────────────────────────

    def _plot_summary(
        self,
        shap_values: np.ndarray,
        X_transformed: np.ndarray,
        feature_names: List[str],
        model_name: str,
    ) -> None:
        safe = model_name.lower().replace(" ", "_")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_transformed,
            feature_names=feature_names,
            show=False, max_display=20,
        )
        plt.title(f"SHAP Summary — {model_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(plt.gcf(), f"shap_summary_{safe}.png")
        plt.close("all")

    def _plot_bar(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        model_name: str,
    ) -> None:
        safe = model_name.lower().replace(" ", "_")
        mean_abs = np.abs(shap_values).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.ravel()[:len(feature_names)]
        fi_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(20)
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(fi_df["feature"][::-1], fi_df["mean_abs_shap"][::-1], color="#4361EE")
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
        ax.set_title(f"SHAP Bar Plot — {model_name}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save(fig, f"shap_bar_{safe}.png")

    def _plot_waterfall(
        self,
        explainer,
        X_sample_row: np.ndarray,
        feature_names: List[str],
        model_name: str,
    ) -> None:
        safe = model_name.lower().replace(" ", "_")
        try:
            shap_exp = explainer(X_sample_row.reshape(1, -1))
            if hasattr(shap_exp, "values") and shap_exp.values.ndim > 1:
                # For classifiers returning class-specific SHAP values
                vals = shap_exp.values[0]
                if vals.ndim > 1:
                    vals = vals[:, 1]
            else:
                vals = shap_exp.values[0]

            ev = float(np.array(explainer.expected_value).flatten()[0])
            shap_exp_obj = shap.Explanation(
                values=vals,
                base_values=ev,
                data=X_sample_row,
                feature_names=feature_names,
            )
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.waterfall_plot(shap_exp_obj, show=False, max_display=15)
            plt.title(f"SHAP Waterfall — {model_name}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            self._save(plt.gcf(), f"shap_waterfall_{safe}.png")
            plt.close("all")
        except Exception as exc:
            logger.warning("  SHAP waterfall plot failed for %s: %s", model_name, exc)

    def _plot_dependence(
        self,
        shap_values: np.ndarray,
        X_transformed: np.ndarray,
        feature_names: List[str],
        model_name: str,
    ) -> None:
        """Dependence plot for the most important feature."""
        safe = model_name.lower().replace(" ", "_")
        try:
            top_idx = int(np.argmax(np.abs(shap_values).mean(axis=0)))
            fig, ax = plt.subplots(figsize=(9, 6))
            shap.dependence_plot(
                top_idx, shap_values, X_transformed,
                feature_names=feature_names, show=False, ax=ax,
            )
            ax.set_title(
                f"SHAP Dependence — {feature_names[top_idx]} | {model_name}",
                fontsize=12, fontweight="bold",
            )
            fig.tight_layout()
            self._save(fig, f"shap_dependence_{safe}.png")
        except Exception as exc:
            logger.warning("  SHAP dependence plot failed for %s: %s", model_name, exc)

    # ── Local prediction explanation ───────────────────────────────────────────

    def _local_explanation(
        self,
        explainer,
        X_sample_row: np.ndarray,
        feature_names: List[str],
        pipe,
        X_original_row: pd.DataFrame,
        model_name: str,
    ) -> Dict:
        """Build a single-prediction explanation dict."""
        try:
            # Prediction
            pred = int(pipe.predict(X_original_row)[0])
            prob = float(pipe.predict_proba(X_original_row)[0][1]) if hasattr(pipe, "predict_proba") else float(pred)
            confidence = max(prob, 1.0 - prob)

            # SHAP values for this row
            sv = explainer.shap_values(X_sample_row.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[1]  # positive class
            sv = np.array(sv).flatten()

            # Top 5 features
            top_idx = np.argsort(np.abs(sv))[::-1][:5]
            top_features = [
                {"feature": feature_names[i], "shap_value": round(float(sv[i]), 6)}
                for i in top_idx
            ]

            machine_id = int(X_original_row["machineID"].iloc[0]) if "machineID" in X_original_row.columns else -1

            return {
                "machine_id": machine_id,
                "model": model_name,
                "prediction": pred,
                "prediction_label": "Failure" if pred == 1 else "No Failure",
                "failure_probability": round(prob, 4),
                "confidence_score": round(confidence, 4),
                "top_5_features": top_features,
            }
        except Exception as exc:
            logger.warning("  Local explanation failed: %s", exc)
            return {}

    # ── Main entry point ───────────────────────────────────────────────────────

    def explain(
        self,
        models: Dict,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict]:
        """
        Generate SHAP plots and local explanations for supported models.

        Parameters
        ----------
        models : dict
            {model_name: fitted_pipeline}
        X_train : pd.DataFrame
            Training data (used to build SHAP background).
        X_test : pd.DataFrame
            Test data for SHAP computation.
        y_test : pd.Series
            Test labels (used to pick a representative sample).

        Returns
        -------
        dict
            {model_name: local_explanation_dict}
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP unavailable — skipping SHAP stage.")
            return {}

        logger.info("=" * 60)
        logger.info("STEP 9: SHAP Explainability")
        logger.info("=" * 60)

        feature_names = [
            c for c in X_test.columns if c not in NON_FEATURE_COLS
        ]
        # Use at most 500 background samples for speed
        X_bg = X_train[feature_names].sample(
            min(500, len(X_train)), random_state=42
        )

        local_explanations: Dict[str, Dict] = {}

        for model_name, pipe in models.items():
            if model_name not in _SUPPORTED_MODELS:
                logger.info("  SHAP: skipping %s (not in supported set)", model_name)
                continue

            logger.info("  SHAP: computing for %s …", model_name)
            clf = pipe.named_steps.get("clf")
            if clf is None:
                logger.warning("  SHAP: no 'clf' step found in %s pipeline.", model_name)
                continue

            try:
                # Transform features through all pre-clf steps
                X_test_trans = self._transform_features(pipe, X_test[feature_names])
                X_bg_trans = self._transform_features(pipe, X_bg)
                if hasattr(X_test_trans, "toarray"):
                    X_test_trans = X_test_trans.toarray()
                if hasattr(X_bg_trans, "toarray"):
                    X_bg_trans = X_bg_trans.toarray()

                X_test_arr = np.array(X_test_trans)
                X_bg_arr = np.array(X_bg_trans)

                explainer = shap.TreeExplainer(clf, data=X_bg_arr)

                # Compute SHAP values on a sample for speed
                sample_size = min(1000, len(X_test_arr))
                idx = np.random.default_rng(42).choice(len(X_test_arr), sample_size, replace=False)
                X_shap_sample = X_test_arr[idx]

                shap_values = explainer.shap_values(X_shap_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # positive class
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    if shap_values.shape[2] >= 2:
                        shap_values = shap_values[:, :, 1]
                    else:
                        shap_values = shap_values[:, :, 0]
                shap_values = np.array(shap_values)

                # Ensure shap_values is 2D
                if shap_values.ndim == 3:
                    shap_values = shap_values[:, :, -1]

                # Plots
                self._plot_summary(shap_values, X_shap_sample, feature_names, model_name)
                self._plot_bar(shap_values, feature_names, model_name)
                self._plot_waterfall(explainer, X_test_arr[0], feature_names, model_name)
                self._plot_dependence(shap_values, X_shap_sample, feature_names, model_name)

                # Local explanation for a positive instance or fallback to first
                positive_idx = np.where(np.array(y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]) == 1)[0]
                explain_row_idx = int(positive_idx[0]) if len(positive_idx) > 0 else 0
                original_idx = idx[explain_row_idx]
                X_orig_row = X_test.iloc[[original_idx]]

                local_exp = self._local_explanation(
                    explainer, X_test_arr[explain_row_idx],
                    feature_names, pipe, X_orig_row, model_name,
                )
                local_explanations[model_name] = local_exp

                logger.info("  SHAP ✔ %s — explanation generated.", model_name)

            except Exception as exc:
                if model_name == "Random Forest":
                    logger.warning("  SHAP skipped for %s (incompatibility handled gracefully): %s", model_name, exc)
                else:
                    logger.error("  SHAP failed for %s: %s", model_name, exc)

        return local_explanations
