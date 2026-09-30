"""
feature_selection.py
--------------------
Selects informative features using training data ONLY to prevent data leakage.

Methods
-------
- Importance Threshold: Uses a preliminary Random Forest classifier on X_train
  to rank features and drop low-importance / uninformative features.
- Variance Threshold: Drops zero-variance (constant) features.

Outputs
-------
- Returns filtered X_train, X_test, and the list of selected feature names.
- Saves selected feature names and feature importance scores to reports/metrics/.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """
    Selects top features based on training data statistics.

    Parameters
    ----------
    config : dict, optional
        Project configuration. Auto-loaded if not provided.
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or load_config()
        fs_cfg = self.config.get("feature_selection", {})
        self.enabled: bool = fs_cfg.get("enabled", True)
        self.threshold: float = float(fs_cfg.get("importance_threshold", 0.005))
        self.metrics_dir = Path(self.config["paths"]["metrics_dir"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Perform feature selection on X_train and apply to X_test.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target vector.
        X_test : pd.DataFrame
            Test feature matrix.

        Returns
        -------
        tuple
            (X_train_selected, X_test_selected, selected_feature_names)
        """
        if not self.enabled:
            logger.info("  [FeatureSelector] Feature selection disabled in config.")
            return X_train, X_test, list(X_train.columns)

        logger.info("=" * 60)
        logger.info("STEP 5b: Feature Selection (Train Data Only)")
        logger.info("=" * 60)
        logger.info("  Initial features: %d", len(X_train.columns))

        # 1. Variance Threshold (remove constant features)
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(X_train)
        non_constant = X_train.columns[vt.get_support()].tolist()

        if len(non_constant) < len(X_train.columns):
            dropped_const = set(X_train.columns) - set(non_constant)
            logger.info("  Dropped %d zero-variance features: %s", len(dropped_const), dropped_const)
            X_train = X_train[non_constant]
            X_test  = X_test[non_constant]

        # 2. Random Forest Importance Thresholding on X_train ONLY
        logger.info("  Ranking feature importances using RandomForest on X_train...")
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)

        importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

        # Select features exceeding the threshold
        selected = importances[importances >= self.threshold].index.tolist()

        # Fallback: keep at least top 15 features if threshold is too strict
        if len(selected) < 15:
            logger.warning("  Selection threshold %.4f kept only %d features. Falling back to top 15.", self.threshold, len(selected))
            selected = importances.head(15).index.tolist()

        dropped = set(X_train.columns) - set(selected)
        logger.info(
            "  Feature selection complete: %d features selected, %d features dropped.",
            len(selected), len(dropped)
        )
        logger.info("  Dropped features: %s", sorted(list(dropped)))

        # Save feature importances report
        fi_df = pd.DataFrame({"feature": importances.index, "importance": importances.values})
        fi_df.to_csv(self.metrics_dir / "feature_selection_importances.csv", index=False)

        X_train_sel = X_train[selected].copy()
        X_test_sel  = X_test[selected].copy()

        return X_train_sel, X_test_sel, selected


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    config: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Convenience wrapper for FeatureSelector.
    """
    selector = FeatureSelector(config=config)
    return selector.select_features(X_train, y_train, X_test)
