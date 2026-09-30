"""
model_trainer.py
----------------
Trains four classification models on the PdM feature set:

  1. Logistic Regression  (with StandardScaler — scale-sensitive)
  2. Decision Tree        (no scaler — tree models are scale-invariant)
  3. Random Forest        (no scaler — tree models are scale-invariant)
  4. XGBoost              (no scaler — gradient boosting is scale-invariant)

Training pipeline
-----------------
A. Chronological (time-based) 70/30 split to prevent data leakage.
   The data is sorted by datetime; the first 70% goes to train, the
   last 30% to test.  This preserves temporal ordering and ensures the
   model is always evaluated on future data it has never seen.

B. Post-split sensor_anomaly_score:
   The z-score-based anomaly feature is computed AFTER the split using
   ONLY training-set statistics (mean/std), then the same transform is
   applied to the test set.  This eliminates the previous target leakage
   where global (train + test) statistics were used.

C. StandardScaler is applied ONLY inside the Logistic Regression pipeline.
   Tree-based models (DT, RF, XGBoost) do NOT include a scaler step.

D. 50% stratified sample of the dataset is used by default (memory saving).
   The sampling is done BEFORE the chronological split, keeping the
   temporal order intact within the sample.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.utils.config_loader import load_config
from src.utils.constants import NON_FEATURE_COLS, SENSOR_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Builds, trains, and manages the four PdM classification models.

    Parameters
    ----------
    config : dict, optional
        Project configuration. Loaded automatically if not provided.
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or load_config()
        self.model_cfg = self.config["models"]
        self.prep_cfg  = self.config["preprocessing"]
        self.processed_dir = Path(self.config["paths"]["processed_data_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Data preparation ──────────────────────────────────────────────────────

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return feature column names (excluding non-feature cols)."""
        return [c for c in df.columns if c not in NON_FEATURE_COLS]

    def _sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly sample a fraction of the dataset (memory saving).

        Sampling is done BEFORE the chronological split, but we sort by
        datetime afterward to ensure the temporal order is preserved within
        the sample.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature-engineered DataFrame.

        Returns
        -------
        pd.DataFrame
            Sampled DataFrame, sorted chronologically.
        """
        sample_ratio  = float(self.prep_cfg.get("sample_ratio", 0.5))
        random_state  = int(self.prep_cfg.get("random_state", 42))

        if sample_ratio < 1.0:
            df_sampled = df.sample(
                frac=sample_ratio,
                random_state=random_state,
            )
            logger.info(
                "  Sampled %.0f%% of dataset → %d rows",
                sample_ratio * 100, len(df_sampled),
            )
        else:
            df_sampled = df

        # Restore chronological ordering within the sample
        if "datetime" in df_sampled.columns:
            df_sampled = df_sampled.sort_values(
                ["machineID", "datetime"]
            ).reset_index(drop=True)

        return df_sampled

    def _chronological_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform a chronological (time-ordered) train/test split.

        Strategy
        --------
        1. Sort the sampled dataset by datetime globally (all machines).
        2. Take the first ``split_ratio`` fraction as training data and
           the remaining fraction as test data.
        3. No shuffling — temporal ordering is strictly preserved.

        This prevents future data from leaking into the training set,
        which is critical for time-series / sensor data modelling.

        Parameters
        ----------
        df : pd.DataFrame
            Sampled, chronologically-sorted feature DataFrame.

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        split_ratio = float(self.prep_cfg.get("split_ratio", 0.70))

        # Sort globally by datetime (ensures temporal ordering across machines)
        if "datetime" in df.columns:
            df = df.sort_values("datetime").reset_index(drop=True)
            # Find datetime threshold to guarantee strict temporal separation
            unique_dates = df["datetime"].drop_duplicates().sort_values().values
            cutoff_idx = int(len(unique_dates) * split_ratio)
            cutoff_time = unique_dates[cutoff_idx]

            train_df = df[df["datetime"] < cutoff_time].copy()
            test_df  = df[df["datetime"] >= cutoff_time].copy()

            train_end   = train_df["datetime"].max()
            test_start  = test_df["datetime"].min()

            logger.info(
                "  Chronological split (%.0f%% / %.0f%%) → "
                "Train: %d rows | Test: %d rows | Features: %d",
                split_ratio * 100, (1 - split_ratio) * 100,
                len(train_df), len(test_df), len(self._get_feature_cols(df)),
            )
            logger.info(
                "  Train period ends: %s | Test period starts: %s",
                train_end, test_start,
            )

            # Strict validation: max(train_datetime) < min(test_datetime)
            if train_end >= test_start:
                msg = (
                    f"TEMPORAL OVERLAP DETECTED! max(train_datetime)={train_end} >= "
                    f"min(test_datetime)={test_start}. Train and test sets must not share timestamps."
                )
                logger.error("  %s", msg)
                raise ValueError(msg)
            else:
                logger.info("  ✔ Chronological split validated: max(train_datetime) < min(test_datetime) (ZERO OVERLAP).")
        else:
            logger.warning(
                "  'datetime' column not found; falling back to row-order split."
            )
            split_idx = int(len(df) * split_ratio)
            train_df = df.iloc[:split_idx]
            test_df  = df.iloc[split_idx:]

        feature_cols = self._get_feature_cols(df)
        X_train = train_df[feature_cols]
        y_train = train_df["failure_label"]
        X_test  = test_df[feature_cols]
        y_test  = test_df["failure_label"]

        logger.info(
            "  Train failure rate: %.4f | Test failure rate: %.4f",
            float(y_train.mean()), float(y_test.mean()),
        )

        # Persist splits
        train_df.to_csv(self.processed_dir / "train_data.csv", index=False)
        test_df.to_csv(self.processed_dir / "test_data.csv", index=False)
        logger.info("  Train split saved → %s", self.processed_dir / "train_data.csv")
        logger.info("  Test split  saved → %s", self.processed_dir / "test_data.csv")

        return X_train, X_test, y_train, y_test

    def _compute_anomaly_score(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute sensor_anomaly_score using ONLY training-set statistics.

        This is the LEAKAGE-FREE version of the former global z-score.
        The training mean/std are computed from X_train only, then applied
        to both X_train and X_test — exactly as a StandardScaler would work.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (already split).
        X_test : pd.DataFrame
            Test feature matrix (already split).

        Returns
        -------
        tuple
            (X_train, X_test) with 'sensor_anomaly_score' column added.
        """
        present = [c for c in SENSOR_COLS if c in X_train.columns]
        if len(present) < 2:
            logger.warning(
                "  Fewer than 2 sensor columns found; skipping anomaly score."
            )
            return X_train, X_test

        logger.info(
            "  Computing sensor_anomaly_score using TRAIN-ONLY statistics "
            "(leakage-free) …"
        )

        # Compute statistics from training data ONLY
        train_means = X_train[present].mean()
        train_stds  = X_train[present].std().replace(0, 1e-6)

        # Apply same transform to both sets
        def _anomaly_score(df: pd.DataFrame) -> pd.Series:
            z = (df[present] - train_means) / train_stds
            return z.abs().mean(axis=1)

        X_train = X_train.copy()
        X_test  = X_test.copy()
        X_train["sensor_anomaly_score"] = _anomaly_score(X_train)
        X_test["sensor_anomaly_score"]  = _anomaly_score(X_test)

        logger.info("  sensor_anomaly_score added to X_train and X_test.")
        return X_train, X_test

    # ── Model definitions ─────────────────────────────────────────────────────

    def _build_lr_pipeline(self) -> Pipeline:
        """
        Logistic Regression pipeline.
        StandardScaler IS required — LR is sensitive to feature scale.
        """
        cfg = self.model_cfg["logistic_regression"]
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=cfg["max_iter"],
                random_state=cfg["random_state"],
                class_weight=cfg["class_weight"],
                solver="lbfgs",
                n_jobs=1,
            )),
        ])

    def _build_dt_pipeline(self) -> Pipeline:
        """
        Decision Tree pipeline.
        No scaler — decision trees are scale-invariant.
        """
        cfg = self.model_cfg["decision_tree"]
        return Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=cfg["max_depth"],
                min_samples_split=cfg.get("min_samples_split", 10),
                min_samples_leaf=cfg.get("min_samples_leaf", 4),
                random_state=cfg["random_state"],
                class_weight=cfg["class_weight"],
            )),
        ])

    def _build_rf_pipeline(self) -> Pipeline:
        """
        Random Forest pipeline.
        No scaler — tree ensembles are scale-invariant.
        max_depth is explicitly capped to prevent 61MB models.
        """
        cfg = self.model_cfg["random_forest"]
        return Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                min_samples_split=cfg.get("min_samples_split", 5),
                min_samples_leaf=cfg.get("min_samples_leaf", 2),
                random_state=cfg["random_state"],
                n_jobs=1,
                class_weight=cfg["class_weight"],
            )),
        ])

    def _build_xgb_pipeline(self) -> Optional[Pipeline]:
        """
        XGBoost pipeline.
        No scaler — gradient boosted trees are scale-invariant.
        Returns None if XGBoost is not installed.
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("  XGBoost not installed; skipping XGB model.")
            return None
        cfg = self.model_cfg["xgboost"]
        return Pipeline([
            ("clf", XGBClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                random_state=cfg["random_state"],
                scale_pos_weight=cfg["scale_pos_weight"],
                eval_metric=cfg["eval_metric"],
                verbosity=0,
                n_jobs=1,
            )),
        ])

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(self, df: pd.DataFrame):
        """
        Split data chronologically and train all four models with optional
        hyperparameter tuning.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame with 'failure_label' column.

        Returns
        -------
        tuple
            (trained_pipelines, X_train, X_test, y_train, y_test, tuner)

            trained_pipelines : dict
                Keys are model names; values are fitted sklearn Pipeline objects.
            tuner : HyperparameterTuner
                Tuner instance exposing .training_times and .best_params.
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Model Training & Tuning")
        logger.info("=" * 60)

        # Sample data (memory optimisation), then chronological split
        df_sampled = self._sample_data(df)
        X_train, X_test, y_train, y_test = self._chronological_split(df_sampled)

        # Post-split anomaly score (leakage-free)
        X_train, X_test = self._compute_anomaly_score(X_train, X_test)

        # Feature selection using training set statistics only (leakage-free)
        from src.features.feature_selection import FeatureSelector
        selector = FeatureSelector(config=self.config)
        X_train, X_test, selected_features = selector.select_features(X_train, y_train, X_test)

        # Build all four model pipelines
        # IMPORTANT: Only Logistic Regression includes StandardScaler.
        #            Tree-based models (DT, RF, XGBoost) do NOT.
        pipelines: Dict[str, Pipeline] = {
            "Logistic Regression": self._build_lr_pipeline(),
            "Decision Tree":       self._build_dt_pipeline(),
            "Random Forest":       self._build_rf_pipeline(),
        }
        xgb_pipe = self._build_xgb_pipeline()
        if xgb_pipe is not None:
            pipelines["XGBoost"] = xgb_pipe
        else:
            logger.warning("  XGBoost not available; training 3 models.")

        logger.info("  Models to train: %s", list(pipelines.keys()))

        # Initialise tuner (handles enabled/disabled internally)
        from src.models.hyperparameter_tuner import HyperparameterTuner
        tuner = HyperparameterTuner(self.config)

        trained: Dict[str, Pipeline] = {}
        for name, pipe in pipelines.items():
            logger.info("  Training/Tuning: %s …", name)
            fitted_pipe, _, _ = tuner.tune(name, pipe, X_train, y_train)
            trained[name] = fitted_pipe
            logger.info("  ✔ %s trained.", name)

        # Persist tuner results (CSV + JSON)
        tuner.save_results()

        logger.info("All models trained successfully.\n")
        return trained, X_train, X_test, y_train, y_test, tuner


def train_models(
    df: pd.DataFrame,
    config: dict = None,
):
    """
    Convenience wrapper around ModelTrainer.train_all().

    Returns
    -------
    tuple
        (trained_pipelines, X_train, X_test, y_train, y_test, tuner)
    """
    trainer = ModelTrainer(config=config)
    return trainer.train_all(df)
