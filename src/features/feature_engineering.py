"""
feature_engineering.py
-----------------------
Creates rich, time-aware features from the cleaned PdM DataFrame.

Feature Groups
--------------
A. Rolling statistics (mean, std, min, max) over 3 h and 24 h windows
   for sensor columns (volt, rotate, pressure, vibration).

B. Lag features (1 h, 3 h, 24 h) for all sensor columns.

C. Error count features (total errors per machine within trailing 24 h).

D. Calendar / time features with CYCLIC ENCODING:
   - hour_of_day_sin / hour_of_day_cos
   - day_of_week_sin / day_of_week_cos
   - month_sin / month_cos
   - is_weekend (binary)

E. Derived interaction features:
   - volt_rotate_ratio
   - pressure_vibration_ratio

F. Maintenance-aware features (NEW — critical for PdM):
   - time_since_last_maint_compN  (hours since last maintenance per component)
   - cumulative_maint_compN       (rolling 30-day maintenance count per component)

G. sensor_anomaly_score (LEAKAGE-FREE):
   This feature is intentionally NOT computed here on the full dataset.
   It is computed AFTER the train/test split using training-set statistics
   only (called from model_trainer._compute_anomaly_score()).
   The placeholder column is left as np.nan and filled post-split.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.constants import SENSOR_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maintenance component column prefixes produced by DataMerger
_MAINT_COMP_COLS = ["maint_comp1", "maint_comp2", "maint_comp3", "maint_comp4"]


class FeatureEngineer:
    """
    Builds a comprehensive feature set from the cleaned PdM DataFrame.

    Parameters
    ----------
    config : dict, optional
        Project configuration. Loaded automatically if not provided.
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or load_config()
        fe_cfg = self.config["feature_engineering"]
        self.rolling_windows: List[int] = fe_cfg["rolling_window_hours"]
        self.lag_hours: List[int] = fe_cfg["lag_hours"]
        self.error_window: int = int(fe_cfg["error_window_hours"])
        self.processed_dir = Path(self.config["paths"]["processed_data_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by machineID and datetime; required for rolling/lag ops."""
        return df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling mean, std, min, max for each sensor over each window.
        Operations are performed per machineID group.
        """
        logger.info("  Adding rolling features (windows=%s) …", self.rolling_windows)
        new_cols: List[str] = []
        for window in self.rolling_windows:
            grp = df.groupby("machineID")
            for sensor in SENSOR_COLS:
                if sensor not in df.columns:
                    continue

                rolling_obj = grp[sensor].rolling(window, min_periods=1)

                mean_col = f"{sensor}_rolling{window}h_mean"
                std_col  = f"{sensor}_rolling{window}h_std"
                min_col  = f"{sensor}_rolling{window}h_min"
                max_col  = f"{sensor}_rolling{window}h_max"

                df[mean_col] = rolling_obj.mean().reset_index(level=0, drop=True)
                df[std_col]  = rolling_obj.std().reset_index(level=0, drop=True)
                df[min_col]  = rolling_obj.min().reset_index(level=0, drop=True)
                df[max_col]  = rolling_obj.max().reset_index(level=0, drop=True)

                new_cols.extend([mean_col, std_col, min_col, max_col])

        # Fill NaN introduced by std on windows of 1
        df[[c for c in new_cols if "std" in c]] = (
            df[[c for c in new_cols if "std" in c]].fillna(0)
        )
        logger.info("  Added %d rolling feature columns.", len(new_cols))
        return df

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features for each sensor at each lag hour.
        Gaps are forward-filled within each machineID group.
        """
        logger.info("  Adding lag features (lags=%s h) …", self.lag_hours)
        new_cols: List[str] = []
        for lag in self.lag_hours:
            grp = df.groupby("machineID")
            for sensor in SENSOR_COLS:
                if sensor not in df.columns:
                    continue
                col_name = f"{sensor}_lag{lag}h"
                df[col_name] = grp[sensor].shift(lag)
                new_cols.append(col_name)

        # Backfill lag NaNs within group
        lag_col_list = [c for c in df.columns if "_lag" in c]
        if lag_col_list:
            df[lag_col_list] = (
                df.groupby("machineID")[lag_col_list]
                .bfill().ffill()
            )
        logger.info("  Added %d lag feature columns.", len(new_cols))
        return df

    def _error_count_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the total error count per machine over a trailing
        ``error_window`` hours, using the one-hot error columns.
        """
        error_cols = [c for c in df.columns if c.startswith("error")]
        if not error_cols:
            logger.warning("  No error columns found; skipping error count features.")
            return df

        logger.info(
            "  Adding total error count features (window=%d h) …",
            self.error_window,
        )
        df["total_errors"] = df[error_cols].sum(axis=1)
        df["total_errors_rolling24h"] = (
            df.groupby("machineID")["total_errors"]
            .rolling(self.error_window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        logger.info("  Added 'total_errors' and 'total_errors_rolling24h'.")
        return df

    def _calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract calendar / temporal features from the datetime column.

        Raw integers (e.g. hour=23 is close to hour=0) are NOT meaningful
        to linear models. Cyclic encoding via sin/cos pairs preserves the
        circular nature of time.  Raw integers are also kept for tree models.
        """
        logger.info("  Adding calendar features (cyclic encoding) …")
        dt = df["datetime"]

        hour      = dt.dt.hour
        dow       = dt.dt.dayofweek
        month     = dt.dt.month

        # Raw integers (useful for tree-based models)
        df["hour_of_day"]  = hour
        df["day_of_week"]  = dow
        df["month"]        = month
        df["is_weekend"]   = (dow >= 5).astype(int)

        # Cyclic encoding (useful for all model types)
        df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
        df["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
        df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

        logger.info(
            "  Added: hour_of_day, day_of_week, month, is_weekend, "
            "hour_sin/cos, dow_sin/cos, month_sin/cos."
        )
        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-driven interaction features for the factory sensors.

        NOTE: sensor_anomaly_score is intentionally excluded here.
        It requires training-set statistics and is computed post-split
        in model_trainer to avoid target leakage.
        """
        logger.info("  Adding interaction / derived features …")

        if "volt" in df.columns and "rotate" in df.columns:
            df["volt_rotate_ratio"] = df["volt"] / (df["rotate"] + 1e-6)

        if "pressure" in df.columns and "vibration" in df.columns:
            df["pressure_vibration_ratio"] = df["pressure"] / (df["vibration"] + 1e-6)

        logger.info("  Added: volt_rotate_ratio, pressure_vibration_ratio.")
        logger.info(
            "  NOTE: sensor_anomaly_score is deferred to post-split "
            "computation to prevent target leakage."
        )
        return df

    def _maintenance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute maintenance-aware features critical for PdM modelling:

        1. time_since_last_maint_compN (hours since last maintenance event
           for each component, per machine).  A machine that was recently
           serviced is much less likely to fail.

        2. cumulative_maint_compN (rolling 30-day count of maintenance events
           per component, per machine).  Captures maintenance frequency.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain machineID, datetime, and maint_compN columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with new maintenance feature columns appended.
        """
        maint_cols = [c for c in _MAINT_COMP_COLS if c in df.columns]
        if not maint_cols:
            logger.warning(
                "  No maintenance component columns found; "
                "skipping maintenance features."
            )
            return df

        logger.info(
            "  Adding maintenance features for components: %s …", maint_cols
        )

        for comp in maint_cols:
            hours_col = f"time_since_last_{comp}"
            cum_col   = f"cumulative_{comp}_30d"

            # ── Time since last maintenance ──────────────────────────────────
            # For each row: how many hours since the last maintenance event
            # for this component on this machine?
            # Strategy: within each machineID group, forward-fill the datetime
            # of the last maint event, then compute the difference.

            def _time_since(group: pd.DataFrame, comp_col: str) -> pd.Series:
                """Return hours-since-last-maint Series for one machine group."""
                last_maint_time = pd.NaT
                result = []
                for _, row in group.iterrows():
                    if row[comp_col] > 0:
                        # Maintenance occurred at this timestamp
                        last_maint_time = row["datetime"]
                    if pd.isna(last_maint_time):
                        result.append(np.nan)
                    else:
                        delta_hours = (
                            row["datetime"] - last_maint_time
                        ).total_seconds() / 3600.0
                        result.append(delta_hours)
                return pd.Series(result, index=group.index)

            logger.info("    Computing %s …", hours_col)
            df[hours_col] = (
                df.groupby("machineID", group_keys=False)
                .apply(lambda g: _time_since(g, comp))
            )

            # Fill NaN (no prior maintenance) with a large sentinel value
            # representing "never maintained" — 8760h = 1 year
            df[hours_col] = df[hours_col].fillna(8760.0)

            # ── Rolling 30-day cumulative maintenance count ──────────────────
            # 30 days × 24 hours = 720 hourly rows
            _WINDOW_30D = 720
            df[cum_col] = (
                df.groupby("machineID")[comp]
                .rolling(_WINDOW_30D, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )

            logger.info("    Added: %s, %s", hours_col, cum_col)

        logger.info("  Maintenance feature engineering complete.")
        return df

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def engineer(self, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame from DataCleaner.
        save : bool
            Persist the feature-engineered DataFrame if True.

        Returns
        -------
        pd.DataFrame
            Feature-rich DataFrame ready for model training.
            NOTE: sensor_anomaly_score is NOT included here.
            It is added post-split in the training stage.
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Feature Engineering")
        logger.info("=" * 60)
        logger.info("  Input shape: %s", df.shape)

        df = self._sort(df)
        df = self._rolling_features(df)
        df = self._lag_features(df)
        df = self._error_count_features(df)
        df = self._calendar_features(df)
        df = self._maintenance_features(df)   # NEW: time_since_last_maint, cumulative
        df = self._interaction_features(df)   # NOTE: anomaly score excluded (leakage fix)

        # Drop any remaining NaNs introduced by feature engineering
        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(
            "  Dropped %d rows with NaN after feature engineering.",
            n_before - len(df),
        )

        logger.info("  Output shape: %s", df.shape)
        logger.info(
            "  Total features (excl. target): %d",
            len(df.columns) - 1,  # subtract failure_label
        )

        if save:
            out_path = self.processed_dir / "features_engineered.csv"
            df.to_csv(out_path, index=False)
            logger.info("  Saved feature dataset → %s", out_path)

        logger.info("Feature engineering complete.\n")
        return df


def engineer_features(
    df: pd.DataFrame, config: dict = None, save: bool = True
) -> pd.DataFrame:
    """
    Convenience wrapper around FeatureEngineer.engineer().

    Returns
    -------
    pd.DataFrame
    """
    fe = FeatureEngineer(config=config)
    return fe.engineer(df, save=save)
