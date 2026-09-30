"""
data_cleaner.py
---------------
Cleans the merged PdM DataFrame:

1.  Drop full duplicate rows.
2.  Handle missing values (imputation strategy per column type).
3.  Clip sensor outliers using IQR fencing.
4.  One-hot encode the 'model' (machine model) categorical column.
5.  Ensure correct dtypes for all columns.
6.  Log a comprehensive cleaning report.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.config_loader import load_config
from src.utils.constants import SENSOR_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Applies a sequential cleaning pipeline to the merged PdM DataFrame.

    Parameters
    ----------
    config : dict, optional
        Project configuration. Loaded automatically if not provided.
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or load_config()
        self.interim_dir = Path(self.config["paths"]["interim_data_dir"])
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    # ── Step helpers ──────────────────────────────────────────────────────────

    def _report_nulls(self, df: pd.DataFrame, stage: str) -> None:
        """Log per-column null counts at the given stage."""
        total_nulls = df.isnull().sum().sum()
        logger.info("  [%s] Total nulls: %d", stage, total_nulls)
        if total_nulls > 0:
            null_summary = df.isnull().sum()
            null_summary = null_summary[null_summary > 0]
            for col, cnt in null_summary.items():
                logger.info("    %-30s  %d nulls", col, cnt)

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove fully duplicate rows."""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        logger.info("  Dropped %d duplicate rows (before=%d, after=%d).", removed, before, len(df))
        return df

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure 'datetime' column is pandas Timestamp and sorted."""
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
            df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
            logger.info("  'datetime' column parsed and DataFrame sorted.")
        return df

    def _impute_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill sensor readings per machine (captures temporal continuity),
        then back-fill any remaining NaNs at the start.
        """
        present_sensors = [c for c in SENSOR_COLS if c in df.columns]
        if not present_sensors:
            return df

        logger.info("  Forward-filling sensor columns: %s", present_sensors)
        df[present_sensors] = (
            df.groupby("machineID")[present_sensors]
            .transform(lambda x: x.ffill().bfill())
        )
        # Any remaining NaN → column median
        for col in present_sensors:
            remaining = df[col].isnull().sum()
            if remaining > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(
                    "    Filled %d remaining NaN in '%s' with median=%.4f",
                    remaining, col, median_val,
                )
        return df

    def _clip_sensor_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip sensor values to [Q1 - 3*IQR, Q3 + 3*IQR] to remove extreme
        outliers while preserving the distribution shape.
        """
        present_sensors = [c for c in SENSOR_COLS if c in df.columns]
        for col in present_sensors:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            clipped = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            if clipped:
                logger.info(
                    "  Clipped %d outliers in '%-12s' → [%.3f, %.3f]",
                    clipped, col, lower, upper,
                )
        return df

    def _impute_count_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN in error/maintenance count columns with 0."""
        count_cols = [
            c for c in df.columns
            if c.startswith("error") or c.startswith("maint_")
        ]
        if count_cols:
            df[count_cols] = df[count_cols].fillna(0)
            logger.info("  Filled NaN in %d count columns with 0.", len(count_cols))
        return df

    def _encode_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the machine 'model' categorical column."""
        if "model" not in df.columns:
            return df
        dummies = pd.get_dummies(df["model"], prefix="model", drop_first=True)
        # Convert bool dummies to int
        dummies = dummies.astype(int)
        df = pd.concat([df.drop(columns=["model"]), dummies], axis=1)
        logger.info(
            "  One-hot encoded 'model' → added columns: %s", list(dummies.columns)
        )
        return df

    def _fill_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN in the 'age' column with the median age."""
        if "age" in df.columns:
            n_null = df["age"].isnull().sum()
            if n_null > 0:
                median_age = df["age"].median()
                df["age"] = df["age"].fillna(median_age)
                logger.info(
                    "  Filled %d NaN in 'age' with median age=%.1f.", n_null, median_age
                )
        return df

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        """
        Execute the full cleaning pipeline on the merged DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Merged PdM DataFrame from DataMerger.
        save : bool
            If True, persist the cleaned dataset to the interim directory.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Data Cleaning")
        logger.info("=" * 60)
        logger.info("  Input shape: %s", df.shape)

        self._report_nulls(df, "before cleaning")

        df = self._parse_datetime(df)
        df = self._drop_duplicates(df)
        df = self._impute_sensors(df)
        df = self._clip_sensor_outliers(df)
        df = self._impute_count_columns(df)
        df = self._fill_age(df)
        df = self._encode_model(df)

        self._report_nulls(df, "after cleaning")
        logger.info("  Output shape: %s", df.shape)

        if save:
            out_path = self.interim_dir / "cleaned_dataset.csv"
            df.to_csv(out_path, index=False)
            logger.info("  Saved cleaned dataset → %s", out_path)

        logger.info("Data cleaning complete.\n")
        return df


def clean_data(df: pd.DataFrame, config: dict = None, save: bool = True) -> pd.DataFrame:
    """
    Convenience wrapper around DataCleaner.clean().

    Parameters
    ----------
    df : pd.DataFrame
        Merged raw DataFrame.
    config : dict, optional
        Project config.
    save : bool
        Persist intermediate output if True.

    Returns
    -------
    pd.DataFrame
    """
    cleaner = DataCleaner(config=config)
    return cleaner.clean(df, save=save)
