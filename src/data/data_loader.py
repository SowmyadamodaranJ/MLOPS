"""
data_loader.py
--------------
Module responsible for loading all five PdM CSV files and performing
initial validation/schema checks.

Responsibilities
----------------
1. Load each CSV file from the raw data directory.
2. Validate expected columns and dtypes.
3. Return validated DataFrames to the caller.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.constants import EXPECTED_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Loads and validates the five Azure PdM CSV files.

    Parameters
    ----------
    config : dict, optional
        Project configuration dictionary. If None, loads default config.
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or load_config()
        self.raw_dir = Path(self.config["paths"]["raw_data_dir"])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_path(self, filename: str) -> Path:
        """Return full path to a raw CSV file."""
        return self.raw_dir / filename

    def _load_csv(self, key: str) -> pd.DataFrame:
        """
        Load a single CSV file by its config key.

        Parameters
        ----------
        key : str
            One of: 'telemetry', 'errors', 'failures', 'maint', 'machines'.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FileNotFoundError
            If the CSV file is not present in the raw data directory.
        """
        filename = self.config["raw_files"][key]
        filepath = self._resolve_path(filename)

        if not filepath.exists():
            raise FileNotFoundError(
                f"[DataLoader] Required file not found: {filepath}\n"
                f"Please place '{filename}' inside '{self.raw_dir}' and retry."
            )

        logger.info("Loading %-12s  ← %s", key.upper(), filepath)
        df = pd.read_csv(filepath)
        logger.info("  Shape: %s", df.shape)
        return df

    def _validate_columns(self, df: pd.DataFrame, key: str) -> None:
        """
        Ensure all expected columns are present in a DataFrame.

        Raises
        ------
        ValueError
            If any expected column is missing.
        """
        expected = EXPECTED_COLUMNS[key]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"[DataLoader] '{key}' is missing columns: {missing}\n"
                f"Found: {list(df.columns)}"
            )
        logger.info("  ✔ Column validation passed for '%s'.", key)

    def _basic_info(self, df: pd.DataFrame, key: str) -> None:
        """Log basic statistics about the loaded DataFrame."""
        logger.info(
            "  %s → rows=%d, cols=%d, nulls=%d, dtypes=%s",
            key.upper(),
            len(df),
            len(df.columns),
            df.isnull().sum().sum(),
            dict(df.dtypes.value_counts()),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                pd.DataFrame, pd.DataFrame]:
        """
        Load all five PdM CSV files and validate their schemas.

        Returns
        -------
        tuple
            (telemetry_df, errors_df, failures_df, maint_df, machines_df)
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Loading all PdM datasets")
        logger.info("=" * 60)

        datasets: Dict[str, pd.DataFrame] = {}
        for key in EXPECTED_COLUMNS:
            df = self._load_csv(key)
            self._validate_columns(df, key)
            self._basic_info(df, key)
            datasets[key] = df

        logger.info("All five datasets loaded and validated successfully.\n")
        
        logger.info("Expanding and shifting datasets to cover the last two years (2024-07-19 to 2026-07-18) ...")
        def _shift_to_last_two_years(df: pd.DataFrame) -> pd.DataFrame:
            if "datetime" not in df.columns:
                return df
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
            
            # First year: shift from 2015 to 2024-07-19 (3487 days)
            df_y1 = df.copy()
            df_y1["datetime"] = df_y1["datetime"] + pd.Timedelta(days=3487)
            
            # Second year: shift from 2015 to 2025-07-19 (3852 days)
            df_y2 = df.copy()
            df_y2["datetime"] = df_y2["datetime"] + pd.Timedelta(days=3852)
            
            combined = pd.concat([df_y1, df_y2], ignore_index=True)
            if "machineID" in combined.columns:
                combined = combined.sort_values(["machineID", "datetime"]).reset_index(drop=True)
            else:
                combined = combined.sort_values("datetime").reset_index(drop=True)
            return combined

        telemetry_2y = _shift_to_last_two_years(datasets["telemetry"])
        errors_2y = _shift_to_last_two_years(datasets["errors"])
        failures_2y = _shift_to_last_two_years(datasets["failures"])
        maint_2y = _shift_to_last_two_years(datasets["maint"])
        machines_df = datasets["machines"]

        logger.info("Datasets successfully expanded to 2 years. Telemetry shape: %s", telemetry_2y.shape)
        return (
            telemetry_2y,
            errors_2y,
            failures_2y,
            maint_2y,
            machines_df,
        )


def load_datasets(config: dict = None) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Convenience wrapper around DataLoader.load_all().

    Parameters
    ----------
    config : dict, optional
        Project config. Loaded automatically if not provided.

    Returns
    -------
    tuple
        (telemetry_df, errors_df, failures_df, maint_df, machines_df)
    """
    loader = DataLoader(config=config)
    return loader.load_all()
