"""
data_merger.py
--------------
Merges all five PdM DataFrames into a single modelling-ready DataFrame.

Merge Strategy
--------------
1. Convert all datetime columns to pandas Timestamp (UTC-naive).
2. Aggregate error counts per (machineID, datetime_hour) window.
3. Aggregate maintenance component flags per (machineID, datetime_hour).
4. Create binary failure labels: 1 if a failure occurs within the next
   ``failure_label_window_hours`` hours (default 24 h).
5. Left-join telemetry ← errors ← maintenance ← machines.
6. Persist merged DataFrame to ``data/interim/``.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataMerger:
    """
    Merges and label-encodes the five PdM source DataFrames.

    Parameters
    ----------
    config : dict, optional
        Project configuration. Loaded automatically if not provided.
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or load_config()
        self.fe_cfg = self.config["feature_engineering"]
        self.interim_dir = Path(self.config["paths"]["interim_data_dir"])
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    # ── Datetime helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_datetime(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
        """Parse a string/object datetime column to pandas Timestamp."""
        df = df.copy()
        df[col] = pd.to_datetime(df[col], format="mixed")
        return df

    @staticmethod
    def _floor_to_hour(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
        """Floor datetime column to hour granularity."""
        df = df.copy()
        df[col] = df[col].dt.floor("H")
        return df

    # ── Aggregation helpers ───────────────────────────────────────────────────

    def _aggregate_errors(self, errors: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode errorID then aggregate to (machineID, datetime) level.

        Returns
        -------
        pd.DataFrame
            Columns: datetime, machineID, error1_count … error5_count
        """
        logger.info("  Aggregating error events …")
        df = self._parse_datetime(errors)
        df = self._floor_to_hour(df)

        # One-hot encode errorID
        error_dummies = pd.get_dummies(df["errorID"], prefix="error")
        df = pd.concat([df[["datetime", "machineID"]], error_dummies], axis=1)

        agg_df = df.groupby(["machineID", "datetime"], as_index=False).sum()
        logger.info("  Error aggregation shape: %s", agg_df.shape)
        return agg_df

    def _aggregate_maintenance(self, maint: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode component then aggregate to (machineID, datetime) level.

        Returns
        -------
        pd.DataFrame
            Columns: datetime, machineID, maint_comp1 … maint_comp4
        """
        logger.info("  Aggregating maintenance records …")
        df = self._parse_datetime(maint)
        df = self._floor_to_hour(df)

        comp_dummies = pd.get_dummies(df["comp"], prefix="maint")
        df = pd.concat([df[["datetime", "machineID"]], comp_dummies], axis=1)

        agg_df = df.groupby(["machineID", "datetime"], as_index=False).sum()
        logger.info("  Maintenance aggregation shape: %s", agg_df.shape)
        return agg_df

    def _build_failure_labels(
        self,
        telemetry: pd.DataFrame,
        failures: pd.DataFrame,
        window_hours: int,
    ) -> pd.DataFrame:
        """
        For every telemetry row, set label=1 if a failure occurs within the
        next ``window_hours`` hours for the same machineID.

        Returns
        -------
        pd.DataFrame
            telemetry DataFrame with added 'failure_label' column (0/1).
        """
        logger.info(
            "  Building failure labels (window=%d h) …", window_hours
        )
        failures_parsed = self._parse_datetime(failures)
        failures_parsed = self._floor_to_hour(failures_parsed)

        telem = telemetry.copy()
        telem["failure_label"] = 0

        # Group failure timestamps by machine for fast lookup
        fail_times: dict = (
            failures_parsed.groupby("machineID")["datetime"]
            .apply(list)
            .to_dict()
        )

        window = pd.Timedelta(hours=window_hours)

        for machine_id, times in fail_times.items():
            mask = telem["machineID"] == machine_id
            machine_rows = telem.loc[mask, "datetime"]
            times_series = pd.Series(times)

            def _has_failure_in_window(ts: pd.Timestamp) -> int:
                future = ts + window
                return int(any((t > ts) & (t <= future) for t in times))

            telem.loc[mask, "failure_label"] = machine_rows.apply(
                _has_failure_in_window
            )

        pos = telem["failure_label"].sum()
        neg = len(telem) - pos
        logger.info(
            "  Label distribution → failure=1: %d  |  no_failure=0: %d  |  ratio: %.4f",
            pos, neg, pos / max(neg, 1),
        )
        return telem

    # ── Main merge pipeline ───────────────────────────────────────────────────

    def merge(
        self,
        telemetry: pd.DataFrame,
        errors: pd.DataFrame,
        failures: pd.DataFrame,
        maint: pd.DataFrame,
        machines: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Execute the full merge pipeline.

        Parameters
        ----------
        telemetry, errors, failures, maint, machines : pd.DataFrame
            Raw source DataFrames.
        save : bool
            If True, persist the merged dataset to the interim directory.

        Returns
        -------
        pd.DataFrame
            Merged dataset ready for feature engineering.
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Merging all PdM datasets")
        logger.info("=" * 60)

        window_h = int(self.fe_cfg["failure_label_window_hours"])

        # --- Telemetry base ---------------------------------------------------
        telem = self._parse_datetime(telemetry)
        telem = self._floor_to_hour(telem)

        # --- Build failure labels on telemetry --------------------------------
        telem = self._build_failure_labels(telem, failures, window_h)

        # --- Aggregate errors & maintenance -----------------------------------
        err_agg = self._aggregate_errors(errors)
        maint_agg = self._aggregate_maintenance(maint)

        # --- Left-join errors → telemetry ------------------------------------
        logger.info("  Merging errors into telemetry …")
        merged = telem.merge(
            err_agg,
            on=["machineID", "datetime"],
            how="left",
        )

        # --- Left-join maintenance -------------------------------------------
        logger.info("  Merging maintenance into dataset …")
        merged = merged.merge(
            maint_agg,
            on=["machineID", "datetime"],
            how="left",
        )

        # --- Left-join machine metadata --------------------------------------
        logger.info("  Merging machine metadata …")
        merged = merged.merge(
            machines,
            on="machineID",
            how="left",
        )

        # --- Fill NaN in error/maintenance counts with 0 --------------------
        err_cols = [c for c in merged.columns if c.startswith("error")]
        maint_cols = [c for c in merged.columns if c.startswith("maint_")]
        merged[err_cols] = merged[err_cols].fillna(0)
        merged[maint_cols] = merged[maint_cols].fillna(0)

        logger.info("  Final merged shape: %s", merged.shape)
        logger.info(
            "  Columns: %s", list(merged.columns)
        )

        # --- Persist ---------------------------------------------------------
        if save:
            out_path = self.interim_dir / "merged_dataset.csv"
            merged.to_csv(out_path, index=False)
            logger.info("  Saved merged dataset → %s", out_path)

        logger.info("Dataset merge complete.\n")
        return merged


def merge_datasets(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    failures: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
    config: dict = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper around DataMerger.merge().

    Returns
    -------
    pd.DataFrame
    """
    merger = DataMerger(config=config)
    return merger.merge(telemetry, errors, failures, maint, machines, save=save)
