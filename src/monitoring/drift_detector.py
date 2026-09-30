"""
drift_detector.py
-----------------
Data drift detection module for the MLOps pipeline.
Uses Kolmogorov-Smirnov (KS) test and Wasserstein Distance to measure distribution shifts.

Author : Smart Factory PdM Team
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any

try:
    from scipy.stats import wasserstein_distance
    WASSERSTEIN_AVAILABLE = True
except ImportError:
    WASSERSTEIN_AVAILABLE = False


class DriftDetector:
    """
    Compares baseline (reference) data to production (target) data to detect feature drift.
    """

    def __init__(self, reference_df: pd.DataFrame) -> None:
        self.reference_df = reference_df
        self.features = ["volt", "rotate", "pressure", "vibration"]

    def detect_drift(self, target_df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """
        Run Kolmogorov-Smirnov test and Wasserstein distance for each telemetry feature.

        Parameters
        ----------
        target_df : pd.DataFrame
            The new/incoming production dataset.
        alpha : float, default 0.05
            Significance level for the KS test.

        Returns
        -------
        dict
            Drift report containing metrics and status for each feature.
        """
        report = {}
        for feature in self.features:
            if feature not in self.reference_df.columns or feature not in target_df.columns:
                continue

            ref_vals = self.reference_df[feature].dropna().values
            tar_vals = target_df[feature].dropna().values

            # KS Test
            ks_stat, p_val = ks_2samp(ref_vals, tar_vals)

            # Wasserstein Distance
            w_dist = 0.0
            if WASSERSTEIN_AVAILABLE:
                w_dist = float(wasserstein_distance(ref_vals, tar_vals))

            drift_detected = bool(p_val < alpha)

            report[feature] = {
                "ks_stat": float(ks_stat),
                "p_val": float(p_val),
                "wasserstein_distance": w_dist,
                "drift_detected": drift_detected,
                "ref_mean": float(ref_vals.mean()),
                "ref_std": float(ref_vals.std()),
                "tar_mean": float(tar_vals.mean()),
                "tar_std": float(tar_vals.std()),
            }

        return report
