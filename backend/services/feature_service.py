"""
feature_service.py
-------------------
Service to cache the latest engineered features for machines and provide
real-time feature engineering overrides for prediction inputs.
"""

from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional
from backend.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "features_engineered.csv"
MINI_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "latest_machine_features.csv"
MACHINES_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "PdM_machines.csv"

class FeatureService:
    """
    Caches latest machine feature states and provides feature manipulation for inference.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FeatureService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.features_cache: Dict[int, Dict[str, Any]] = {}
        self.machines_df: Optional[pd.DataFrame] = None
        self._load_features_cache()
        self._load_machines_data()
        self._initialized = True

    def _load_features_cache(self):
        """
        Load latest features for each machine.
        Uses a pre-saved mini-cache file if available; otherwise parses features_engineered.csv in chunks.
        """
        if MINI_CACHE_PATH.exists():
            try:
                logger.info(f"Loading features mini-cache from {MINI_CACHE_PATH}...")
                df = pd.read_csv(MINI_CACHE_PATH)
                for _, row in df.iterrows():
                    m_id = int(row["machineID"])
                    self.features_cache[m_id] = row.to_dict()
                logger.info(f"Loaded feature state for {len(self.features_cache)} machines from mini-cache.")
                return
            except Exception as e:
                logger.error(f"Error loading mini-cache: {e}. Rebuilding...")

        if not FEATURES_CSV_PATH.exists():
            logger.error(f"Engineered features file not found at: {FEATURES_CSV_PATH}. Caching skipped.")
            return

        try:
            logger.info(f"Rebuilding machine features cache from 1.7GB dataset: {FEATURES_CSV_PATH} ...")
            
            # Read in chunks of 100,000 rows to optimize memory and keep track of latest records
            chunksize = 100000
            temp_cache = {}
            
            for chunk in pd.read_csv(FEATURES_CSV_PATH, chunksize=chunksize):
                if "machineID" not in chunk.columns:
                    continue
                # Since the dataset is chronologically ordered, later chunks have newer timestamps
                chunk_latest = chunk.groupby("machineID").last().reset_index()
                for _, row in chunk_latest.iterrows():
                    m_id = int(row["machineID"])
                    temp_cache[m_id] = row.to_dict()
            
            self.features_cache = temp_cache
            logger.info(f"Successfully cached latest records for {len(self.features_cache)} machines.")
            
            # Save to mini-cache CSV for future fast startups
            if self.features_cache:
                cache_df = pd.DataFrame(list(self.features_cache.values()))
                cache_df.to_csv(MINI_CACHE_PATH, index=False)
                logger.info(f"Saved mini-cache to {MINI_CACHE_PATH}")
                
        except Exception as e:
            logger.error(f"Error rebuilding features cache: {e}")

    def _load_machines_data(self):
        """Load the basic machine list from PdM_machines.csv."""
        if not MACHINES_CSV_PATH.exists():
            logger.error(f"Machines CSV not found at: {MACHINES_CSV_PATH}")
            return
            
        try:
            self.machines_df = pd.read_csv(MACHINES_CSV_PATH)
            logger.info(f"Loaded machine profiles for {len(self.machines_df)} machines.")
        except Exception as e:
            logger.error(f"Error loading machines CSV: {e}")

    def get_machines_list(self):
        """Return the list of machines with their models, ages, and real predicted health status."""
        if self.machines_df is None:
            return []
            
        machines = self.machines_df.copy()
        
        # Derive machine status from real decision engine risk tiers
        try:
            from backend.services.decision_engine_service import DecisionEngineService
            de = DecisionEngineService()
            queue = de.get_health_queue()
            if queue:
                tier_map = {m["machine_id"]: m["risk_tier"] for m in queue}
                status_map = {
                    "CRITICAL": "Critical",
                    "WARNING":  "Warning",
                    "MONITOR":  "Healthy",
                }
                machines["status"] = machines["machineID"].map(
                    lambda m_id: status_map.get(tier_map.get(m_id, "MONITOR"), "Healthy")
                )
            else:
                machines["status"] = "Healthy"
        except Exception as e:
            logger.debug(f"Falling back to default Healthy status for machines: {e}")
            machines["status"] = "Healthy"

        return machines.to_dict(orient="records")

    def get_latest_features(self, machine_id: int) -> Optional[Dict[str, Any]]:
        """Return the latest cached feature dict for a specific machine ID."""
        if not self.features_cache and MINI_CACHE_PATH.exists():
            self._load_features_cache()
        return self.features_cache.get(machine_id)

    def prepare_inference_features(self, machine_id: int, overrides: Dict[str, float]) -> Optional[pd.DataFrame]:
        """
        Merge user sensor overrides with the cached latest feature row.
        Recomputes key ratio variables.
        """
        cached_row = self.get_latest_features(machine_id)
        if not cached_row:
            logger.warning(f"No cached features found for machine {machine_id}.")
            return None

        # Work on a copy of the cached dictionary
        features_dict = cached_row.copy()
        
        # Apply overrides (e.g. volt, rotate, pressure, vibration)
        for key, value in overrides.items():
            if key in features_dict:
                features_dict[key] = float(value)
                logger.debug(f"Overrode feature {key} = {value} for machine {machine_id}")

        # Recompute ratio features if telemetries are modified
        if "volt" in overrides or "rotate" in overrides:
            volt = features_dict.get("volt", 1.0)
            rotate = features_dict.get("rotate", 1.0)
            features_dict["volt_rotate_ratio"] = volt / (rotate if rotate != 0 else 1.0)

        if "pressure" in overrides or "vibration" in overrides:
            pressure = features_dict.get("pressure", 1.0)
            vibration = features_dict.get("vibration", 1.0)
            features_dict["pressure_vibration_ratio"] = pressure / (vibration if vibration != 0 else 1.0)

        # Convert to single-row DataFrame
        df = pd.DataFrame([features_dict])
        return df
