"""
model_service.py
----------------
Service to load, cache, and query the best predictive maintenance model.

Production-readiness additions
-------------------------------
- _validate_artifacts() checks all four required artifact files at startup.
- self.artifacts_status exposes per-artifact availability to /health.
- self.model_ready is the single authoritative flag checked before any
  inference call.  When False, get_model() returns None and routes return
  a 503 with error_code MODEL_NOT_LOADED.
- Every missing artifact is logged at ERROR level so the issue is
  immediately visible in the server log.
"""

from pathlib import Path
import joblib
import json
from datetime import datetime
from typing import Dict
from backend.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models" / "saved_models"
METRICS_DIR  = PROJECT_ROOT / "reports" / "metrics"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "artifact_manifest.json"
MODEL_MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

# ─── Required artifact registry ──────────────────────────────────────────────
# Maps a human-readable key to the file path that must exist before the
# service is considered "ready".  best_model.joblib is the primary model;
# the .pkl copies of preprocessor, feature_names, and label_encoder are also
# required because the inference pipeline references them at prediction time.
REQUIRED_ARTIFACTS: Dict[str, Path] = {
    "best_model.joblib": MODELS_DIR / "best_model.joblib",
    "preprocessor.pkl":  MODELS_DIR / "preprocessor.pkl",
    "feature_names.pkl": MODELS_DIR / "feature_names.pkl",
    "label_encoder.pkl": MODELS_DIR / "label_encoder.pkl",
}


class ModelService:
    """
    Singleton that loads, validates, and serves the ML pipeline artifact.

    Attributes
    ----------
    model_ready : bool
        True only when all required artifacts exist AND the model loaded
        without errors.  Routes must check this before running inference.
    artifacts_status : dict[str, bool]
        Per-artifact existence flag exposed to /health.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model          = None
        self.metadata       = {}
        self.training_date  = None
        self.version        = "2.0.0"
        self.algorithm      = "Unknown"
        self.model_source   = "Not loaded"
        self.mlflow_run_id  = "N/A"
        self.mlflow_model_uri = "N/A"
        self.model_ready    = False                       # authoritative readiness flag
        self.artifacts_status: Dict[str, bool] = {}      # exposed to /health

        # Run startup sequence
        self._validate_artifacts()
        self._load_model_and_metadata()
        self._initialized = True

    # ─── Startup validation ───────────────────────────────────────────────────

    def _validate_artifacts(self) -> None:
        """
        Check that every required artifact file exists on disk.

        Logs an ERROR for each missing file so the problem is visible
        immediately in the server log without needing to hit an endpoint.
        Sets self.artifacts_status and the preliminary self.model_ready.
        """
        logger.info("Validating required ML artifacts at startup...")
        all_present = True

        for name, path in REQUIRED_ARTIFACTS.items():
            exists = path.exists()
            self.artifacts_status[name] = exists
            if exists:
                logger.info(f"  [OK]     {name} — found at {path}")
            else:
                logger.error(
                    f"  [MISSING] {name} — expected at {path}. "
                    "Prediction requests will be blocked until this file is restored."
                )
                all_present = False

        if all_present:
            logger.info("All required ML artifacts present.")
        else:
            missing = [k for k, v in self.artifacts_status.items() if not v]
            logger.error(
                f"Startup artifact validation FAILED. Missing: {missing}. "
                "The /predict and /explain endpoints will return 503 until resolved."
            )

        # model_ready will be finalised after the actual joblib load succeeds
        self.model_ready = all_present

    # ─── Model & metadata loading ─────────────────────────────────────────────

    def _load_model_and_metadata(self) -> None:
        """Load the model pipeline and its metadata from MLflow / disk."""
        model_path = MODELS_DIR / "best_model.joblib"
        meta_path  = METRICS_DIR / "best_model_meta.json"

        # 1. Load from model_manifest.json as authoritative source of truth
        if MODEL_MANIFEST_PATH.exists():
            try:
                with open(MODEL_MANIFEST_PATH, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                self.algorithm = manifest.get("algorithm", "XGBoost")
                self.version = manifest.get("model_version", "2.0.0")
                self.training_date = manifest.get("training_timestamp")
                self.mlflow_run_id = manifest.get("mlflow_run_id", "N/A")
                self.mlflow_model_uri = manifest.get("mlflow_model_uri", "N/A")
                self.metadata["feature_names"] = manifest.get("feature_names", [])
                self.metadata["optimal_threshold"] = manifest.get("optimal_threshold", 0.5)
                self.metadata["split_strategy"] = manifest.get("split_strategy", "chronological_70_30")
                self.metadata["selected_metric_value"] = manifest.get("selected_metric_value")
                logger.info(f"Loaded metadata from model_manifest.json: Algorithm={self.algorithm}, Version={self.version}, Features={len(self.metadata['feature_names'])}")
            except Exception as e:
                logger.error(f"Error parsing model_manifest.json: {e}")

        # Fallback metadata if not yet loaded
        if not self.algorithm or self.algorithm == "Unknown":
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                    self.algorithm = self.metadata.get("model_name", "Unknown")
                    logger.info(f"Loaded model metadata from best_model_meta.json: {self.algorithm}")
                except Exception as e:
                    logger.error(f"Error loading model metadata: {e}")

        if not self.training_date and MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                gen_at = manifest.get("generated_at")
                if gen_at:
                    self.training_date = gen_at
                    logger.info(f"Loaded training date from manifest: {self.training_date}")
            except Exception as e:
                logger.error(f"Error parsing artifact manifest: {e}")

        if not self.training_date and model_path.exists():
            mtime = model_path.stat().st_mtime
            self.training_date = datetime.fromtimestamp(mtime).isoformat()
            logger.info(f"Using file modification time as training date: {self.training_date}")

        # Load the serialised sklearn Pipeline
        if not self.model_ready:
            # Artifact validation already failed — skip loading
            logger.error(
                "Skipping model load because one or more artifacts are missing."
            )
            return

        # 2. Preferred resolution: MLflow registered/serving model -> XGBoost 2.0.0
        #    Try @champion alias first, then fall back to runs:/ URI, then local disk.
        loaded_via_mlflow = False
        mlflow_uris_to_try = []

        # Highest priority: @champion alias from the registry
        registry_name = "PdM_BestModel"
        mlflow_uris_to_try.append(f"models:/{registry_name}@champion")

        # Second priority: runs:/ URI from the manifest
        if self.mlflow_model_uri and self.mlflow_model_uri != "N/A":
            mlflow_uris_to_try.append(self.mlflow_model_uri)

        for uri in mlflow_uris_to_try:
            try:
                import mlflow.sklearn
                mlflow.set_tracking_uri(f"sqlite:///{MODELS_DIR.parents[1] / 'mlflow.db'}")
                logger.info(f"Attempting to load model from MLflow: {uri} ...")
                self.model = mlflow.sklearn.load_model(uri)
                self.model_source = f"MLflow Model Registry ({uri})"
                self.model_ready = True
                loaded_via_mlflow = True
                logger.info(f"Successfully loaded model from MLflow via: {uri}")
                break
            except Exception as mlflow_exc:
                logger.warning(
                    f"MLflow model load failed for '{uri}': {mlflow_exc}. "
                    f"Trying next source..."
                )

        if not loaded_via_mlflow:
            try:
                logger.info(f"Loading best model from {model_path} ...")
                self.model = joblib.load(model_path)
                self.model_source = f"Local disk ({model_path})"
                logger.info("Best model successfully loaded into memory from local disk.")
                self.model_ready = True
            except Exception as e:
                logger.error(f"Error loading model joblib: {e}")
                self.model       = None
                self.model_ready = False
                # Update artifact status to reflect the load failure
                self.artifacts_status["best_model.joblib"] = False

        # 3. Print / Log startup info banner
        feature_count = len(self.get_feature_names())
        banner = (
            f"\n============================================================\n"
            f"MODEL SOURCE:       {self.model_source}\n"
            f"MODEL ALGORITHM:    {self.algorithm}\n"
            f"MODEL VERSION:      {self.version}\n"
            f"TRAINING TIMESTAMP: {self.training_date}\n"
            f"FEATURE COUNT:      {feature_count}\n"
            f"MLFLOW RUN ID:      {self.mlflow_run_id}\n"
            f"MODEL URI:          {self.mlflow_model_uri}\n"
            f"============================================================"
        )
        logger.info(banner)
        print(banner)

    # ─── Public interface ─────────────────────────────────────────────────────

    def get_model(self):
        """
        Return the loaded sklearn Pipeline, or None if not ready.

        Routes must treat None as a 503 condition (model not loaded).
        """
        return self.model

    def get_feature_names(self) -> list:
        """Return the list of feature names expected by the model."""
        features = self.metadata.get("feature_names", [])
        if not features:
            fn_path = MODELS_DIR / "feature_names.pkl"
            if fn_path.exists():
                try:
                    features = joblib.load(fn_path)
                    self.metadata["feature_names"] = features
                except Exception as e:
                    logger.error(f"Error loading feature_names.pkl: {e}")
        return features

    def validate_feature_contract(self, df) -> tuple:
        """
        Validate that the inference DataFrame satisfies the exact feature contract:
        - Exact feature count
        - Exact feature names
        - Exact ordering
        - Compatible numeric dtypes
        """
        import numpy as np

        expected = self.get_feature_names()
        expected_len = len(expected)

        cols = list(df.columns)
        if len(cols) != expected_len:
            return False, f"Feature contract error: input DataFrame has {len(cols)} columns, expected exactly {expected_len}."

        if cols != expected:
            missing = [f for f in expected if f not in cols]
            extra = [f for f in cols if f not in expected]
            mismatches = [f"Col {i}: got '{c}' != expected '{e}'" for i, (c, e) in enumerate(zip(cols, expected)) if c != e]
            err_details = []
            if missing:
                err_details.append(f"Missing features: {missing}")
            if extra:
                err_details.append(f"Unexpected features: {extra}")
            if mismatches:
                err_details.append(f"Order mismatch: {mismatches[:3]}")
            return False, f"Feature contract violation: {'; '.join(err_details)}"

        # Check numeric dtypes
        for col in cols:
            if not np.issubdtype(df[col].dtype, np.number):
                return False, f"Feature contract violation: column '{col}' has non-numeric dtype '{df[col].dtype}'."

        return True, ""

    def get_optimal_threshold(self) -> float:
        """Return the optimal classification threshold from metadata, defaulting to 0.5."""
        return float(self.metadata.get("optimal_threshold", 0.5))

    def get_artifacts_status(self) -> Dict[str, bool]:
        """Return per-artifact availability dict for the /health endpoint."""
        return dict(self.artifacts_status)

    def get_model_info(self) -> dict:
        """Return model metadata dictionary for HTTP endpoints."""
        return {
            "active_model":      f"{self.algorithm} Pipeline",
            "version":           self.version,
            "algorithm":         self.algorithm,
            "training_date":     self.training_date or "Unknown",
            "feature_count":     len(self.get_feature_names()),
            "optimal_threshold": self.get_optimal_threshold(),
            "split_strategy":    self.metadata.get("split_strategy", "Unknown"),
            "status":            "Loaded" if self.model_ready else "Not Loaded",
            "model_ready":       self.model_ready,
            "model_source":      self.model_source,
            "mlflow_run_id":     self.mlflow_run_id,
            "model_uri":         self.mlflow_model_uri,
        }
