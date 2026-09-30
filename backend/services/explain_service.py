"""
explain_service.py
------------------
Service to compute local SHAP explanations for machine failure predictions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import shap

from backend.services.model_service import ModelService
from backend.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

class ExplainService:
    """
    Computes local SHAP values and feature contributions for predictions.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ExplainService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.model_service = ModelService()
        self.background_data: Optional[np.ndarray] = None
        self.explainer = None
        self._init_explainer()
        self._initialized = True

    def _transform_features(self, pipe, X: pd.DataFrame) -> np.ndarray:
        """Apply all preprocessing steps in the pipeline except the estimator."""
        X_trans = X.copy()
        # Apply scaling and other steps
        for name, step in pipe.steps[:-1]:
            X_trans = step.transform(X_trans)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        return np.array(X_trans)

    def _init_explainer(self):
        """Initialize the SHAP explainer with background data."""
        pipe = self.model_service.get_model()
        if pipe is None:
            logger.error("SHAP explainer cannot start because model is not loaded.")
            return

        clf = pipe.named_steps.get("clf")
        feature_names = self.model_service.get_feature_names()

        if not feature_names:
            logger.warning("No feature names found. Cannot initialize background data.")
            return

        try:
            # Load a small sample from test_data.csv to use as SHAP background
            if TEST_DATA_PATH.exists():
                logger.info("Loading SHAP background data from test dataset...")
                test_df = pd.read_csv(TEST_DATA_PATH, nrows=100)
                # Keep only valid features
                test_features = test_df[[c for c in feature_names if c in test_df.columns]]
                
                # Transform features using preprocessing pipeline
                self.background_data = self._transform_features(pipe, test_features)
                
                logger.info(f"SHAP background initialized with shape: {self.background_data.shape}")
                
                # Create explainer
                self.explainer = shap.TreeExplainer(clf, data=self.background_data)
            else:
                logger.warning("Test dataset not found for SHAP background. Using tree model explainer directly.")
                self.explainer = shap.TreeExplainer(clf)
                
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}. Fallbacks will be used.")
            self.explainer = None

    def get_local_explanation(self, machine_id: int, X_row_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate local SHAP values for a single prediction row.
        Returns feature contributions (top contributing features and shap values).
        """
        pipe = self.model_service.get_model()
        feature_names = self.model_service.get_feature_names()

        if pipe is None or not feature_names:
            return self._get_fallback_explanation(X_row_df, "Model or features not loaded")

        try:
            # Verify feature columns match the model expectations
            missing_cols = [c for c in feature_names if c not in X_row_df.columns]
            if missing_cols:
                logger.error(f"Input features are missing required columns: {missing_cols}")
                return self._get_fallback_explanation(X_row_df, f"Missing columns: {missing_cols}")

            # Align columns
            df_aligned = X_row_df[feature_names]
            
            # Preprocess the input row
            X_trans = self._transform_features(pipe, df_aligned)
            
            if self.explainer is None:
                # Retry initializing if not done
                self._init_explainer()
                
            if self.explainer is not None:
                # Compute SHAP values
                # sv is shape (1, num_features) or list of class arrays
                sv = self.explainer.shap_values(X_trans)
                
                # For classification, check if it returned a list of class-specific SHAP values
                if isinstance(sv, list):
                    sv = sv[1]  # positive class (Failure)
                sv = np.array(sv).flatten()
                
                # Pair with feature names
                contributions = []
                for name, val in zip(feature_names, sv):
                    contributions.append({
                        "feature": name,
                        "shap_value": round(float(val), 6),
                        "abs_value": abs(float(val))
                    })
                    
                # Sort by absolute SHAP values descending
                contributions = sorted(contributions, key=lambda x: x["abs_value"], reverse=True)
                
                # Get prediction & probability
                pred = int(pipe.predict(df_aligned)[0])
                prob = float(pipe.predict_proba(df_aligned)[0][1]) if hasattr(pipe, "predict_proba") else float(pred)
                
                logger.info(f"SHAP explanation successfully generated for machine {machine_id}")
                return {
                    "status": "success",
                    "machine_id": machine_id,
                    "prediction": pred,
                    "probability": round(prob, 4),
                    "base_value": float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value),
                    "contributions": contributions[:15],  # Return top 15 features to frontend
                }
            else:
                return self._get_fallback_explanation(df_aligned, "SHAP explainer not initialized")

        except Exception as e:
            logger.error(f"Error computing local SHAP explanation: {e}")
            return self._get_fallback_explanation(X_row_df, str(e))

    def _get_fallback_explanation(self, X_row_df: pd.DataFrame, reason: str) -> Dict[str, Any]:
        """Generate a rule-based fallback feature importance when SHAP is unavailable."""
        logger.warning(f"Generating fallback explanation. Reason: {reason}")
        
        # Load feature importance from the pipeline if possible
        pipe = self.model_service.get_model()
        feature_names = self.model_service.get_feature_names()
        
        contributions = []
        if pipe is not None and feature_names:
            try:
                clf = pipe.named_steps.get("clf")
                importances = None
                if hasattr(clf, "feature_importances_"):
                    importances = clf.feature_importances_
                elif hasattr(clf, "coef_"):
                    importances = clf.coef_[0]
                    
                if importances is not None:
                    for name, imp in zip(feature_names, importances):
                        # Direction based on telemetry reading difference from mean
                        # (e.g. if volt is high, increase impact)
                        val = float(imp) * 0.1
                        contributions.append({
                            "feature": name,
                            "shap_value": round(val, 6),
                            "abs_value": abs(val)
                        })
            except Exception as e:
                logger.error(f"Failed to get fallback importances: {e}")

        # If still empty, return simple static importance based on key sensors
        if not contributions:
            key_sensors = ["vibration", "volt", "rotate", "pressure", "age"]
            for s in key_sensors:
                val = 0.25 if s == "vibration" else (0.15 if s == "volt" else 0.05)
                contributions.append({
                    "feature": s,
                    "shap_value": val,
                    "abs_value": abs(val)
                })

        contributions = sorted(contributions, key=lambda x: x["abs_value"], reverse=True)
        
        pred = 0
        prob = 0.05
        if pipe is not None:
            try:
                pred = int(pipe.predict(X_row_df)[0])
                prob = float(pipe.predict_proba(X_row_df)[0][1]) if hasattr(pipe, "predict_proba") else float(pred)
            except:
                pass
                
        return {
            "status": "fallback",
            "message": f"SHAP unavailable ({reason}). Rule-based analysis shown.",
            "prediction": pred,
            "probability": round(prob, 4),
            "base_value": 0.02,
            "contributions": contributions[:15]
        }
