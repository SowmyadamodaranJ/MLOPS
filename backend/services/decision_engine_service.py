"""
decision_engine_service.py
---------------------------
Maintenance Decision Engine for Smart Factory Predictive Maintenance.
Operational decision layer that converts existing ML model predictions and SHAP
explanations into prioritized, actionable maintenance decisions.

Operational Note:
-----------------
Thresholds (CRITICAL >= 0.50, WARNING 0.25-0.50, MONITOR < 0.25) and Health Scores
are operational indicators derived from model statistical outputs. They do not
represent physical machine wear measurements or calibrated physical failure probabilities.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml
import numpy as np
import pandas as pd

from backend.services.model_service import ModelService
from backend.services.feature_service import FeatureService
from backend.services.explain_service import ExplainService
from backend.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class DecisionEngineService:
    """
    Singleton service implementing the Phase 3 Maintenance Decision Engine.
    Converts ML model predictions and SHAP explanations into operational decisions.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DecisionEngineService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model_service = ModelService()
        self.feature_service = FeatureService()
        self.explain_service = ExplainService()

        # Cache for fleet-wide health queue
        self._health_queue_cache: Optional[List[Dict[str, Any]]] = None
        self._health_queue_cache_time: float = 0.0
        self._cache_ttl: float = 60.0  # seconds

        # Load operational configuration
        self._load_config()
        self._initialized = True
        logger.info(
            f"DecisionEngineService initialized (Critical Threshold: {self.critical_threshold}, "
            f"Warning Threshold: {self.warning_threshold})."
        )

    def _load_config(self):
        """Load decision thresholds and settings from config.yaml, env vars, or defaults."""
        # Defaults
        self.critical_threshold: float = 0.50
        self.warning_threshold: float = 0.25
        self.health_score_scale: float = 100.0
        self.health_score_min: float = 0.0
        self.health_score_max: float = 100.0

        self.urgency_map: Dict[str, str] = {
            "CRITICAL": "Immediate",
            "WARNING": "Scheduled",
            "MONITOR": "Routine",
        }

        self.priority_map: Dict[str, str] = {
            "CRITICAL": "P1 - Critical",
            "WARNING": "P2 - High",
            "MONITOR": "P3 - Low",
        }

        # Attempt to read from configs/config.yaml
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                de_cfg = cfg.get("decision_engine", {})
                thresholds = de_cfg.get("thresholds", {})
                if "critical" in thresholds:
                    self.critical_threshold = float(thresholds["critical"])
                if "warning" in thresholds:
                    self.warning_threshold = float(thresholds["warning"])

                hs_cfg = de_cfg.get("health_score", {})
                if "scale" in hs_cfg:
                    self.health_score_scale = float(hs_cfg["scale"])
                if "min_score" in hs_cfg:
                    self.health_score_min = float(hs_cfg["min_score"])
                if "max_score" in hs_cfg:
                    self.health_score_max = float(hs_cfg["max_score"])

                if "urgency" in de_cfg:
                    self.urgency_map.update(de_cfg["urgency"])
                if "priority" in de_cfg:
                    self.priority_map.update(de_cfg["priority"])
            except Exception as exc:
                logger.warning(f"Could not load decision engine config from {CONFIG_PATH}: {exc}")

        # Environment variable overrides
        env_crit = os.environ.get("PDM_DECISION_CRITICAL_THRESHOLD")
        if env_crit is not None:
            try:
                self.critical_threshold = float(env_crit)
            except ValueError:
                pass

        env_warn = os.environ.get("PDM_DECISION_WARNING_THRESHOLD")
        if env_warn is not None:
            try:
                self.warning_threshold = float(env_warn)
            except ValueError:
                pass

    # ─── Operational Rules & Calculations ─────────────────────────────────────

    def calculate_health_score(self, failure_probability: float) -> float:
        """
        Calculate model-derived operational health score.
        Formula: health_score = 100 * (1 - failure_probability)
        Clamped to [0.0, 100.0].

        Note: Represents a statistical model-derived indicator, not a physical wear measurement.
        """
        raw_score = self.health_score_scale * (1.0 - float(failure_probability))
        clamped = max(self.health_score_min, min(self.health_score_max, raw_score))
        return round(clamped, 1)

    def determine_risk_tier(self, failure_probability: float) -> str:
        """
        Map failure probability to operational risk tier:
        - CRITICAL: failure_probability >= critical_threshold (default 0.50)
        - WARNING:  warning_threshold <= failure_probability < critical_threshold (default 0.25)
        - MONITOR:  failure_probability < warning_threshold
        """
        p = float(failure_probability)
        if p >= self.critical_threshold:
            return "CRITICAL"
        elif p >= self.warning_threshold:
            return "WARNING"
        return "MONITOR"

    def determine_urgency(self, risk_tier: str) -> str:
        """Return operational urgency string."""
        return self.urgency_map.get(risk_tier, "Routine")

    def determine_priority(self, risk_tier: str) -> str:
        """Return operational maintenance priority code."""
        return self.priority_map.get(risk_tier, "P3 - Low")

    def generate_maintenance_action(
        self,
        risk_tier: str,
        primary_driver_feature: Optional[str] = None
    ) -> str:
        """
        Generate evidence-based actionable recommendation without inventing
        physical component failures.
        """
        feature_context = ""
        if primary_driver_feature and primary_driver_feature != "none":
            feature_context = f" Review telemetry factor '{primary_driver_feature}' according to factory maintenance procedures."

        if risk_tier == "CRITICAL":
            return (
                "Recommend immediate maintenance inspection. "
                "Recommend prioritizing the machine in the maintenance queue."
                f"{feature_context}"
            )
        elif risk_tier == "WARNING":
            return (
                "Recommend scheduling maintenance inspection. "
                "Recommend increased monitoring."
                f"{feature_context}"
            )
        else:
            return (
                "Continue normal operation. "
                "Continue monitoring sensor profiles. "
                "No immediate maintenance escalation."
            )

    def extract_shap_drivers(
        self,
        contributions: Optional[List[Dict[str, Any]]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extract primary and secondary drivers from SHAP contributions.
        Each driver provides: feature, contribution, direction ('increases_risk' | 'decreases_risk' | 'neutral').
        """
        if not contributions:
            default_driver = {
                "feature": "none",
                "contribution": 0.0,
                "direction": "neutral",
            }
            return default_driver, []

        formatted_drivers = []
        for item in contributions:
            feat = item.get("feature", "unknown")
            shap_val = float(item.get("shap_value", 0.0))
            direction = (
                "increases_risk" if shap_val > 0.0 else
                ("decreases_risk" if shap_val < 0.0 else "neutral")
            )
            formatted_drivers.append({
                "feature": feat,
                "contribution": round(shap_val, 6),
                "direction": direction,
            })

        primary = formatted_drivers[0] if formatted_drivers else {
            "feature": "none",
            "contribution": 0.0,
            "direction": "neutral"
        }
        secondary = formatted_drivers[1:5]
        return primary, secondary

    def generate_reason(
        self,
        risk_tier: str,
        failure_probability: float,
        primary_driver: Dict[str, Any]
    ) -> str:
        """
        Generate transparent explanation without causal claims.
        Uses 'contributes to the model predicted risk' rather than 'causes the failure'.
        """
        feat_name = primary_driver.get("feature", "telemetry")
        contrib = primary_driver.get("contribution", 0.0)
        direction = primary_driver.get("direction", "neutral")

        if risk_tier == "CRITICAL":
            return (
                f"Operational failure probability ({failure_probability:.4f}) meets or exceeds "
                f"the critical operational threshold ({self.critical_threshold:.2f}). "
                f"Feature '{feat_name}' contributes most strongly ({contrib:+.4f}, {direction}) "
                "to the model's predicted risk."
            )
        elif risk_tier == "WARNING":
            return (
                f"Operational failure probability ({failure_probability:.4f}) is within the warning "
                f"threshold range [{self.warning_threshold:.2f}, {self.critical_threshold:.2f}). "
                f"Feature '{feat_name}' contributes most strongly ({contrib:+.4f}, {direction}) "
                "to the model's predicted risk."
            )
        else:
            return (
                f"Operational failure probability ({failure_probability:.4f}) remains below the "
                f"operational warning threshold ({self.warning_threshold:.2f}). "
                f"Operating parameters are within normal variance; feature '{feat_name}' "
                f"contributes ({contrib:+.4f}, {direction}) to the model's predicted risk."
            )

    def build_decision(
        self,
        machine_id: Any,
        failure_probability: float,
        prediction: int,
        contributions: Optional[List[Dict[str, Any]]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Construct a complete, structured decision object.
        Dynamically uses model metadata.
        """
        prob = float(failure_probability)
        health_score = self.calculate_health_score(prob)
        risk_tier = self.determine_risk_tier(prob)
        urgency = self.determine_urgency(risk_tier)
        priority = self.determine_priority(risk_tier)

        primary_driver, secondary_drivers = self.extract_shap_drivers(contributions)
        primary_feature = primary_driver.get("feature")

        action = self.generate_maintenance_action(risk_tier, primary_feature)
        reason = self.generate_reason(risk_tier, prob, primary_driver)

        # Dynamic model version
        version = model_version or self.model_service.version or "2.0.0"

        return {
            "machine_id": machine_id,
            "failure_probability": round(prob, 4),
            "health_score": health_score,
            "risk_tier": risk_tier,
            "maintenance_action": action,
            "recommended_action": action,  # Alias for consistency
            "urgency": urgency,
            "priority": priority,
            "primary_driver": primary_driver,
            "primary_driver_feature": primary_feature,
            "secondary_drivers": secondary_drivers,
            "reason": reason,
            "model_version": version,
            "prediction": int(prediction),
            "telemetry": telemetry or {},
            "operational_disclaimer": (
                "Model-derived operational health indicator based on statistical inference. "
                "Does not represent physical component diagnosis or calibrated failure probability."
            ),
        }

    # ─── High-Level Operations ───────────────────────────────────────────────

    def get_machine_decision(
        self,
        machine_id: int,
        overrides: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch real machine features, run live model inference & real SHAP explanation,
        and return the structured operational decision.
        Returns None if machine is not found.
        """
        # 1. Fetch real features for this machine
        features_df = self.feature_service.prepare_inference_features(machine_id, overrides or {})
        if features_df is None:
            return None

        # 2. Extract telemetry snapshot
        cached = self.feature_service.get_latest_features(machine_id) or {}
        telemetry = {
            "volt": cached.get("volt"),
            "rotate": cached.get("rotate"),
            "pressure": cached.get("pressure"),
            "vibration": cached.get("vibration"),
            "age": cached.get("age"),
            "model": cached.get("model"),
        }
        if overrides:
            telemetry.update({k: v for k, v in overrides.items() if k in telemetry})

        # 3. Model inference
        pipe = self.model_service.get_model()
        feature_names = self.model_service.get_feature_names()

        if pipe is None or not feature_names:
            logger.error("Model or feature names unavailable for decision.")
            return None

        X_infer = features_df[feature_names]

        prob = 0.0
        if hasattr(pipe, "predict_proba"):
            prob = float(pipe.predict_proba(X_infer)[0][1])
        prediction = int(pipe.predict(X_infer)[0])

        # 4. Real SHAP explanation
        contributions = []
        try:
            explanation = self.explain_service.get_local_explanation(machine_id, features_df)
            if explanation.get("status") == "success":
                contributions = explanation.get("contributions", [])
        except Exception as exc:
            logger.warning(f"Could not compute SHAP for machine {machine_id}: {exc}")

        # 5. Build decision
        return self.build_decision(
            machine_id=machine_id,
            failure_probability=prob,
            prediction=prediction,
            contributions=contributions,
            telemetry=telemetry,
            model_version=self.model_service.version,
        )

    def get_health_queue(
        self,
        risk_tier_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate operational maintenance priority queue for real machines in the fleet.
        Ordered by maintenance priority:
          1. Risk Tier (CRITICAL > WARNING > MONITOR)
          2. Failure Probability (descending)
          3. Health Score (ascending)

        Each item contains:
          machine_id, failure_probability, health_score, risk_tier, priority,
          recommended_action, primary_driver.
        """
        pipe = self.model_service.get_model()
        feature_names = self.model_service.get_feature_names()

        if not self.feature_service.features_cache or pipe is None or not feature_names:
            return []

        # Return cached queue if available and within TTL
        if not risk_tier_filter and (limit is None or limit <= 0):
            if self._health_queue_cache is not None and (time.time() - self._health_queue_cache_time) < self._cache_ttl:
                return [dict(item) for item in self._health_queue_cache]

        # Gather real machines
        machine_ids = sorted(list(self.feature_service.features_cache.keys()))
        df = pd.DataFrame([self.feature_service.features_cache[m] for m in machine_ids])
        X = df[feature_names]

        # Fast batch prediction
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X)[:, 1]
        else:
            probs = pipe.predict(X).astype(float)

        # Batch SHAP for all machines
        shap_top_drivers: Dict[int, Dict[str, Any]] = {}
        try:
            if self.explain_service.explainer is not None:
                sv = self.explain_service.explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
                sv = np.array(sv)
                top_indices = np.argmax(np.abs(sv), axis=1)

                for idx, m_id in enumerate(machine_ids):
                    feat_idx = top_indices[idx]
                    f_name = feature_names[feat_idx]
                    f_val = float(sv[idx, feat_idx])
                    direction = (
                        "increases_risk" if f_val > 0.0 else
                        ("decreases_risk" if f_val < 0.0 else "neutral")
                    )
                    shap_top_drivers[m_id] = {
                        "feature": f_name,
                        "contribution": round(f_val, 6),
                        "direction": direction,
                    }
        except Exception as exc:
            logger.warning(f"Batch SHAP computation in health queue encountered an issue: {exc}")

        tier_rank = {"CRITICAL": 0, "WARNING": 1, "MONITOR": 2}
        queue = []

        for m_id, prob in zip(machine_ids, probs):
            prob = float(prob)
            risk_tier = self.determine_risk_tier(prob)

            if risk_tier_filter and risk_tier != risk_tier_filter.upper():
                continue

            health_score = self.calculate_health_score(prob)
            priority = self.determine_priority(risk_tier)
            primary_driver = shap_top_drivers.get(m_id, {
                "feature": "none",
                "contribution": 0.0,
                "direction": "neutral",
            })
            action = self.generate_maintenance_action(risk_tier, primary_driver.get("feature"))

            queue.append({
                "machine_id": m_id,
                "failure_probability": round(prob, 4),
                "health_score": health_score,
                "risk_tier": risk_tier,
                "urgency": self.determine_urgency(risk_tier),
                "priority": priority,
                "recommended_action": action,
                "maintenance_action": action,
                "primary_driver": primary_driver,
                "primary_driver_feature": primary_driver.get("feature"),
                "_tier_rank": tier_rank.get(risk_tier, 99),
            })

        # Sort: Highest priority tier first, then highest failure_probability, then lowest health_score
        queue.sort(key=lambda x: (x["_tier_rank"], -x["failure_probability"], x["health_score"]))

        # Remove internal sort key
        for item in queue:
            if "_tier_rank" in item:
                del item["_tier_rank"]

        # Store full queue in cache
        if not risk_tier_filter and (limit is None or limit <= 0):
            self._health_queue_cache = [dict(item) for item in queue]
            self._health_queue_cache_time = time.time()

        if limit is not None and limit > 0:
            queue = queue[:limit]

        return queue
