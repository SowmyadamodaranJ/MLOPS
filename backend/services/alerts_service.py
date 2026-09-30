"""
alerts_service.py
-----------------
Automated enterprise alerting engine for system resources, model readiness,
data drift, prediction failures, and service dependencies.
"""

import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AlertsService:
    """
    Singleton alerts service to evaluate system health rules and manage alert state.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AlertsService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.acknowledged_alert_ids: set = set()
        self.custom_alerts: List[Dict[str, Any]] = []
        self._initialized = True
        logger.info("AlertsService initialized successfully.")

    def evaluate_system_alerts(self, system_metrics: Dict[str, Any], model_ready: bool,
                               artifacts_status: Dict[str, bool], db_connected: bool,
                               drift_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate active rules and return all triggered active alerts.
        """
        alerts = []

        # 1. Model Unavailable
        if not model_ready:
            alerts.append({
                "id": "ALERT_MODEL_UNAVAILABLE",
                "severity": "CRITICAL",
                "category": "Model",
                "title": "Model Unavailable",
                "message": "The primary ML prediction model pipeline is not loaded in memory.",
                "timestamp": datetime.now().isoformat(),
                "action": "Inspect model artifacts in /models/saved_models/ and verify startup logs.",
            })

        # 2. Missing Model Artifacts
        missing_artifacts = [k for k, v in artifacts_status.items() if not v]
        if missing_artifacts:
            alerts.append({
                "id": "ALERT_MISSING_ARTIFACTS",
                "severity": "HIGH",
                "category": "Model",
                "title": "Missing Model Artifacts",
                "message": f"Required ML artifacts are missing: {', '.join(missing_artifacts)}",
                "timestamp": datetime.now().isoformat(),
                "action": "Re-run the training pipeline (run_pipeline.py) to regenerate joblib/pkl artifacts.",
            })

        # 3. Database Disconnected
        if not db_connected:
            alerts.append({
                "id": "ALERT_DB_DISCONNECTED",
                "severity": "CRITICAL",
                "category": "Database",
                "title": "Database Disconnected",
                "message": "SQLite storage backend predictions.db is unreachable or locked.",
                "timestamp": datetime.now().isoformat(),
                "action": "Check directory permissions on data/predictions.db.",
            })

        # 4. High Memory Usage
        mem_pct = system_metrics.get("memory_percent", 0.0)
        if mem_pct > 85.0:
            alerts.append({
                "id": "ALERT_HIGH_MEMORY",
                "severity": "HIGH",
                "category": "Infrastructure",
                "title": "High Memory Usage",
                "message": f"RAM utilization has reached {mem_pct}% (threshold: 85%).",
                "timestamp": datetime.now().isoformat(),
                "action": "Scale memory allocation or optimize cached dataset sizes.",
            })

        # 5. High API Latency
        latency = system_metrics.get("avg_latency_ms", 0.0)
        if latency > 150.0:
            alerts.append({
                "id": "ALERT_HIGH_LATENCY",
                "severity": "MEDIUM",
                "category": "API",
                "title": "High API Latency",
                "message": f"Average API response time is {latency}ms (threshold: 150ms).",
                "timestamp": datetime.now().isoformat(),
                "action": "Optimize database queries and feature extraction latency.",
            })

        # 6. Prediction Failures
        fail_count = system_metrics.get("prediction_failure_rate", 0.0)
        if fail_count > 5.0:  # > 5% failure rate
            alerts.append({
                "id": "ALERT_PREDICTION_FAILURES",
                "severity": "HIGH",
                "category": "Inference",
                "title": "Prediction Failure Spike",
                "message": f"Prediction error rate is currently {fail_count}% of total requests.",
                "timestamp": datetime.now().isoformat(),
                "action": "Review exceptions log in SQLite database for /predict route failures.",
            })

        # 7. Data Drift Detected
        data_drift = drift_summary.get("data_drift", {}).get("detected", False)
        if data_drift:
            drifted_cnt = drift_summary.get("data_drift", {}).get("number_of_drifted_features", 0)
            alerts.append({
                "id": "ALERT_DATA_DRIFT",
                "severity": "MEDIUM",
                "category": "Data Drift",
                "title": "Data Drift Detected",
                "message": f"Feature distribution shift detected across {drifted_cnt} telemetry inputs.",
                "timestamp": datetime.now().isoformat(),
                "action": "Inspect Evidently AI report and evaluate model retraining requirements.",
            })

        # 8. Model Drift Detected
        model_drift = drift_summary.get("model_drift", {}).get("detected", False)
        if model_drift:
            alerts.append({
                "id": "ALERT_MODEL_DRIFT",
                "severity": "HIGH",
                "category": "Model Drift",
                "title": "Model Drift & Performance Degradation",
                "message": "Production prediction distributions diverge significantly from baseline training data.",
                "timestamp": datetime.now().isoformat(),
                "action": "Trigger hyperparameter retune or model re-registration in MLflow.",
            })

        # Filter out acknowledged alerts
        active_alerts = []
        for a in alerts:
            a["acknowledged"] = a["id"] in self.acknowledged_alert_ids
            active_alerts.append(a)

        return active_alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged by human operator."""
        self.acknowledged_alert_ids.add(alert_id)
        logger.info(f"Alert acknowledged: {alert_id}")
        return True

    def clear_acknowledged(self):
        """Clear acknowledged state."""
        self.acknowledged_alert_ids.clear()
