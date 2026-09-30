"""
monitoring_service.py
----------------------
Service for tracking system resource utilization (CPU, RAM, Disk), API latency,
throughput, prediction error rates, and model performance metrics.
"""

import time
import os
from typing import Dict, Any, List
from collections import deque
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MonitoringService:
    """
    Singleton monitoring service to manage application & model metrics.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MonitoringService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.latency_window: deque = deque(maxlen=200)       # last 200 request latencies in ms
        self.request_timestamps: deque = deque(maxlen=500)   # timestamps of recent requests for throughput
        self.total_predictions: int = 0
        self.prediction_failures: int = 0
        self.prediction_successes: int = 0

        # Model metrics window
        self.recent_predictions: deque = deque(maxlen=200)   # 0 or 1
        self.recent_confidences: deque = deque(maxlen=200)     # confidence scores [0.5, 1.0]
        self.recent_probabilities: deque = deque(maxlen=200)   # failure probabilities [0.0, 1.0]
        self.recent_features: Dict[str, deque] = {
            "volt": deque(maxlen=200),
            "rotate": deque(maxlen=200),
            "pressure": deque(maxlen=200),
            "vibration": deque(maxlen=200),
        }

        self._initialized = True
        logger.info("MonitoringService initialized successfully.")

    def record_request(self, latency_ms: float, success: bool = True):
        """Record an API request latency and timestamp."""
        now = time.time()
        self.latency_window.append(latency_ms)
        self.request_timestamps.append(now)

    def record_prediction(self, prediction: int, probability: float, confidence: float, features: Dict[str, float], success: bool = True):
        """Record details of a single model inference run."""
        self.total_predictions += 1
        if success:
            self.prediction_successes += 1
        else:
            self.prediction_failures += 1

        self.recent_predictions.append(prediction)
        self.recent_probabilities.append(probability)
        self.recent_confidences.append(confidence)

        for key in ["volt", "rotate", "pressure", "vibration"]:
            if key in features and features[key] is not None:
                self.recent_features[key].append(float(features[key]))

    def get_system_metrics(self) -> Dict[str, Any]:
        """Fetch current hardware and system performance metrics."""
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_used_mb = 0.0
        memory_total_mb = 0.0
        disk_percent = 0.0

        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                memory_percent = mem.percent
                memory_used_mb = round(mem.used / (1024 * 1024), 1)
                memory_total_mb = round(mem.total / (1024 * 1024), 1)

                disk = psutil.disk_usage("/")
                disk_percent = disk.percent
            except Exception as e:
                logger.warning(f"Error reading psutil metrics: {e}")
        else:
            # Fallback mock values if psutil is not available
            cpu_percent = 18.4
            memory_percent = 42.1
            memory_used_mb = 3450.0
            memory_total_mb = 8192.0
            disk_percent = 35.8

        # Latency calculations
        avg_latency = round(sum(self.latency_window) / len(self.latency_window), 2) if self.latency_window else 15.0
        p95_latency = round(float(np.percentile(list(self.latency_window), 95)), 2) if len(self.latency_window) >= 5 else avg_latency

        # Request throughput (requests per minute in the last 60s)
        now = time.time()
        recent_reqs = sum(1 for ts in self.request_timestamps if now - ts <= 60.0)
        throughput_rpm = recent_reqs

        total_preds = max(self.total_predictions, 1)
        success_rate = round((self.prediction_successes / total_preds) * 100.0, 2) if self.total_predictions > 0 else 100.0
        failure_rate = round((self.prediction_failures / total_preds) * 100.0, 2) if self.total_predictions > 0 else 0.0

        return {
            "psutil_available": PSUTIL_AVAILABLE,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_mb": memory_used_mb,
            "memory_total_mb": memory_total_mb,
            "disk_percent": disk_percent,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "throughput_rpm": throughput_rpm,
            "prediction_count": self.total_predictions,
            "prediction_success_rate": success_rate,
            "prediction_failure_rate": failure_rate,
        }

    def get_model_monitoring_metrics(self) -> Dict[str, Any]:
        """Fetch statistical distributions and health indicators for model inference."""
        preds = list(self.recent_predictions)
        probs = list(self.recent_probabilities)
        confs = list(self.recent_confidences)

        total_recent = len(preds)
        if total_recent == 0:
            # Baseline fallbacks when no predictions recorded in window yet
            return {
                "sample_count": 0,
                "failure_prediction_rate": 0.05,
                "normal_prediction_rate": 0.95,
                "avg_confidence": 0.942,
                "confidence_distribution": {"high": 85, "medium": 12, "low": 3},
                "prediction_distribution": {"normal": 95, "failure": 5},
                "feature_statistics": {
                    "volt": {"mean": 170.8, "std": 14.2, "min": 120.0, "max": 220.0},
                    "rotate": {"mean": 448.2, "std": 52.1, "min": 250.0, "max": 600.0},
                    "pressure": {"mean": 100.4, "std": 11.5, "min": 60.0, "max": 150.0},
                    "vibration": {"mean": 40.1, "std": 5.8, "min": 20.0, "max": 75.0},
                },
                "abnormal_behavior_detected": False,
                "abnormal_reasons": []
            }

        failures = sum(preds)
        failure_rate = round(failures / total_recent, 4)
        avg_conf = round(sum(confs) / total_recent, 4)

        high_conf = sum(1 for c in confs if c >= 0.85)
        med_conf = sum(1 for c in confs if 0.65 <= c < 0.85)
        low_conf = sum(1 for c in confs if c < 0.65)

        feature_stats = {}
        for feat, vals in self.recent_features.items():
            if len(vals) > 0:
                arr = np.array(list(vals))
                feature_stats[feat] = {
                    "mean": round(float(np.mean(arr)), 2),
                    "std": round(float(np.std(arr)), 2),
                    "min": round(float(np.min(arr)), 2),
                    "max": round(float(np.max(arr)), 2),
                }
            else:
                feature_stats[feat] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        abnormal_reasons = []
        if failure_rate > 0.4:
            abnormal_reasons.append(f"Unusually high failure prediction rate ({failure_rate*100:.1f}%)")
        if avg_conf < 0.70:
            abnormal_reasons.append(f"Low average model confidence ({avg_conf:.2f})")

        return {
            "sample_count": total_recent,
            "failure_prediction_rate": failure_rate,
            "normal_prediction_rate": round(1.0 - failure_rate, 4),
            "avg_confidence": avg_conf,
            "confidence_distribution": {"high": high_conf, "medium": med_conf, "low": low_conf},
            "prediction_distribution": {"normal": total_recent - failures, "failure": failures},
            "feature_statistics": feature_stats,
            "abnormal_behavior_detected": len(abnormal_reasons) > 0,
            "abnormal_reasons": abnormal_reasons,
        }
