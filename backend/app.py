"""
app.py
------
Main application factory and execution entry point for the Predictive
Maintenance API with enterprise observability middleware.

Security notes
--------------
- DEBUG mode is controlled exclusively via the FLASK_DEBUG environment
  variable (default: False).  Never set debug=True in source code.
- Allowed CORS origins are read from ALLOWED_ORIGINS (comma-separated);
  defaults to the local dev frontend only.
"""

import os
import sys
import time
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

# Ensure project root is on sys.path so backend.* imports resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.routes import api_bp
from backend.services.db_service import DatabaseService
from backend.services.model_service import ModelService
from backend.services.feature_service import FeatureService
from backend.services.monitoring_service import MonitoringService
from backend.services.drift_service import DriftService
from backend.services.alerts_service import AlertsService
from backend.services.mlflow_service import MLflowService
from backend.services.decision_engine_service import DecisionEngineService
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Application uptime anchor
APP_START_TIME: float = time.time()


def create_app() -> Flask:
    """
    Create and configure the Flask application with full observability.
    """
    app = Flask(__name__)

    # CORS — restrict origins via ALLOWED_ORIGINS env var
    # Default: localhost dev frontend only (not wildcard)
    _raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
    _allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": _allowed_origins}})

    # Blueprint
    app.register_blueprint(api_bp, url_prefix="/api")

    # Service initialisation
    logger.info("Initialising enterprise monitoring services and caches...")
    db_service              = DatabaseService()
    model_service           = ModelService()
    feature_service         = FeatureService()
    monitoring_service      = MonitoringService()
    drift_service           = DriftService()
    alerts_service          = AlertsService()
    mlflow_service          = MLflowService()
    decision_engine_service = DecisionEngineService()

    # Request-timing & Observability middleware
    @app.before_request
    def start_timer():
        """Record the request start time for latency calculation."""
        request.start_time = time.time()

    @app.after_request
    def log_api_request(response):
        """Log request to console, DB, and record telemetry in MonitoringService."""
        latency = 0.0
        if hasattr(request, "start_time"):
            latency = (time.time() - request.start_time) * 1000.0

        is_success = response.status_code < 400
        monitoring_service.record_request(latency_ms=round(latency, 2), success=is_success)

        logger.info(
            f"REQUEST: {request.method} {request.path} | "
            f"Status: {response.status_code} | "
            f"Latency: {latency:.2f}ms | "
            f"IP: {request.remote_addr}"
        )

        # Persist to database — skip health checks to avoid noise
        if "/api/health" not in request.path and "/api/monitoring" not in request.path:
            db_service.log_request(
                method=request.method,
                path=request.path,
                status_code=response.status_code,
                latency_ms=round(latency, 2),
            )

        return response

    # Root health endpoint
    @app.route("/health", methods=["GET"])
    def root_health():
        """Lightweight root status check."""
        return jsonify({
            "success": True,
            "data": {
                "status":  "healthy",
                "service": "smart-factory-pdm-backend",
                "version": "2.0.0",
            },
            "message": "Backend is running.",
        })

    # Global error handlers
    @app.errorhandler(400)
    def bad_request(error):
        logger.warning(f"400 Bad Request on {request.path}: {error}")
        return jsonify({
            "success":    False,
            "message":    str(error) or "The request body is malformed.",
            "error_code": "BAD_REQUEST",
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success":    False,
            "message":    f"Endpoint '{request.path}' does not exist.",
            "error_code": "NOT_FOUND",
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "success":    False,
            "message":    f"HTTP method '{request.method}' is not allowed on '{request.path}'.",
            "error_code": "METHOD_NOT_ALLOWED",
        }), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        err_msg = str(error)
        logger.error(f"500 Internal Server Error on {request.path}: {err_msg}")
        db_service.log_exception(
            endpoint=request.path,
            error_message=err_msg,
            stack_trace="Caught by 500 error handler",
        )
        return jsonify({
            "success":    False,
            "message":    "An internal server error occurred. The incident has been logged.",
            "error_code": "INTERNAL_SERVER_ERROR",
        }), 500

    @app.errorhandler(Exception)
    def handle_unhandled_exception(exc: Exception):
        import traceback as tb

        stack = tb.format_exc()
        err_msg = str(exc)

        logger.error(
            f"UNHANDLED EXCEPTION on {request.path}: {type(exc).__name__}: {err_msg}\n{stack}"
        )

        try:
            db_service.log_exception(
                endpoint=request.path,
                error_message=f"{type(exc).__name__}: {err_msg}",
                stack_trace=stack,
            )
        except Exception:
            pass

        return jsonify({
            "success":    False,
            "message":    "An unexpected server error occurred. The incident has been logged.",
            "error_code": "INTERNAL_SERVER_ERROR",
        }), 500

    # Startup summary
    artifacts = model_service.get_artifacts_status()
    logger.info("=" * 60)
    logger.info("Smart Factory PDM Backend — STARTUP SUMMARY")
    logger.info(f"  Model ready    : {model_service.model_ready}")
    logger.info(f"  Algorithm      : {model_service.algorithm}")
    logger.info(f"  DB connected   : {db_service.check_connection()}")
    logger.info(f"  MLflow status  : {'Available' if mlflow_service.check_mlflow_available() else 'Unavailable'}")
    for art_name, present in artifacts.items():
        status_str = "OK" if present else "MISSING"
        logger.info(f"  Artifact [{status_str}] : {art_name}")
    logger.info("=" * 60)

    return app


if __name__ == "__main__":
    app = create_app()
    # DEBUG is driven by environment variable only — never hard-coded.
    # Set FLASK_DEBUG=1 in .env for local development.
    _debug = os.environ.get("FLASK_DEBUG", "0").strip() == "1"
    _port  = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(host="0.0.0.0", port=_port, debug=_debug)
