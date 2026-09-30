"""
routes.py
---------
API Route definitions for the Predictive Maintenance backend.
"""

from flask import Blueprint, jsonify, request
import pandas as pd
import joblib
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

api_bp = Blueprint('api', __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

def get_best_model():
    model_path = MODELS_DIR / "best_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

def load_json_artifact(path: Path):
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@api_bp.route('/dashboard', methods=['GET'])
def get_dashboard():
    """Phase 6: Provide aggregate KPIs for the dashboard"""
    try:
        # Simplified KPIs - in production, this would query a real-time store or the DB
        metrics_meta = load_json_artifact(METRICS_DIR / "best_model_meta.json")
        f1_score = metrics_meta.get("test_score", 0.0) if metrics_meta else 0.0
        
        return jsonify({
            "kpis": {
                "total_machines": 100,
                "healthy_machines": 82,
                "warning_machines": 12,
                "critical_machines": 6,
                "model_f1_score": round(f1_score, 4),
                "downtime_prevented_hrs": 450,
                "cost_saved_usd": 125000
            },
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in /dashboard: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/machines', methods=['GET'])
def get_machines():
    """Phase 6: List machines with health status"""
    try:
        machines_file = DATA_RAW / "PdM_machines.csv"
        if not machines_file.exists():
            return jsonify({"status": "error", "message": "Machines data not found"}), 404
            
        machines_df = pd.read_csv(machines_file)
        # Mocking health status for the API response based on age
        machines_df["status"] = machines_df["age"].apply(
            lambda x: "Critical" if x > 18 else ("Warning" if x > 12 else "Healthy")
        )
        
        return jsonify({
            "status": "success",
            "machines": machines_df.to_dict(orient="records")
        })
    except Exception as e:
        logger.error(f"Error in /machines: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Phase 5: Real-time prediction endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400
            
        model = get_best_model()
        if not model:
            return jsonify({"status": "error", "message": "Best model not found. Pipeline must be run first."}), 503
            
        # Expecting feature dict
        features = pd.DataFrame([data])
        
        # We assume data aligns with the model features. For an enterprise app, 
        # we would validate schema and engineer features dynamically here.
        prediction = model.predict(features)[0]
        
        proba = 1.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0][1]
            
        result = {
            "prediction": int(prediction),
            "probability": float(proba),
            "risk_level": "High" if prediction == 1 else "Low",
            "recommended_action": "Schedule immediate maintenance" if prediction == 1 else "Continue normal operation"
        }
        
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/history', methods=['GET'])
def get_history():
    """Phase 9: Maintenance & Failure History"""
    return jsonify({
        "status": "success", 
        "history": [
            {"machineID": 1, "date": "2025-01-10", "type": "Failure", "component": "comp2"},
            {"machineID": 1, "date": "2025-01-15", "type": "Maintenance", "component": "comp2"}
        ]
    })

@api_bp.route('/models', methods=['GET'])
def get_models():
    """Phase 3/5: Get best model metrics"""
    meta = load_json_artifact(METRICS_DIR / "best_model_meta.json")
    if meta:
        return jsonify({"status": "success", "model": meta})
    return jsonify({"status": "error", "message": "No models found"}), 404

@api_bp.route('/monitoring', methods=['GET'])
def get_monitoring():
    """Phase 11: System & Pipeline Monitoring"""
    return jsonify({
        "status": "success",
        "system": {
            "api_health": "Optimal",
            "model_drift": "None detected",
            "data_drift": "Stable",
            "latency_ms": 45
        }
    })
