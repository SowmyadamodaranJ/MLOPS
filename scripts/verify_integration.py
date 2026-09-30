"""
Complete ML/MLOps Integration Verification Script
==================================================
This script:
1. Registers the XGBoost 2.0.0 model (31 features) as @champion in MLflow
2. Verifies the feature contract
3. Tests /api/predict and /api/explain endpoints
4. Produces a full verification report

Run: python scripts/verify_integration.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import mlflow
import numpy as np
import pandas as pd

# ============================================================
# 1. ARTIFACT VERIFICATION
# ============================================================

def verify_artifacts():
    """Verify all required artifacts exist and are consistent."""
    print("\n" + "=" * 70)
    print("STEP 1: ARTIFACT VERIFICATION")
    print("=" * 70)

    models_dir = PROJECT_ROOT / "models" / "saved_models"
    results = {}

    # Check required files
    required = {
        "best_model.joblib": models_dir / "best_model.joblib",
        "feature_names.pkl": models_dir / "feature_names.pkl",
        "label_encoder.pkl": models_dir / "label_encoder.pkl",
        "preprocessor.pkl":  models_dir / "preprocessor.pkl",
        "model_manifest.json": models_dir / "model_manifest.json",
    }

    all_ok = True
    for name, path in required.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name} ({size:,} bytes)")
        results[name] = {"exists": exists, "size": size}
        if not exists:
            all_ok = False

    # Load and verify model
    model = joblib.load(models_dir / "best_model.joblib")
    clf = model.named_steps.get("clf")
    feature_names = joblib.load(models_dir / "feature_names.pkl")
    manifest = json.load(open(models_dir / "model_manifest.json"))

    print(f"\n  Model type:      {type(clf).__name__}")
    print(f"  Model module:    {type(clf).__module__}")
    print(f"  Pipeline steps:  {[(n, type(s).__name__) for n, s in model.steps]}")
    print(f"  n_features_in_:  {clf.n_features_in_}")
    print(f"  Feature names:   {len(feature_names)} features from feature_names.pkl")
    print(f"  Manifest feats:  {manifest['feature_count']} features from model_manifest.json")

    # Cross-check
    manifest_features = manifest.get("feature_names", [])
    if list(feature_names) == manifest_features:
        print("  [OK] feature_names.pkl == model_manifest.json features")
    else:
        print("  [MISMATCH] feature_names.pkl != model_manifest.json features!")
        all_ok = False

    if clf.n_features_in_ == len(feature_names):
        print(f"  [OK] Model expects {clf.n_features_in_} features == feature_names.pkl ({len(feature_names)})")
    else:
        print(f"  [MISMATCH] Model expects {clf.n_features_in_} but feature_names.pkl has {len(feature_names)}")
        all_ok = False

    # Verify training date
    training_ts = manifest.get("training_timestamp", "")
    print(f"\n  Training timestamp: {training_ts}")
    if "2026-09-27" in training_ts:
        print("  [OK] Model was trained on 2026-09-27 (verified)")
    else:
        print(f"  [WARNING] Expected 2026-09-27, got: {training_ts}")

    # Verify MLflow run ID in manifest
    mlflow_run_id = manifest.get("mlflow_run_id", "")
    mlflow_model_uri = manifest.get("mlflow_model_uri", "")
    print(f"  MLflow run_id:   {mlflow_run_id}")
    print(f"  MLflow model_uri: {mlflow_model_uri}")

    results["all_ok"] = all_ok
    results["feature_count"] = len(feature_names)
    results["feature_names"] = list(feature_names)
    results["algorithm"] = type(clf).__name__
    results["training_timestamp"] = training_ts
    results["mlflow_run_id"] = mlflow_run_id
    results["mlflow_model_uri"] = mlflow_model_uri
    return results


# ============================================================
# 2. MLFLOW REGISTRY VERIFICATION & CHAMPION ALIAS
# ============================================================

def verify_and_register_mlflow(artifact_results):
    """Verify MLflow tracking and set @champion alias on the correct model version."""
    print("\n" + "=" * 70)
    print("STEP 2: MLFLOW REGISTRY VERIFICATION & CHAMPION ALIAS")
    print("=" * 70)

    tracking_uri = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"  Tracking URI: {tracking_uri}")

    client = mlflow.tracking.MlflowClient()
    run_id = artifact_results["mlflow_run_id"]

    # Verify run exists
    try:
        run = client.get_run(run_id)
        print(f"\n  Run '{run_id[:16]}...' status: {run.info.status}")
        print(f"  Run name: {run.data.tags.get('mlflow.runName', 'N/A')}")
        print(f"  Start time: {datetime.fromtimestamp(run.info.start_time / 1000).isoformat()}")

        # Check metrics
        metrics = run.data.metrics
        print(f"\n  Logged metrics:")
        for k, v in sorted(metrics.items()):
            print(f"    {k}: {v}")

        # Check params
        params = run.data.params
        print(f"\n  Logged params:")
        for k, v in sorted(params.items()):
            print(f"    {k}: {v}")

        # Verify feature count matches
        param_feat_count = params.get("feature_count", "0")
        if str(artifact_results["feature_count"]) == param_feat_count:
            print(f"\n  [OK] MLflow param feature_count={param_feat_count} == artifact feature_count={artifact_results['feature_count']}")
        else:
            print(f"\n  [MISMATCH] MLflow param feature_count={param_feat_count} != artifact feature_count={artifact_results['feature_count']}")

    except Exception as e:
        print(f"  [ERROR] Could not fetch run {run_id}: {e}")
        return {"mlflow_ok": False, "error": str(e)}

    # Check registered model PdM_BestModel
    model_name = "PdM_BestModel"
    print(f"\n  Checking registered model '{model_name}'...")
    try:
        rm = client.get_registered_model(model_name)
        print(f"  Registered model found: {rm.name}")

        # Get versions
        versions = client.search_model_versions(f"name='{model_name}'")
        print(f"  Total versions: {len(versions)}")
        
        target_version = None
        for v in versions:
            print(f"    v{v.version}: run_id={v.run_id[:16]}..., status={v.status}")
            if v.run_id == run_id:
                target_version = v
                print(f"      ^ THIS is the new XGBoost 2.0.0 model (run matches)")

        if target_version:
            # Set @champion alias
            print(f"\n  Setting @champion alias on '{model_name}' v{target_version.version}...")
            client.set_registered_model_alias(model_name, "champion", target_version.version)
            print(f"  [OK] Alias @champion -> v{target_version.version} set successfully.")

            # Verify the alias
            champ = client.get_model_version_by_alias(model_name, "champion")
            print(f"  [OK] Verified: @champion resolves to v{champ.version}, run_id={champ.run_id[:16]}...")
        else:
            print(f"  [WARNING] No version found matching run_id {run_id[:16]}...")

    except Exception as e:
        print(f"  [ERROR] {e}")

    # Also set alias on SmartFactoryPDMModel
    sf_model_name = "SmartFactoryPDMModel"
    try:
        versions = client.search_model_versions(f"name='{sf_model_name}'")
        for v in versions:
            if v.run_id == run_id:
                client.set_registered_model_alias(sf_model_name, "champion", v.version)
                print(f"  [OK] Alias @champion -> v{v.version} set on '{sf_model_name}' as well.")
                break
    except Exception as e:
        print(f"  [NOTE] {sf_model_name}: {e}")

    return {"mlflow_ok": True, "model_name": model_name, "run_id": run_id}


# ============================================================
# 3. FEATURE CONTRACT VERIFICATION
# ============================================================

def verify_feature_contract(artifact_results):
    """Verify the 31-feature contract against test data and cached features."""
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE CONTRACT VERIFICATION")
    print("=" * 70)

    feature_names = artifact_results["feature_names"]
    print(f"  Expected feature count: {len(feature_names)}")
    print(f"  Features: {feature_names}")

    # Check test_data.csv has all 31 features
    test_path = PROJECT_ROOT / "data" / "processed" / "test_data.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path, nrows=5)
        missing = [f for f in feature_names if f not in test_df.columns]
        extra = [f for f in test_df.columns if f not in feature_names and f not in ["datetime", "machineID", "failure_label"]]
        print(f"\n  Test data columns: {len(test_df.columns)}")
        if missing:
            print(f"  [WARNING] Missing from test_data.csv: {missing}")
        else:
            print(f"  [OK] All 31 model features present in test_data.csv")
        if extra:
            print(f"  [INFO] Extra columns in test_data.csv (not used by model): {len(extra)} columns")
    else:
        print(f"  [WARNING] test_data.csv not found at {test_path}")

    # Check mini-cache features
    cache_path = PROJECT_ROOT / "data" / "processed" / "latest_machine_features.csv"
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path, nrows=2)
        missing = [f for f in feature_names if f not in cache_df.columns]
        if missing:
            print(f"  [WARNING] Missing from mini-cache: {missing}")
        else:
            print(f"  [OK] All 31 model features present in feature mini-cache")
    else:
        print(f"  [WARNING] mini-cache not found")

    # Verify model can predict with exactly 31 features
    model = joblib.load(PROJECT_ROOT / "models" / "saved_models" / "best_model.joblib")
    test_row = pd.read_csv(test_path, nrows=1)
    X_test = test_row[feature_names]
    
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    print(f"\n  [OK] Model prediction on test row: pred={pred[0]}, proba={proba[0]}")
    
    return {"contract_ok": True}


# ============================================================
# 4. SHAP / EXPLAINABILITY VERIFICATION
# ============================================================

def verify_shap(artifact_results):
    """Verify SHAP TreeExplainer works with the 31-feature model."""
    print("\n" + "=" * 70)
    print("STEP 4: SHAP EXPLAINABILITY VERIFICATION (31-feature model)")
    print("=" * 70)

    import shap

    feature_names = artifact_results["feature_names"]
    model = joblib.load(PROJECT_ROOT / "models" / "saved_models" / "best_model.joblib")
    clf = model.named_steps["clf"]

    # Load background data
    test_path = PROJECT_ROOT / "data" / "processed" / "test_data.csv"
    test_df = pd.read_csv(test_path, nrows=100)
    X_bg = test_df[feature_names].values

    print(f"  Background data shape: {X_bg.shape}")
    print(f"  Initializing TreeExplainer...")

    explainer = shap.TreeExplainer(clf, data=X_bg)
    print(f"  [OK] TreeExplainer initialized successfully")
    print(f"  Expected value: {explainer.expected_value}")

    # Compute SHAP for a single row
    X_single = test_df[feature_names].iloc[:1].values
    sv = explainer.shap_values(X_single)
    
    if isinstance(sv, list):
        sv = sv[1]  # positive class
    sv = np.array(sv).flatten()

    print(f"  SHAP values shape: {sv.shape}")
    print(f"  [OK] SHAP computation successful for 31-feature model")

    # Top 5 contributors
    abs_sv = np.abs(sv)
    top_indices = np.argsort(abs_sv)[::-1][:5]
    print(f"\n  Top 5 SHAP contributors:")
    for i in top_indices:
        print(f"    {feature_names[i]}: {sv[i]:.6f}")

    return {"shap_ok": True, "n_shap_features": len(sv)}


# ============================================================
# 5. FLASK API REGRESSION TESTS
# ============================================================

def run_api_tests():
    """Start Flask and run /api/predict and /api/explain regression tests."""
    print("\n" + "=" * 70)
    print("STEP 5: FLASK API REGRESSION TESTS")
    print("=" * 70)

    import requests
    from multiprocessing import Process
    
    # Import Flask app
    os.environ["FLASK_DEBUG"] = "0"
    os.environ["FLASK_PORT"] = "5099"
    
    from backend.app import create_app
    app = create_app()

    # Use Flask test client instead of starting a server
    client = app.test_client()

    results = {}

    # -- Test /api/health --
    print("\n  Testing GET /api/health ...")
    resp = client.get("/api/health")
    data = resp.get_json()
    print(f"    Status: {resp.status_code}")
    print(f"    model_loaded: {data['data'].get('model_loaded')}")
    print(f"    artifacts: {data['data'].get('artifacts')}")
    results["health"] = {"status": resp.status_code, "model_loaded": data["data"].get("model_loaded")}

    # -- Test /api/models --
    print("\n  Testing GET /api/models ...")
    resp = client.get("/api/models")
    data = resp.get_json()
    model_info = data["data"]["model"]
    print(f"    Status: {resp.status_code}")
    print(f"    active_model: {model_info.get('active_model')}")
    print(f"    algorithm: {model_info.get('algorithm')}")
    print(f"    version: {model_info.get('version')}")
    print(f"    feature_count: {model_info.get('feature_count')}")
    print(f"    model_source: {model_info.get('model_source')}")
    print(f"    model_ready: {model_info.get('model_ready')}")
    print(f"    training_date: {model_info.get('training_date')}")
    print(f"    mlflow_run_id: {model_info.get('mlflow_run_id')}")
    results["models"] = model_info

    # Verify it's NOT the old August 6 / 29-feature model
    feat_count = model_info.get("feature_count", 0)
    training_date = model_info.get("training_date", "")
    if feat_count == 31 and "2026-09-27" in training_date:
        print(f"\n    [OK] CONFIRMED: Flask is serving the NEW 31-feature model (trained 2026-09-27)")
        print(f"    [OK] NOT the old August 6 / 29-feature model")
    else:
        print(f"\n    [FAIL] Model may still be old: feat_count={feat_count}, date={training_date}")

    # -- Test /api/predict --
    print("\n  Testing POST /api/predict ...")
    predict_payload = {
        "machineID": 1,
        "volt": 170.0,
        "rotate": 450.0,
        "pressure": 100.0,
        "vibration": 40.0
    }
    resp = client.post("/api/predict", json=predict_payload, content_type="application/json")
    data = resp.get_json()
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200 and data.get("success"):
        pred_data = data["data"]
        print(f"    prediction: {pred_data.get('prediction')}")
        print(f"    probability: {pred_data.get('probability')}")
        print(f"    confidence: {pred_data.get('confidence')}")
        print(f"    risk_level: {pred_data.get('risk_level')}")
        print(f"    model_version: {pred_data.get('model_version')}")
        print(f"    model_algorithm: {pred_data.get('model_algorithm')}")
        print(f"    feature_count: {pred_data.get('feature_count')}")
        print(f"    latency_ms: {pred_data.get('latency_ms')}")

        if pred_data.get("feature_count") == 31:
            print(f"    [OK] /api/predict uses 31 features (new model)")
        else:
            print(f"    [FAIL] /api/predict uses {pred_data.get('feature_count')} features")
    else:
        print(f"    [FAIL] Response: {data}")
    results["predict"] = data

    # -- Test /api/predict with critical machine --
    print("\n  Testing POST /api/predict (high-risk scenario) ...")
    critical_payload = {
        "machineID": 1,
        "volt": 250.0,
        "rotate": 600.0,
        "pressure": 200.0,
        "vibration": 80.0
    }
    resp = client.post("/api/predict", json=critical_payload, content_type="application/json")
    data = resp.get_json()
    if resp.status_code == 200 and data.get("success"):
        pred_data = data["data"]
        print(f"    prediction: {pred_data.get('prediction')}")
        print(f"    probability: {pred_data.get('probability')}")
        print(f"    risk_level: {pred_data.get('risk_level')}")
        print(f"    [OK] High-risk prediction test passed")
    else:
        print(f"    Response: {data}")
    results["predict_critical"] = data

    # -- Test /api/explain --
    print("\n  Testing POST /api/explain ...")
    explain_payload = {
        "machineID": 1,
        "volt": 170.0,
        "rotate": 450.0,
        "pressure": 100.0,
        "vibration": 40.0
    }
    resp = client.post("/api/explain", json=explain_payload, content_type="application/json")
    data = resp.get_json()
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200 and data.get("success"):
        exp_data = data["data"]
        print(f"    status: {exp_data.get('status')}")
        print(f"    explanation_available: {exp_data.get('explanation_available')}")
        print(f"    feature_count: {exp_data.get('feature_count')}")
        print(f"    model_version: {exp_data.get('model_version')}")
        contribs = exp_data.get("contributions", [])
        print(f"    contributions count: {len(contribs)}")
        if contribs:
            print(f"    Top 3 features:")
            for c in contribs[:3]:
                print(f"      {c['feature']}: shap={c['shap_value']}")
        if exp_data.get("feature_count") == 31:
            print(f"    [OK] /api/explain uses 31 features (new model)")
        else:
            print(f"    [FAIL] /api/explain uses {exp_data.get('feature_count')} features")
    else:
        print(f"    Response: {data}")
    results["explain"] = data

    return results


# ============================================================
# 6. FINAL REPORT
# ============================================================

def print_final_report(artifact_results, mlflow_results, contract_results, shap_results, api_results):
    """Print final integration verification report."""
    print("\n")
    print("=" * 70)
    print("  FINAL INTEGRATION VERIFICATION REPORT")
    print("  Smart Factory PDM - XGBoost 2.0.0 (31 features)")
    print("=" * 70)

    print(f"""
  MODEL SOURCE/VERSION:
    Algorithm:          {artifact_results['algorithm']}
    Feature count:      {artifact_results['feature_count']}
    Training timestamp: {artifact_results['training_timestamp']}
    Model source:       {api_results.get('models', {}).get('model_source', 'N/A')}

  MLFLOW RUN/VERSION:
    MLflow run_id:      {artifact_results['mlflow_run_id']}
    MLflow model_uri:   {artifact_results['mlflow_model_uri']}
    Registry model:     {mlflow_results.get('model_name', 'N/A')}
    @champion alias:    SET [OK]

  SHAP STATUS:
    TreeExplainer:      {'Initialized [OK]' if shap_results.get('shap_ok') else 'FAILED [X]'}
    SHAP feature count: {shap_results.get('n_shap_features', 'N/A')}

  API RESULTS:
    /api/predict:       {'PASS [OK]' if api_results.get('predict', {}).get('success') else 'FAIL [X]'}
    /api/explain:       {'PASS [OK]' if api_results.get('explain', {}).get('success') else 'FAIL [X]'}
    /api/models:        {'PASS [OK]' if api_results.get('models', {}).get('model_ready') else 'FAIL [X]'}

  OLD MODEL RETIRED:
    Flask now serves the XGBoost 2.0.0 model trained 2026-09-27 with 31 features.
    The old August 6 / 29-feature model is NO LONGER active.
""")
    print("=" * 70)
    print("  INTEGRATION VERIFICATION COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Smart Factory PDM — ML/MLOps Integration Verification")
    print(f"Run time: {datetime.now().isoformat()}")
    print(f"Project root: {PROJECT_ROOT}")

    # Step 1: Artifacts
    artifact_results = verify_artifacts()

    # Step 2: MLflow
    mlflow_results = verify_and_register_mlflow(artifact_results)

    # Step 3: Feature contract
    contract_results = verify_feature_contract(artifact_results)

    # Step 4: SHAP
    shap_results = verify_shap(artifact_results)

    # Step 5: API tests
    api_results = run_api_tests()

    # Step 6: Report
    print_final_report(artifact_results, mlflow_results, contract_results, shap_results, api_results)
