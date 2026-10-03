"""
verify_user_flow.py
Verifies the complete Phase 6 end-to-end user diagnostic flow:
Select Machine -> Load Sensor Data -> Execute Diagnostics ->
XGBoost Prediction -> SHAP Explanation -> Decision Engine ->
Risk Tier -> Health Score -> Priority -> Maintenance Recommendation.
"""

import requests

BASE = "http://localhost:5000/api"

for mid in [1, 35, 94]:
    print("==================================================")
    print(f"STEP 1: Select Machine -> Machine ID: {mid}")
    
    # 1. Fetch Features (Load Sensor Data)
    f_res = requests.get(f"{BASE}/machines/{mid}/features").json()
    feats = f_res.get("data", {}).get("features", {})
    print("STEP 2: Load Sensor Data ->")
    for k in ["model", "age", "volt", "rotate", "pressure", "vibration"]:
        print(f"    {k}: {feats.get(k)}")
    
    # 2. Execute Diagnostics (XGBoost Prediction)
    payload = {
        "machineID": mid,
        "volt": feats.get("volt"),
        "rotate": feats.get("rotate"),
        "pressure": feats.get("pressure"),
        "vibration": feats.get("vibration"),
        "model": feats.get("model"),
        "age": feats.get("age")
    }
    p_res = requests.post(f"{BASE}/predict", json=payload).json()
    p_data = p_res.get("data", {})
    print("STEP 3 & 4: XGBoost Prediction ->")
    print(f"    Failure Probability: {p_data.get('probability')}")
    print(f"    Predicted Class: {p_data.get('prediction')}")
    print(f"    Confidence: {p_data.get('confidence')}")
    print(f"    Risk Level: {p_data.get('risk_level')}")
    
    # 3. SHAP Explanation
    e_res = requests.post(f"{BASE}/explain", json=payload).json()
    top_feats = e_res.get("data", {}).get("contributions", [])[:3]
    print("STEP 5: SHAP Explanation (Top 3 Drivers) ->")
    for d in top_feats:
        feat_name = d.get("feature")
        impact = round(d.get("abs_value", 0), 4)
        shap_val = d.get("shap_value")
        print(f"    Feature: {feat_name} | Impact: {impact} | SHAP Value: {shap_val}")
        
    # 4. Decision Engine -> Risk Tier -> Health Score -> Priority -> Recommendation
    d_res = requests.get(f"{BASE}/machines/{mid}/decision").json()
    d_data = d_res.get("data", {})
    print("STEP 6-10: Decision Engine Evaluation ->")
    print(f"    Risk Tier: {d_data.get('risk_tier')}")
    print(f"    Health Score: {d_data.get('health_score')}")
    print(f"    Priority: {d_data.get('priority')}")
    print(f"    Urgency: {d_data.get('urgency')}")
    rec = d_data.get("maintenance_action") or d_data.get("recommendation")
    print(f"    Maintenance Action: {rec}")
    print()
