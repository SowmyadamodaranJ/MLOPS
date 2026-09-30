"""
test_decision_engine.py
-----------------------
Unit and integration tests for the Maintenance Decision Engine (Phase 3).

Covers all 10 core requirements:
1.  probability < 0.25 -> MONITOR
2.  probability = 0.25 -> WARNING
3.  probability = 0.50 -> CRITICAL
4.  health score calculation
5.  decision object structure
6.  SHAP driver extraction
7.  real machine decision endpoint
8.  priority queue ordering
9.  model version propagation
10. invalid machine ID handling
"""

import pytest
from backend.services.decision_engine_service import DecisionEngineService
from backend.services.model_service import ModelService


@pytest.fixture
def decision_service():
    """Fixture providing a fresh or singleton DecisionEngineService instance."""
    return DecisionEngineService()


# ==============================================================================
# 1. probability < 0.25 -> MONITOR
# ==============================================================================
@pytest.mark.parametrize("prob", [0.0, 0.05, 0.12, 0.20, 0.2499])
def test_risk_tier_monitor_below_25(decision_service, prob):
    """Verify that failure probability strictly below 0.25 maps to MONITOR."""
    tier = decision_service.determine_risk_tier(prob)
    assert tier == "MONITOR", f"Probability {prob} should map to MONITOR, got {tier}"
    assert decision_service.determine_urgency(tier) == "Routine"
    assert "Low" in decision_service.determine_priority(tier)


# ==============================================================================
# 2. probability = 0.25 -> WARNING
# ==============================================================================
@pytest.mark.parametrize("prob", [0.25, 0.30, 0.40, 0.4999])
def test_risk_tier_warning_at_25_and_above(decision_service, prob):
    """Verify that failure probability in [0.25, 0.50) maps to WARNING."""
    tier = decision_service.determine_risk_tier(prob)
    assert tier == "WARNING", f"Probability {prob} should map to WARNING, got {tier}"
    assert decision_service.determine_urgency(tier) == "Scheduled"
    assert "High" in decision_service.determine_priority(tier)


# ==============================================================================
# 3. probability = 0.50 -> CRITICAL
# ==============================================================================
@pytest.mark.parametrize("prob", [0.50, 0.51, 0.75, 0.999, 1.0])
def test_risk_tier_critical_at_50_and_above(decision_service, prob):
    """Verify that failure probability >= 0.50 maps to CRITICAL."""
    tier = decision_service.determine_risk_tier(prob)
    assert tier == "CRITICAL", f"Probability {prob} should map to CRITICAL, got {tier}"
    assert decision_service.determine_urgency(tier) == "Immediate"
    assert "Critical" in decision_service.determine_priority(tier)


# ==============================================================================
# 4. health score calculation
# ==============================================================================
def test_health_score_calculation(decision_service):
    """
    Verify transparent formula: health_score = 100 * (1 - failure_probability).
    Clamped to 0.0 <= health_score <= 100.0.
    """
    assert decision_service.calculate_health_score(0.0) == 100.0
    assert decision_service.calculate_health_score(0.25) == 75.0
    assert decision_service.calculate_health_score(0.50) == 50.0
    assert decision_service.calculate_health_score(0.80) == 20.0
    assert decision_service.calculate_health_score(1.0) == 0.0

    # Clamping tests
    assert decision_service.calculate_health_score(-0.1) == 100.0
    assert decision_service.calculate_health_score(1.2) == 0.0


# ==============================================================================
# 5. decision object structure
# ==============================================================================
def test_decision_object_structure(decision_service):
    """Verify complete decision object schema matches requirements."""
    contributions = [
        {"feature": "volt_rolling24h_mean", "shap_value": 0.45},
        {"feature": "rotate_rolling3h_max", "shap_value": -0.22},
    ]
    telemetry = {"volt": 170.0, "rotate": 450.0}

    decision = decision_service.build_decision(
        machine_id=7,
        failure_probability=0.35,
        prediction=0,
        contributions=contributions,
        telemetry=telemetry,
        model_version="2.0.0",
    )

    required_keys = [
        "machine_id",
        "failure_probability",
        "health_score",
        "risk_tier",
        "maintenance_action",
        "urgency",
        "priority",
        "primary_driver",
        "secondary_drivers",
        "reason",
        "model_version",
    ]
    for key in required_keys:
        assert key in decision, f"Missing required key '{key}' in decision object"

    assert decision["machine_id"] == 7
    assert decision["failure_probability"] == 0.35
    assert decision["health_score"] == 65.0
    assert decision["risk_tier"] == "WARNING"
    assert decision["urgency"] == "Scheduled"
    assert "P2" in decision["priority"]
    assert "contributes" in decision["reason"]
    assert "causes" not in decision["reason"]
    assert "bearing" not in decision["maintenance_action"].lower()


# ==============================================================================
# 6. SHAP driver extraction
# ==============================================================================
def test_shap_driver_extraction(decision_service):
    """Verify SHAP driver extraction: feature, contribution, direction."""
    contributions = [
        {"feature": "vibration_rolling24h_mean", "shap_value": 0.42},
        {"feature": "pressure_rolling3h_mean", "shap_value": -0.18},
        {"feature": "age", "shap_value": 0.0},
    ]
    primary, secondary = decision_service.extract_shap_drivers(contributions)

    assert primary["feature"] == "vibration_rolling24h_mean"
    assert primary["contribution"] == 0.42
    assert primary["direction"] == "increases_risk"

    assert len(secondary) == 2
    assert secondary[0]["feature"] == "pressure_rolling3h_mean"
    assert secondary[0]["contribution"] == -0.18
    assert secondary[0]["direction"] == "decreases_risk"

    assert secondary[1]["direction"] == "neutral"


# ==============================================================================
# 7. real machine decision endpoint
# ==============================================================================
def test_real_machine_decision_endpoint(test_client):
    """Test GET /api/machines/{machine_id}/decision and alias with real machine data."""
    # Machine 1 is a known real machine
    response = test_client.get("/api/machines/1/decision")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    data = payload["data"]

    assert data["machine_id"] == 1
    assert "failure_probability" in data
    assert "health_score" in data
    assert data["risk_tier"] in ["CRITICAL", "WARNING", "MONITOR"]
    assert "maintenance_action" in data
    assert "primary_driver" in data
    assert "model_version" in data

    # Test alias route
    alias_resp = test_client.get("/api/decision/1")
    assert alias_resp.status_code == 200
    assert alias_resp.get_json()["data"]["machine_id"] == 1

    # Machine 16 is a real high-risk machine
    resp16 = test_client.get("/api/machines/16/decision")
    assert resp16.status_code == 200
    d16 = resp16.get_json()["data"]
    assert d16["risk_tier"] == "CRITICAL"
    assert d16["urgency"] == "Immediate"


# ==============================================================================
# 8. priority queue ordering
# ==============================================================================
def test_priority_queue_ordering(test_client):
    """Test GET /api/machines/health-queue ordering and structure."""
    response = test_client.get("/api/machines/health-queue")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["success"] is True
    data = payload["data"]

    queue = data["queue"]
    assert len(queue) == 100, f"Expected 100 machines in queue, got {len(queue)}"

    # Check required fields on each machine item
    for item in queue[:10]:
        assert "machine_id" in item
        assert "failure_probability" in item
        assert "health_score" in item
        assert "risk_tier" in item
        assert "priority" in item
        assert "recommended_action" in item
        assert "primary_driver" in item

    # Verify priority ordering: CRITICAL before WARNING before MONITOR
    tier_rank = {"CRITICAL": 0, "WARNING": 1, "MONITOR": 2}
    ranks = [tier_rank[m["risk_tier"]] for m in queue]
    assert ranks == sorted(ranks), "Queue is not sorted by risk tier rank!"

    # Verify that within the CRITICAL tier, machines are sorted by probability descending
    critical_probs = [m["failure_probability"] for m in queue if m["risk_tier"] == "CRITICAL"]
    assert critical_probs == sorted(critical_probs, reverse=True), (
        "Critical tier machines are not sorted by failure probability descending"
    )

    # Test alias
    alias_resp = test_client.get("/api/health-queue")
    assert alias_resp.status_code == 200


# ==============================================================================
# 9. model version propagation
# ==============================================================================
def test_model_version_propagation(test_client):
    """Verify model version propagates dynamically from ModelService metadata."""
    ms = ModelService()
    active_version = ms.version or "2.0.0"

    response = test_client.get("/api/machines/1/decision")
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["model_version"] == active_version


# ==============================================================================
# 10. invalid machine ID handling
# ==============================================================================
def test_invalid_machine_id_handling(test_client):
    """Verify appropriate error responses for non-existent and non-positive machine IDs."""
    # Non-existent machine (cache has 100 machines)
    r_not_found = test_client.get("/api/machines/999/decision")
    assert r_not_found.status_code == 404
    data_nf = r_not_found.get_json()
    assert data_nf["success"] is False
    assert data_nf["error_code"] == "MACHINE_NOT_FOUND"

    # Non-positive machine ID
    r_zero = test_client.get("/api/machines/0/decision")
    assert r_zero.status_code == 400
    data_zero = r_zero.get_json()
    assert data_zero["success"] is False
    assert data_zero["error_code"] == "INVALID_MACHINE_ID"

    # Negative machine ID
    r_neg = test_client.get("/api/machines/-1/decision")
    assert r_neg.status_code in [400, 404]
