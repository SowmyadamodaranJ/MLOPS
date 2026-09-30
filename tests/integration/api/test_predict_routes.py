import pytest

@pytest.mark.integration
def test_predict_endpoint_valid(test_client, mock_db, valid_prediction_payload):
    """Test the /predict endpoint with a valid payload."""
    response = test_client.post("/api/predict", json=valid_prediction_payload)
    
    # Depending on model presence, might return 200 or 503
    assert response.status_code in [200, 503]
    
    data = response.get_json()
    if response.status_code == 200:
        assert data["success"] is True
        assert "prediction" in data["data"]
        assert "risk_level" in data["data"]
    else:
        assert data["success"] is False
        assert data["error_code"] == "MODEL_NOT_LOADED"

@pytest.mark.integration
def test_predict_endpoint_invalid(test_client):
    """Test the /predict endpoint rejects invalid payloads."""
    invalid_payload = {"volt": 170.5}  # Missing machineID
    
    response = test_client.post("/api/predict", json=invalid_payload)
    
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert data["error_code"] == "MISSING_FIELD"

@pytest.mark.integration
def test_explain_endpoint_valid(test_client, valid_prediction_payload):
    """Test the /explain endpoint with a valid payload."""
    response = test_client.post("/api/explain", json=valid_prediction_payload)
    
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.get_json()
        assert data["success"] is True
        assert "contributions" in data["data"]
