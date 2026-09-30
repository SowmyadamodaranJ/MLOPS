import pytest
import json

@pytest.mark.security
def test_large_payload_rejection(test_client):
    """Test that the API rejects exceedingly large payloads."""
    large_payload = {
        "machineID": "1",
        "datetime": "2024-01-01 10:00:00",
        "volt": 170.5,
        "rotate": 450.2,
        "pressure": 100.1,
        "vibration": 40.5,
        # Create a massive string to bloat the payload
        "malicious_padding": "A" * 1024 * 1024 * 5 # 5MB string
    }
    
    response = test_client.post(
        "/api/predict", 
        json=large_payload,
        # Simulate content length checking if implemented
        # Flask usually handles this via MAX_CONTENT_LENGTH, 
        # but we're testing the application layer handling.
    )
    
    # Ideally, should be rejected (413 Payload Too Large or 400 Bad Request)
    assert response.status_code in [413, 400, 503]

@pytest.mark.security
def test_malformed_json_rejection(test_client):
    """Test that the API rejects invalid JSON safely without crashing."""
    malformed_json = "{ \"machineID\": 1, \"volt\": 170.5 " # Missing closing brace
    
    response = test_client.post(
        "/api/predict",
        data=malformed_json,
        content_type="application/json"
    )
    
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "BAD_REQUEST" in data["error_code"]
