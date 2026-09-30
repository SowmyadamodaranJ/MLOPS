import pytest

@pytest.mark.security
@pytest.mark.parametrize("malicious_string", [
    "' OR 1=1 --",
    "1; DROP TABLE predictions;",
    "\" OR \"\"=\"",
    "admin' --",
    "<script>alert('xss')</script>",
    "javascript:alert(1)",
    "<img src=x onerror=alert(1)>"
])
def test_sql_and_xss_injection_attempts(test_client, malicious_string):
    """Test that endpoints reject or sanitize common SQLi and XSS patterns."""
    
    # Attempt injection via URL parameter
    response_get = test_client.get(f"/api/monitoring/logs?search={malicious_string}")
    
    # The application should either return 200 with 0 results (sanitized),
    # or 400 Bad Request. It should NEVER return a 500 error.
    assert response_get.status_code in [200, 400]
    
    # Attempt injection via JSON payload (numeric field)
    invalid_payload = {
        "machineID": "1",
        "datetime": "2024-01-01 10:00:00",
        "volt": malicious_string,
        "rotate": 450.2,
        "pressure": 100.1,
        "vibration": 40.5
    }
    
    response_post = test_client.post("/api/predict", json=invalid_payload)
    
    # Must be rejected because 'volt' expects a float
    assert response_post.status_code == 400
    data = response_post.get_json()
    assert data["success"] is False
