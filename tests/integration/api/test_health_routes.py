import pytest

@pytest.mark.integration
def test_health_check(test_client):
    """Test the /health endpoint returns expected schema."""
    response = test_client.get("/api/health")
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert data["success"] is True
    assert "status" in data["data"]
    assert "api_health" in data["data"]
    
@pytest.mark.integration
def test_get_dashboard(test_client):
    """Test the /dashboard endpoint returns KPIs."""
    response = test_client.get("/api/dashboard")
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert data["success"] is True
    assert "kpis" in data["data"]
    assert "total_machines" in data["data"]["kpis"]

@pytest.mark.integration
def test_get_metrics(test_client):
    """Test the /metrics endpoint handling."""
    response = test_client.get("/api/metrics")
    
    # Could be 200 or 404 depending on if metrics file is generated in CI
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        data = response.get_json()
        assert "metrics" in data["data"]
        assert "Accuracy" in data["data"]["metrics"]
