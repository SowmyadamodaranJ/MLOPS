import pytest
from backend.services.model_service import ModelService

@pytest.fixture
def model_service():
    """Returns a fresh instance of the ModelService."""
    # Reset singleton instance for isolated testing
    ModelService._instance = None
    return ModelService()

@pytest.mark.unit
def test_model_service_initialization(model_service):
    """Test that the ModelService initializes and checks artifacts."""
    artifacts_status = model_service.get_artifacts_status()
    
    assert isinstance(artifacts_status, dict)
    assert "best_model.joblib" in artifacts_status
    
    # We don't strictly assert model_ready=True here because CI might not have models trained,
    # but we assert that the structure is correct.
    assert isinstance(model_service.model_ready, bool)

@pytest.mark.unit
def test_get_model_info(model_service):
    """Test retrieval of model metadata."""
    info = model_service.get_model_info()
    
    assert "active_model" in info
    assert "version" in info
    assert "status" in info
    assert "feature_count" in info
    
@pytest.mark.unit
def test_get_optimal_threshold(model_service):
    """Test threshold retrieval defaults or returns a float."""
    threshold = model_service.get_optimal_threshold()
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0
