import pytest
import pandas as pd
from backend.services.feature_service import FeatureService

@pytest.fixture
def feature_service():
    """Returns an instance of FeatureService."""
    return FeatureService()

@pytest.mark.unit
def test_feature_service_initialization(feature_service):
    """Test that the feature service initializes and loads machine data."""
    assert feature_service.machines_df is not None
    assert not feature_service.machines_df.empty
    
    machines = feature_service.get_machines_list()
    assert isinstance(machines, list)
    assert len(machines) > 0
    assert "machineID" in machines[0]
    assert "status" in machines[0]

@pytest.mark.unit
def test_prepare_inference_features_success(feature_service):
    """Test feature preparation for inference with valid inputs."""
    machine_id = 1
    overrides = {
        "volt": 180.0,
        "rotate": 400.0,
        "pressure": 110.0,
        "vibration": 50.0
    }
    
    # Generate features
    features_df = feature_service.prepare_inference_features(machine_id, overrides)
    
    assert features_df is not None
    assert isinstance(features_df, pd.DataFrame)
    
    # Check that overrides were applied
    assert features_df.iloc[0]["volt"] == 180.0
    assert features_df.iloc[0]["rotate"] == 400.0
    
    # Check that engineered features exist (e.g., rolling means, if applicable based on the service's logic)
    # The actual columns depend on the pipeline, but we can verify it's a non-empty DF
    assert len(features_df.columns) > 5

@pytest.mark.unit
def test_prepare_inference_features_invalid_machine(feature_service):
    """Test feature preparation fails gracefully for unknown machines."""
    features_df = feature_service.prepare_inference_features(99999, {})
    assert features_df is None
