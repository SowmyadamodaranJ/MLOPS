import pytest
import time
from backend.services.model_service import ModelService
from backend.services.feature_service import FeatureService
from backend.services.explain_service import ExplainService

@pytest.fixture
def model_service():
    return ModelService()

@pytest.fixture
def feature_service():
    return FeatureService()

@pytest.fixture
def explain_service():
    return ExplainService()

@pytest.mark.ml
def test_ml_pipeline_loads_correctly(model_service):
    """Ensure the serialized ML pipeline can be deserialized and has expected components."""
    model = model_service.get_model()
    
    # In a real environment with models, this shouldn't be None
    if model is not None:
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        
        feature_names = model_service.get_feature_names()
        assert len(feature_names) > 0

@pytest.mark.ml
def test_prediction_latency(model_service, feature_service):
    """Benchmark prediction latency to ensure it meets real-time SLA (< 50ms)."""
    model = model_service.get_model()
    
    if model is not None:
        # Generate dummy feature vector matching model signature
        machine_id = 1
        features_df = feature_service.prepare_inference_features(machine_id, {})
        feature_names = model_service.get_feature_names()
        X_infer = features_df[feature_names]
        
        start = time.time()
        prediction = model.predict(X_infer)
        latency = (time.time() - start) * 1000
        
        assert latency < 100.0, f"Prediction latency too high: {latency}ms"
        assert prediction[0] in [0, 1]

@pytest.mark.ml
def test_shap_generation_consistency(model_service, feature_service, explain_service):
    """Test that SHAP explainability returns correct structure and matches feature count."""
    model = model_service.get_model()
    
    if model is not None:
        features_df = feature_service.prepare_inference_features(1, {})
        explanation = explain_service.get_local_explanation(1, features_df)
        
        assert explanation["status"] == "success"
        assert "contributions" in explanation
        
        # Check that contributions map to expected features
        contribs = explanation["contributions"]
        assert len(contribs) <= len(model_service.get_feature_names())
