import pytest

@pytest.mark.performance
def test_predict_latency(benchmark, test_client, valid_prediction_payload):
    """
    Benchmark the /predict endpoint using pytest-benchmark.
    Ensures that under nominal conditions, prediction latency is acceptable.
    """
    
    def run_prediction():
        response = test_client.post("/api/predict", json=valid_prediction_payload)
        return response.status_code
        
    # Run the benchmark
    status_code = benchmark(run_prediction)
    
    # Assert successful or safely unavailable
    assert status_code in [200, 503]
    
    # Ensure average latency is under 200ms
    # (benchmark.stats.stats.mean is in seconds)
    assert benchmark.stats.stats.mean < 0.200
