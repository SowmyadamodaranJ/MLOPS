import pytest
from backend.app import create_app
from backend.services.db_service import DatabaseService
import os

@pytest.fixture(scope="session")
def test_app():
    """Create a Flask application context for testing."""
    os.environ["FLASK_ENV"] = "testing"
    # Ensure test database doesn't overwrite production DB
    os.environ["DATABASE_PATH"] = "data/test_predictions.db"
    
    app = create_app()
    app.config["TESTING"] = True
    
    yield app

@pytest.fixture
def test_client(test_app):
    """A test client for the app."""
    return test_app.test_client()

@pytest.fixture
def mock_db(monkeypatch):
    """Mock the database service to prevent writing to disk during unit tests."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.check_connection.return_value = True
    mock.log_prediction.return_value = 1
    monkeypatch.setattr("backend.api.routes.db_service", mock)
    return mock

@pytest.fixture
def valid_prediction_payload():
    """A valid JSON payload for the /predict endpoint."""
    return {
        "machineID": "1",
        "datetime": "2024-01-01 10:00:00",
        "volt": 170.5,
        "rotate": 450.2,
        "pressure": 100.1,
        "vibration": 40.5
    }

@pytest.fixture
def benchmark():
    """Fallback benchmark fixture when pytest-benchmark is not installed."""
    import time

    class BenchmarkHelper:
        def __init__(self):
            class Stats:
                def __init__(self):
                    self.mean = 0.0
            class Container:
                def __init__(self):
                    self.stats = Stats()
            self.stats = Container()

        def __call__(self, func, *args, **kwargs):
            start = time.perf_counter()
            res = func(*args, **kwargs)
            self.stats.stats.mean = time.perf_counter() - start
            return res

    return BenchmarkHelper()

