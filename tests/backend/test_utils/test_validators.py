import pytest
from backend.utils.validators import validate_prediction_request

@pytest.mark.unit
def test_validate_prediction_request_valid():
    """Test validation of a well-formed prediction request."""
    valid_data = {
        "machineID": "1",
        "datetime": "2024-01-01 10:00:00",
        "volt": 170.5,
        "rotate": 450.2,
        "pressure": 100.1,
        "vibration": 40.5
    }
    
    is_valid, msg, code = validate_prediction_request(valid_data)
    assert is_valid is True
    assert msg == ""
    assert code == ""

@pytest.mark.unit
def test_validate_prediction_request_missing_field():
    """Test validation fails when a required field is missing."""
    invalid_data = {
        "datetime": "2024-01-01 10:00:00",
        "volt": 170.5
    }
    
    is_valid, msg, code = validate_prediction_request(invalid_data)
    assert is_valid is False
    assert "machineID" in msg
    assert code == "MISSING_FIELD"

@pytest.mark.unit
def test_validate_prediction_request_invalid_type():
    """Test validation fails when a numeric field receives a string."""
    invalid_data = {
        "machineID": "1",
        "datetime": "2024-01-01 10:00:00",
        "volt": "not_a_number",
        "rotate": 450.2,
        "pressure": 100.1,
        "vibration": 40.5
    }
    
    is_valid, msg, code = validate_prediction_request(invalid_data)
    assert is_valid is False
    assert "numeric" in msg.lower()
    assert code == "INVALID_TYPE"
