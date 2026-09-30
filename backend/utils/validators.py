"""
validators.py
-------------
Centralised, reusable request-validation functions for the Predictive
Maintenance API.  All validation logic lives here so that routes.py
contains zero duplicated validation code.

Design rationale
----------------
- Returns a typed (is_valid, message, error_code) triple so callers can
  build a consistent JSON error response without any additional logic.
- Uses plain Python — no Flask imports — keeping the module testable
  in isolation.
- Sensor bounds are conservative physical limits derived from the
  Microsoft Azure PdM dataset distribution; they guard against
  accidental or malicious out-of-range inputs that could cause
  silent numeric overflow in the ML pipeline.
"""

from typing import Any, Dict, Optional, Tuple

# ─── Physical sensor bounds ───────────────────────────────────────────────────
# Values outside these ranges indicate corrupt / malformed input.
SENSOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "volt":      (0.0,  400.0),   # Voltage in Volts
    "rotate":    (0.0,  1500.0),  # Rotation speed in RPM
    "pressure":  (0.0,  400.0),   # Pressure in PSI
    "vibration": (0.0,  200.0),   # Vibration in mm/s
}

# Valid machine model identifiers as defined in PdM_machines.csv
VALID_MACHINE_MODELS = {"model1", "model2", "model3", "model4"}

# Machine age in years — domain bounds from the dataset
MACHINE_AGE_BOUNDS: Tuple[int, int] = (0, 50)

# Required top-level keys for a prediction/explain request
REQUIRED_PREDICTION_FIELDS = {"machineID"}


def _validate_result(message: str, error_code: str) -> Tuple[bool, str, str]:
    """Convenience helper to build a failed-validation result triple."""
    return False, message, error_code


def _ok() -> Tuple[bool, str, str]:
    """Convenience helper to build a passing-validation result triple."""
    return True, "", ""


# ─── Public API ───────────────────────────────────────────────────────────────

def validate_prediction_request(
    data: Optional[Any],
) -> Tuple[bool, str, str]:
    """
    Validate an incoming prediction (or explain) request body.

    Parameters
    ----------
    data : dict | None
        The parsed JSON body from flask.request.json.

    Returns
    -------
    (is_valid, human_readable_message, error_code)
        is_valid     – True iff all checks pass.
        message      – Empty string on success, descriptive sentence on failure.
        error_code   – Machine-readable code for the frontend (empty on success).

    Error Codes
    -----------
    MISSING_BODY          – Request body is null or empty.
    MISSING_FIELD         – A required field is absent.
    INVALID_TYPE          – A field cannot be cast to the expected type.
    INVALID_SENSOR_VALUE  – A sensor reading is outside physical bounds.
    INVALID_MACHINE_MODEL – machine model string is not in the allowed set.
    INVALID_MACHINE_AGE   – machine age is outside the allowed range.
    """
    # 1. Body presence
    if not data:
        return _validate_result(
            "Request body is missing or empty. Send a JSON object with machineID and sensor readings.",
            "MISSING_BODY",
        )

    if not isinstance(data, dict):
        return _validate_result(
            "Request body must be a JSON object.",
            "INVALID_TYPE",
        )

    # 2. Required fields
    for field in REQUIRED_PREDICTION_FIELDS:
        if field not in data:
            return _validate_result(
                f"Required field '{field}' is missing from the request body.",
                "MISSING_FIELD",
            )

    # 3. machineID must be a positive integer
    try:
        machine_id = int(data["machineID"])
        if machine_id <= 0:
            return _validate_result(
                f"'machineID' must be a positive integer. Received: {data['machineID']}",
                "INVALID_TYPE",
            )
    except (ValueError, TypeError):
        return _validate_result(
            f"'machineID' must be an integer. Received type {type(data['machineID']).__name__}.",
            "INVALID_TYPE",
        )

    # 4. Sensor readings — type + physical bounds
    for sensor, (lo, hi) in SENSOR_BOUNDS.items():
        if sensor in data:
            try:
                value = float(data[sensor])
            except (ValueError, TypeError):
                return _validate_result(
                    f"Sensor '{sensor}' must be a numeric value. "
                    f"Received type {type(data[sensor]).__name__}.",
                    "INVALID_TYPE",
                )

            if not (lo <= value <= hi):
                return _validate_result(
                    f"Sensor '{sensor}' value {value} is outside the valid physical range "
                    f"[{lo}, {hi}]. Please check the telemetry data.",
                    "INVALID_SENSOR_VALUE",
                )

    # 5. Optional: machine model string validation
    if "model" in data:
        model_val = str(data["model"]).strip().lower()
        if model_val not in VALID_MACHINE_MODELS:
            return _validate_result(
                f"'model' must be one of {sorted(VALID_MACHINE_MODELS)}. "
                f"Received: '{data['model']}'.",
                "INVALID_MACHINE_MODEL",
            )

    # 6. Optional: machine age validation
    if "age" in data:
        try:
            age = int(data["age"])
            lo, hi = MACHINE_AGE_BOUNDS
            if not (lo <= age <= hi):
                return _validate_result(
                    f"'age' must be between {lo} and {hi} years. Received: {age}.",
                    "INVALID_MACHINE_AGE",
                )
        except (ValueError, TypeError):
            return _validate_result(
                f"'age' must be an integer. Received type {type(data['age']).__name__}.",
                "INVALID_TYPE",
            )

    return _ok()


def validate_machine_id(machine_id: Any) -> Tuple[bool, str, str]:
    """
    Validate a machine ID extracted from a URL path parameter.

    Parameters
    ----------
    machine_id : any
        The raw value from the URL converter (usually already int via Flask).

    Returns
    -------
    (is_valid, message, error_code)
    """
    try:
        mid = int(machine_id)
        if mid <= 0:
            return _validate_result(
                f"Machine ID must be a positive integer. Received: {machine_id}",
                "INVALID_MACHINE_ID",
            )
    except (ValueError, TypeError):
        return _validate_result(
            f"Machine ID must be an integer. Received: {machine_id}",
            "INVALID_TYPE",
        )
    return _ok()
