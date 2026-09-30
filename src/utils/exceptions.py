"""
exceptions.py
-------------
Custom exception hierarchy for the Smart Factory PdM project.

All custom exceptions inherit from PdMBaseError, which itself inherits
from Exception. This allows callers to catch all project-specific errors
with a single except clause while still being able to catch specific
error types.

Author : Smart Factory PdM Team
PEP8   : Compliant
"""


class PdMBaseError(Exception):
    """Base exception for all Smart Factory PdM errors."""
    pass


class DataValidationError(PdMBaseError, ValueError):
    """Raised when data fails schema or quality validation checks."""
    pass


class DataLoadError(PdMBaseError, FileNotFoundError):
    """Raised when a required data file cannot be found or loaded."""
    pass


class ConfigError(PdMBaseError):
    """Raised when configuration is invalid or missing required keys."""
    pass


class PipelineError(PdMBaseError):
    """Raised when a pipeline stage fails unexpectedly."""
    pass


class ModelNotFoundError(PdMBaseError, FileNotFoundError):
    """Raised when a trained model file cannot be found."""
    pass


class FeatureEngineeringError(PdMBaseError):
    """Raised when feature engineering encounters an unrecoverable error."""
    pass


class PredictionError(PdMBaseError):
    """Raised when model inference fails."""
    pass


class DriftDetectionError(PdMBaseError):
    """Raised when drift detection encounters an error."""
    pass
