"""
config_loader.py
----------------
Utility to load and expose the project YAML configuration.

Usage
-----
    from src.utils.config_loader import load_config
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_data_dir"]
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default config path relative to project root
DEFAULT_CONFIG_PATH = os.path.join(
    Path(__file__).resolve().parents[2], "configs", "config.yaml"
)


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load the YAML configuration file into a Python dictionary.

    Parameters
    ----------
    config_path : str
        Absolute or relative path to the YAML config file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )

    logger.info("Loading configuration from: %s", config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    logger.info("Configuration loaded successfully.")
    return config
