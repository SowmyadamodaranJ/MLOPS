"""
logger.py
---------
Centralized logging configuration for the Smart Factory PdM project.

All modules should obtain their logger via:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a named logger with both console and file handlers.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    log_dir : str
        Directory where log files are stored.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid duplicate handlers when function is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── File handler ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"pdm_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
