"""
logger.py
---------
Centralised logging configuration for the Predictive Maintenance API.

Improvements over the original
-------------------------------
- StructuredLogAdapter injects optional context fields (request_id,
  endpoint, machine_id) into every log record so log aggregators can
  filter and group entries without regex parsing.
- get_logger() signature is unchanged — all existing call-sites keep
  working with zero modifications.
- A module-level get_context_logger() factory is provided for routes
  that want to attach per-request context to their logger.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"


class StructuredLogAdapter(logging.LoggerAdapter):
    """
    A LoggerAdapter that merges caller-supplied context into every
    log record's 'extra' dict.  The formatter then picks up any
    extra keys it references.

    Usage
    -----
    logger = get_context_logger(__name__, request_id="abc", endpoint="/predict")
    logger.info("Running inference")
    # → 2026-07-23 15:00:00 | INFO | backend.api.routes | [req=abc endpoint=/predict] Running inference
    """

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> Tuple[str, MutableMapping[str, Any]]:
        # Build a bracketed context prefix from non-empty context fields.
        ctx_parts = [
            f"{k}={v}"
            for k, v in self.extra.items()
            if v is not None and v != ""
        ]
        prefix = f"[{' '.join(ctx_parts)}] " if ctx_parts else ""
        return f"{prefix}{msg}", kwargs


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a named logger with console and rolling file handlers.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module.
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.  Idempotent — calling twice with the
        same name returns the same logger without adding duplicate handlers.
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Guard: do not add handlers if they already exist (e.g. app restart)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — useful in Docker / terminal contexts
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Daily file handler — one file per calendar day
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = LOGS_DIR / f"api_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def get_context_logger(
    name: str,
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    machine_id: Optional[Any] = None,
    level: int = logging.INFO,
) -> StructuredLogAdapter:
    """
    Return a StructuredLogAdapter wrapping a standard logger.  The
    supplied context fields are prepended to every log message emitted
    through this adapter, making it easy to correlate log lines from
    the same request.

    Parameters
    ----------
    name : str
        Passed to get_logger() — usually __name__.
    request_id : str, optional
        A unique request identifier (auto-generated UUID if omitted).
    endpoint : str, optional
        The Flask route path, e.g. "/predict".
    machine_id : int | str, optional
        The target machine ID for machine-scoped operations.
    level : int
        Logging level.

    Returns
    -------
    StructuredLogAdapter
        Drop-in replacement for a standard logger.
    """
    base_logger = get_logger(name, level)
    context: Dict[str, Any] = {
        "req": request_id or str(uuid.uuid4())[:8],
    }
    if endpoint:
        context["endpoint"] = endpoint
    if machine_id is not None:
        context["machine"] = machine_id

    return StructuredLogAdapter(base_logger, context)
