"""
Structured logging configuration for CreditRiskEngine.

Provides sensible defaults for production and development use.
Supports JSON structured logging for log aggregation systems
(ELK, Splunk, CloudWatch, etc.).

Usage:
    from creditriskengine.core.logging_config import configure_logging
    configure_logging()  # Development defaults
    configure_logging(level="WARNING", json_format=True)  # Production
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production environments.

    Produces one JSON object per log line, suitable for ingestion by
    log aggregation tools (ELK stack, Splunk, AWS CloudWatch, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Include any extra fields attached to the record
        for key in ("exposure_id", "jurisdiction", "approach", "calculation_id"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, default=str)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure logging for CreditRiskEngine.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON structured format (recommended for production).
        log_file: Optional file path for log output. If None, logs to stderr.

    Example:
        >>> from creditriskengine.core.logging_config import configure_logging
        >>> configure_logging(level="DEBUG")  # Development
        >>> configure_logging(level="WARNING", json_format=True)  # Production
        >>> configure_logging(level="INFO", log_file="/var/log/cre.log")
    """
    root_logger = logging.getLogger("creditriskengine")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if log_file:
        handler: logging.Handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
