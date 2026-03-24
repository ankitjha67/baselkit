"""Tests for structured logging configuration."""

import json
import logging
import tempfile

import pytest

from creditriskengine.core.logging_config import (
    JSONFormatter,
    configure_logging,
)


class TestJSONFormatter:
    def test_basic_format(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="creditriskengine.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["logger"] == "creditriskengine.test"
        assert parsed["line"] == 42
        assert "timestamp" in parsed

    def test_exception_format(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_extra_fields(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="Calc", args=(), exc_info=None,
        )
        record.exposure_id = "EXP001"  # type: ignore[attr-defined]
        record.jurisdiction = "eu"  # type: ignore[attr-defined]
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["exposure_id"] == "EXP001"
        assert parsed["jurisdiction"] == "eu"


class TestConfigureLogging:
    def test_default_config(self) -> None:
        configure_logging()
        logger = logging.getLogger("creditriskengine")
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1

    def test_debug_level(self) -> None:
        configure_logging(level="DEBUG")
        logger = logging.getLogger("creditriskengine")
        assert logger.level == logging.DEBUG

    def test_json_format(self) -> None:
        configure_logging(json_format=True)
        logger = logging.getLogger("creditriskengine")
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_file_handler(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            configure_logging(log_file=f.name)
        logger = logging.getLogger("creditriskengine")
        assert isinstance(logger.handlers[0], logging.FileHandler)

    def test_clears_existing_handlers(self) -> None:
        configure_logging()
        configure_logging()
        logger = logging.getLogger("creditriskengine")
        assert len(logger.handlers) == 1

    def test_warning_level(self) -> None:
        configure_logging(level="WARNING")
        logger = logging.getLogger("creditriskengine")
        assert logger.level == logging.WARNING
