# tests/test_logging.py
import pytest
import structlog
import json
import io
import sys
from src.core.logging import setup_logging

def test_logging_setup(settings, capsys):
    """Test logging configuration."""
    setup_logging(settings)
    logger = structlog.get_logger()
    assert logger is not None

    # Log a test message
    logger.info("test message")
    
    # Capture the output
    captured = capsys.readouterr()
    output = captured.out.strip()
    
    # For development environment, check console output
    if settings.ENVIRONMENT == "development":
        assert "test message" in output
    # For other environments, verify JSON structure
    else:
        try:
            log_dict = json.loads(output)
            assert log_dict["event"] == "test message"
            assert log_dict["level"] == "info"
            assert "timestamp" in log_dict
        except json.JSONDecodeError as e:
            pytest.fail(f"Log output is not valid JSON: {output}")