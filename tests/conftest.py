# tests/conftest.py
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import os

@pytest.fixture
def test_env_vars():
    """Set up test environment variables."""
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "true"
    os.environ["APP_NAME"] = "Interview Prep Test"
    yield
    # Clean up
    os.environ.pop("ENVIRONMENT", None)
    os.environ.pop("DEBUG", None)
    os.environ.pop("APP_NAME", None)

@pytest.fixture
def settings(test_env_vars):
    """Get test settings."""
    from src.core.config import get_settings
    return get_settings()

@pytest.fixture
def app(settings):
    """Create test app instance."""
    from src.interface.api.main import create_app
    return create_app()

@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)