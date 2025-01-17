# tests/test_config.py
import pytest
from src.core.config import Settings, EnvironmentType

def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()
    assert settings.APP_NAME == "Interview Preparation Tool"
    assert settings.ENVIRONMENT == EnvironmentType.DEVELOPMENT
    assert settings.DEBUG is True
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000

def test_settings_environment_override(test_env_vars):
    """Test environment variable overrides."""
    settings = Settings()
    assert settings.APP_NAME == "Interview Prep Test"
    assert settings.ENVIRONMENT == EnvironmentType.TESTING