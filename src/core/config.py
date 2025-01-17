from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from enum import Enum
import os
from pathlib import Path

class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class Settings(BaseSettings):
    # Basic Settings
    APP_NAME: str = "Interview Preparation Tool"
    ENVIRONMENT: EnvironmentType = EnvironmentType.DEVELOPMENT
    DEBUG: bool = True
    API_PREFIX: str = "/api"
    WEBSOCKET_PATH: str = "/ws"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "DEBUG"
    
    # AI Settings
    OPENAI_API_KEY: str | None = None
    AI_MODEL: str = "gpt-4-turbo-preview"
    
    # Video Processing
    VIDEO_FRAME_RATE: int = 30
    EMOTION_DETECTION_INTERVAL: int = 5
    
    # Audio Processing
    AUDIO_CHUNK_SIZE: int = 1024
    AUDIO_SAMPLE_RATE: int = 16000
    
    # Database
    DATABASE_URL: str = "sqlite:///./interview_sessions.db"
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    LOGS_DIR: Path = BASE_DIR / "logs"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra='ignore'
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()