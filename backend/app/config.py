"""
Configuration module for FastAPI backend.

This module manages environment variables and application settings.
Uses pydantic-settings for type-safe configuration management.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # App settings
    APP_NAME: str = "AI Market Sentiment API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # External services (placeholders - update when services are live)
    NLP_SERVICE_URL: Optional[str] = None
    PREDICTION_SERVICE_URL: Optional[str] = None
    DATA_SERVICE_URL: Optional[str] = None

    # API keys (for external services like Reddit API, etc.)
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: Optional[str] = None

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        """Accept common env-style strings in addition to strict booleans."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "false", "0", "no", "off"}:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "yes", "on"}:
                return True
        return value

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance (singleton pattern)
settings = Settings()
