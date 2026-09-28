"""
Configuration module for FastAPI backend.

This module manages environment variables and application settings.
Uses pydantic-settings for type-safe configuration management.
"""

from pydantic_settings import BaseSettings
from typing import Literal, Optional


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

    # Phase 0 narrative read source.  "fixture" (the default) serves the
    # committed fixture; "sqlite" reads the pipeline's persisted output and
    # never falls back to the fixture.  The database path and pipeline
    # version use the same variables, and the same defaults, as pipeline.py.
    PHASE0_NARRATIVE_SOURCE: Literal["fixture", "sqlite"] = "fixture"
    PHASE0_DATABASE_PATH: Optional[str] = None
    PHASE0_PIPELINE_VERSION: str = "phase0-v1"

    # API keys (for external services like Reddit API, etc.)
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance (singleton pattern)
settings = Settings()
