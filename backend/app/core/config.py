"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Career Match API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql://career_match:career_match@localhost:5432/career_match"

    # Authentication
    SECRET_KEY: str = "CHANGE-ME-IN-PRODUCTION"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # File upload
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list[str] = ["pdf", "docx", "txt"]

    # ML / Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_VERSION: str = "match-model-v1.0"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
