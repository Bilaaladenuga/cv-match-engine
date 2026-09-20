"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

import json
import re
from functools import lru_cache

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
    CORS_ORIGINS: str = "http://localhost:3000"

    # Rate limiting (security: the API is public and the ML endpoints are
    # expensive — one abusive client must not be able to starve everyone).
    # Rules are '<limit>/<period>' with period in sec|min|hour.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "60/min"
    RATE_LIMIT_MATCHES: str = "5/min"
    RATE_LIMIT_RANKING: str = "5/min"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def split_cors_origins(raw: str | None) -> list[str]:
    """Parse a CORS_ORIGINS environment value into a list of origins.

    Deployment platforms (Render, Railway) store this as a JSON array, local
    `.env` files usually use a comma-separated list, and a single origin has
    no separator at all. All three are accepted:

        '["https://a.com", "https://b.com"]'  -> ["https://a.com", ...]
        'https://a.com,https://b.com'          -> ["https://a.com", ...]
        'https://a.com'                        -> ["https://a.com"]
        '*', '', None                          -> ['*'] / [] / []
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_cors_origin_regex(origins: list[str]) -> str | None:
    """Compile wildcard origins into a regex Starlette can match.

    Starlette's `allow_origins` compares literal strings, so a pattern like
    `https://*.vercel.app` (common for preview deploys) never matches. This
    converts each `*` into `.*` and anchors the result. Returns None when no
    wildcard pattern is present.
    """
    patterns = [origin for origin in origins if "*" in origin and origin != "*"]
    if not patterns:
        return None
    compiled = [re.escape(pattern).replace(r"\*", ".*") for pattern in patterns]
    return "^(" + "|".join(compiled) + ")$"
