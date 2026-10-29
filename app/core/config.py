"""
All environment-driven configuration in one typed, validated place.

Previously these values were read with os.environ.get() scattered across
db/base.py, tasks/celery_app.py, helpers/llm.py and service.py, each with
its own inline default. That meant a missing or malformed GROQ_API_KEY
only surfaced mid-conversation, as a 502 from the first LLM call.

Settings are instantiated once at import time, so a bad value fails at
boot with a clear pydantic error instead of at request time.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- external services ---
    groq_api_key: str = Field(alias="GROQ_API_KEY")
    database_url: str = Field(
        default="postgresql+psycopg://restaurant:restaurant@localhost:5433/restaurant_order_agent",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # --- LLM ---
    llm_model: str = Field(default="openai/gpt-oss-120b", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_timeout_seconds: int = Field(default=15, alias="LLM_TIMEOUT_SECONDS")

    # --- logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- session limits (were module constants in service.py) ---
    session_ttl_seconds: int = Field(default=60 * 60 * 24, alias="SESSION_TTL_SECONDS")
    max_turns_per_session: int = Field(default=50, alias="MAX_TURNS_PER_SESSION")

    # --- checkout locking (were module constants in service.py) ---
    checkout_lock_timeout_seconds: int = Field(default=10, alias="CHECKOUT_LOCK_TIMEOUT_SECONDS")
    checkout_lock_wait_seconds: int = Field(default=8, alias="CHECKOUT_LOCK_WAIT_SECONDS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached so every import shares one instance (and one validation)."""
    return Settings()


settings = get_settings()
