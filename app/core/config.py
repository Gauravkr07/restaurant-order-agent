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

from pydantic import Field, field_validator
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
    # Path to a CA bundle for a TLS Redis (rediss://) whose certificate
    # is signed by a CA the container doesn't already trust. Left unset,
    # the system trust store is used and certificates are still verified.
    redis_ssl_ca_certs: str | None = Field(default=None, alias="REDIS_SSL_CA_CERTS")

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, value: str) -> str:
        """Managed Postgres providers (Render, Heroku, Railway) hand out
        URLs with a bare `postgres://` or `postgresql://` scheme. This app
        uses psycopg 3, which SQLAlchemy only selects when the scheme says
        so explicitly - otherwise it reaches for psycopg2, which isn't
        installed, and the app dies at boot with ModuleNotFoundError.
        Normalising here means the provider's URL can be used verbatim."""
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+psycopg://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value

    # --- database pool (see app/domain/base.py) ---
    # Deliberately small: a managed Postgres caps total connections, and
    # several processes share that budget (web + Celery worker, each with
    # its own pool). Neon's pooled connection string raises that ceiling
    # a lot, but the app shouldn't depend on it.
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=5, alias="DB_MAX_OVERFLOW")
    db_pool_recycle_seconds: int = Field(default=300, alias="DB_POOL_RECYCLE_SECONDS")

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
