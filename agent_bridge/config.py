"""Configuration management for agent_bridge.

Loads and validates all application configuration from environment variables
using pydantic-settings. Supports a .env file for local development.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for the agent_bridge application.

    Values are loaded from environment variables or a .env file.
    Required fields have no default and must be provided at runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Server settings
    # ------------------------------------------------------------------ #
    host: str = Field(default="0.0.0.0", description="Bind host for the uvicorn server")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port for the uvicorn server")
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info", description="Logging verbosity level"
    )
    secret_token: SecretStr = Field(
        default=SecretStr("changeme"),
        description="Shared secret used to authenticate incoming webhook requests",
    )

    # ------------------------------------------------------------------ #
    # Telegram settings
    # ------------------------------------------------------------------ #
    telegram_bot_token: SecretStr | None = Field(
        default=None,
        description="Telegram Bot API token obtained from @BotFather",
    )
    telegram_webhook_url: str | None = Field(
        default=None,
        description="Public HTTPS URL where Telegram will POST updates (e.g. https://example.com/telegram/webhook)",
    )
    telegram_allowed_users: list[int] = Field(
        default_factory=list,
        description="Comma-separated list of Telegram user IDs permitted to use the bot",
    )

    # ------------------------------------------------------------------ #
    # Discord settings
    # ------------------------------------------------------------------ #
    discord_bot_token: SecretStr | None = Field(
        default=None,
        description="Discord bot token from the Discord Developer Portal",
    )
    discord_allowed_users: list[int] = Field(
        default_factory=list,
        description="Comma-separated list of Discord user IDs permitted to use the bot",
    )
    discord_allowed_guilds: list[int] = Field(
        default_factory=list,
        description="Comma-separated list of Discord guild IDs where the bot operates (empty = all guilds)",
    )

    # ------------------------------------------------------------------ #
    # AI backend settings
    # ------------------------------------------------------------------ #
    ai_backend_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the AI backend (Ollama default or any OpenAI-compatible API)",
    )
    ai_backend_type: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="Backend type: 'ollama' or 'openai' (any OpenAI-compatible API)",
    )
    ai_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the AI backend (required for OpenAI; optional for Ollama)",
    )
    ai_model: str = Field(
        default="llama3",
        description="Model name to use for completions (e.g. 'llama3', 'gpt-4o', 'codellama')",
    )
    ai_max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum number of tokens to generate in a single completion",
    )
    ai_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for AI completions (lower = more deterministic)",
    )
    ai_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description="HTTP timeout in seconds when calling the AI backend",
    )
    ai_system_prompt: str = Field(
        default=(
            "You are an expert AI coding assistant. "
            "Provide clear, concise, and correct code and explanations. "
            "Format code in fenced code blocks with the appropriate language tag."
        ),
        description="System prompt prepended to every AI request",
    )

    # ------------------------------------------------------------------ #
    # Task queue settings
    # ------------------------------------------------------------------ #
    max_concurrent_tasks: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Maximum number of AI tasks that may run concurrently",
    )
    task_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Maximum wall-clock time in seconds before a queued task is cancelled",
    )

    # ------------------------------------------------------------------ #
    # Database settings
    # ------------------------------------------------------------------ #
    database_url: str = Field(
        default="agent_bridge.db",
        description="Path to the SQLite database file (use ':memory:' for in-memory testing)",
    )

    # ------------------------------------------------------------------ #
    # Webhook callback settings
    # ------------------------------------------------------------------ #
    callback_url: str | None = Field(
        default=None,
        description="Optional external URL to POST task completion payloads to",
    )
    callback_secret: SecretStr | None = Field(
        default=None,
        description="Optional shared secret sent in the X-Callback-Secret header",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("telegram_allowed_users", "discord_allowed_users", "discord_allowed_guilds", mode="before")
    @classmethod
    def _parse_int_list(cls, value: object) -> list[int]:
        """Parse a comma-separated string of integers into a list.

        Accepts either a pre-parsed list or a comma-separated string such as
        ``"123,456,789"``.
        """
        if isinstance(value, list):
            return [int(v) for v in value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            return [int(part.strip()) for part in stripped.split(",") if part.strip()]
        if isinstance(value, int):
            return [value]
        return []

    @model_validator(mode="after")
    def _validate_ai_backend(self) -> "Settings":
        """Ensure that OpenAI backend is accompanied by an API key."""
        if self.ai_backend_type == "openai" and self.ai_api_key is None:
            raise ValueError(
                "ai_api_key is required when ai_backend_type is 'openai'. "
                "Set the AI_API_KEY environment variable."
            )
        return self

    @model_validator(mode="after")
    def _warn_no_bots_configured(self) -> "Settings":
        """Validate that at least one bot token is provided in non-test environments."""
        testing = os.environ.get("AGENT_BRIDGE_TESTING", "").lower() in ("1", "true", "yes")
        if not testing and self.telegram_bot_token is None and self.discord_bot_token is None:
            import warnings

            warnings.warn(
                "Neither TELEGRAM_BOT_TOKEN nor DISCORD_BOT_TOKEN is set. "
                "The bridge will not be able to receive or send messages.",
                UserWarning,
                stacklevel=2,
            )
        return self

    # ------------------------------------------------------------------ #
    # Helper properties
    # ------------------------------------------------------------------ #
    @property
    def telegram_enabled(self) -> bool:
        """Return True if the Telegram bot token is configured."""
        return self.telegram_bot_token is not None

    @property
    def discord_enabled(self) -> bool:
        """Return True if the Discord bot token is configured."""
        return self.discord_bot_token is not None

    @property
    def openai_chat_completions_url(self) -> str:
        """Return the full chat completions endpoint URL for an OpenAI-compatible backend."""
        base = self.ai_backend_url.rstrip("/")
        return f"{base}/v1/chat/completions"

    @property
    def ollama_generate_url(self) -> str:
        """Return the full generate endpoint URL for the Ollama backend."""
        base = self.ai_backend_url.rstrip("/")
        return f"{base}/api/chat"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings singleton.

    The result is cached so that environment variables are read only once.
    In tests, call ``get_settings.cache_clear()`` after patching environment
    variables to force re-evaluation.

    Returns:
        A fully validated :class:`Settings` instance.

    Raises:
        pydantic.ValidationError: If any required field is missing or invalid.
    """
    return Settings()
