"""Tests for agent_bridge.config – settings loading and validation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agent_bridge.config import Settings, get_settings


# Ensure tests never accidentally read a real .env file\[email protected](autouse=True)
def _clear_settings_cache():
    """Clear the lru_cache on get_settings before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestDefaultSettings:
    """Verify that sensible defaults are applied when env vars are absent."""

    def test_default_host(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}, clear=False):
            s = Settings()
        assert s.host == "0.0.0.0"

    def test_default_port(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}, clear=False):
            s = Settings()
        assert s.port == 8000

    def test_default_ai_backend_url(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}, clear=False):
            s = Settings()
        assert s.ai_backend_url == "http://localhost:11434"

    def test_default_ai_backend_type(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}, clear=False):
            s = Settings()
        assert s.ai_backend_type == "ollama"

    def test_default_model(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}, clear=False):
            s = Settings()
        assert s.ai_model == "llama3"

    def test_default_max_concurrent_tasks(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}, clear=False):
            s = Settings()
        assert s.max_concurrent_tasks == 3

    def test_empty_allowed_users_by_default(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}, clear=False):
            s = Settings()
        assert s.telegram_allowed_users == []
        assert s.discord_allowed_users == []


class TestEnvVarOverrides:
    """Verify that environment variables correctly override defaults."""

    def test_override_port(self):
        with patch.dict(os.environ, {"PORT": "9090", "AGENT_BRIDGE_TESTING": "1"}):
            s = Settings()
        assert s.port == 9090

    def test_override_ai_model(self):
        with patch.dict(os.environ, {"AI_MODEL": "codellama", "AGENT_BRIDGE_TESTING": "1"}):
            s = Settings()
        assert s.ai_model == "codellama"

    def test_telegram_token_loaded(self):
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "123:ABC", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert s.telegram_bot_token is not None
        assert s.telegram_bot_token.get_secret_value() == "123:ABC"

    def test_discord_token_loaded(self):
        with patch.dict(
            os.environ,
            {"DISCORD_BOT_TOKEN": "disc.token.here", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert s.discord_bot_token is not None
        assert s.discord_bot_token.get_secret_value() == "disc.token.here"

    def test_openai_backend_requires_api_key(self):
        env = {
            "AI_BACKEND_TYPE": "openai",
            "AI_API_KEY": "sk-test",
            "AGENT_BRIDGE_TESTING": "1",
        }
        with patch.dict(os.environ, env):
            s = Settings()
        assert s.ai_api_key is not None
        assert s.ai_api_key.get_secret_value() == "sk-test"

    def test_openai_backend_without_api_key_raises(self):
        env = {"AI_BACKEND_TYPE": "openai", "AGENT_BRIDGE_TESTING": "1"}
        # Remove AI_API_KEY if it happens to be set in the ambient environment
        clean_env = {k: v for k, v in os.environ.items() if k != "AI_API_KEY"}
        clean_env.update(env)
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(ValidationError, match="ai_api_key"):
                Settings()


class TestAllowedUserParsing:
    """Verify comma-separated user ID parsing."""

    def test_parse_telegram_allowed_users(self):
        with patch.dict(
            os.environ,
            {"TELEGRAM_ALLOWED_USERS": "111,222,333", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert s.telegram_allowed_users == [111, 222, 333]

    def test_parse_discord_allowed_users(self):
        with patch.dict(
            os.environ,
            {"DISCORD_ALLOWED_USERS": "99999,88888", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert s.discord_allowed_users == [99999, 88888]

    def test_empty_string_gives_empty_list(self):
        with patch.dict(
            os.environ,
            {"TELEGRAM_ALLOWED_USERS": "", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert s.telegram_allowed_users == []

    def test_single_user_id(self):
        with patch.dict(
            os.environ,
            {"TELEGRAM_ALLOWED_USERS": "42", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert s.telegram_allowed_users == [42]


class TestHelperProperties:
    """Verify computed helper properties on Settings."""

    def test_telegram_enabled_false_when_no_token(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}):
            s = Settings()
        assert s.telegram_enabled is False

    def test_telegram_enabled_true_when_token_set(self):
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "tok", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert s.telegram_enabled is True

    def test_discord_enabled_false_when_no_token(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}):
            s = Settings()
        assert s.discord_enabled is False

    def test_openai_chat_completions_url(self):
        with patch.dict(
            os.environ,
            {"AI_BACKEND_URL": "https://api.openai.com", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert s.openai_chat_completions_url == "https://api.openai.com/v1/chat/completions"

    def test_ollama_generate_url(self):
        with patch.dict(
            os.environ,
            {"AI_BACKEND_URL": "http://localhost:11434", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert s.ollama_generate_url == "http://localhost:11434/api/chat"

    def test_trailing_slash_stripped_from_urls(self):
        with patch.dict(
            os.environ,
            {"AI_BACKEND_URL": "http://localhost:11434/", "AGENT_BRIDGE_TESTING": "1"},
        ):
            s = Settings()
        assert not s.ollama_generate_url.endswith("//api/chat")


class TestGetSettingsSingleton:
    """Verify that get_settings() returns a cached singleton."""

    def test_returns_settings_instance(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}):
            s = get_settings()
        assert isinstance(s, Settings)

    def test_is_cached(self):
        with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}):
            s1 = get_settings()
            s2 = get_settings()
        assert s1 is s2
