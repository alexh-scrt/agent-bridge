"""Integration tests for agent_bridge HTTP routes using FastAPI TestClient.

Tests cover:
- Health check endpoint
- Telegram webhook ingestion (valid, invalid user, no text, commands)
- Discord webhook ingestion (valid, invalid user, bot author, commands)
- Task REST API (list, get, delete)
- Manual deliver and callback trigger endpoints
- Application factory and lifespan tests
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from agent_bridge.config import Settings, get_settings
from agent_bridge.db import Database
from agent_bridge.messenger import Messenger
from agent_bridge.models import (
    Platform,
    TaskCreate,
    TaskRecord,
    TaskStatus,
    TaskStatusUpdate,
)
from agent_bridge.queue import TaskQueue


# ---------------------------------------------------------------------------
# Settings cache cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear the LRU-cached settings before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings() -> Settings:
    """Settings for testing with both Telegram and Discord enabled."""
    with patch.dict(
        os.environ,
        {
            "AGENT_BRIDGE_TESTING": "1",
            "TELEGRAM_BOT_TOKEN": "123456:TestToken",
            "DISCORD_BOT_TOKEN": "discord-test-token",
            "AI_BACKEND_TYPE": "ollama",
            "AI_BACKEND_URL": "http://localhost:11434",
            "AI_MODEL": "llama3",
            "SECRET_TOKEN": "test-secret",
            "MAX_CONCURRENT_TASKS": "2",
            "TASK_TIMEOUT_SECONDS": "30",
            "DATABASE_URL": ":memory:",
        },
    ):
        get_settings.cache_clear()
        yield Settings()


# ---------------------------------------------------------------------------
# Synchronous TestClient fixture (primary)
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_client(test_settings: Settings):
    """Synchronous TestClient backed by an in-memory DB."""
    from agent_bridge.app import create_app

    _app = create_app(settings=test_settings, testing=True)
    with TestClient(_app, raise_server_exceptions=True) as client:
        yield client, _app


# ---------------------------------------------------------------------------
# Helper payload builders
# ---------------------------------------------------------------------------


def _telegram_update(
    update_id: int = 1,
    message_id: int = 10,
    user_id: int = 42,
    chat_id: int = 100,
    text: str = "Write a hello world function",
    is_bot: bool = False,
) -> dict[str, Any]:
    """Build a minimal Telegram update payload."""
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "from": {
                "id": user_id,
                "is_bot": is_bot,
                "first_name": "Test",
                "username": "testuser",
            },
            "chat": {"id": chat_id, "type": "private"},
            "date": 1700000000,
            "text": text,
        },
    }


def _discord_payload(
    message_id: str = "111",
    channel_id: str = "222",
    user_id: str = "333",
    username: str = "testuser",
    content: str = "Write a hello world function",
    is_bot: bool = False,
    guild_id: str | None = None,
) -> dict[str, Any]:
    """Build a minimal Discord inbound payload."""
    return {
        "message": {
            "id": message_id,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "author": {
                "id": user_id,
                "username": username,
                "discriminator": "0",
                "bot": is_bot,
            },
            "content": content,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    }


def _mark_task_done(app: Any, task_id: str, result: str = "Great result") -> None:
    """Synchronously update a task to DONE status via a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            app.state.db.update_task_status(
                task_id,
                TaskStatusUpdate(status=TaskStatus.DONE, result=result),
            )
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for the GET /health endpoint."""

    def test_health_returns_200(self, sync_client: tuple) -> None:
        """Health endpoint must return HTTP 200."""
        client, _app = sync_client
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, sync_client: tuple) -> None:
        """Health response body must have status='ok'."""
        client, _app = sync_client
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_response_has_version(self, sync_client: tuple) -> None:
        """Health response must include a version field."""
        client, _app = sync_client
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_version_matches_package(self, sync_client: tuple) -> None:
        """Health response version must match the installed package version."""
        from agent_bridge import __version__

        client, _app = sync_client
        data = client.get("/health").json()
        assert data["version"] == __version__


# ---------------------------------------------------------------------------
# Telegram webhook
# ---------------------------------------------------------------------------


class TestTelegramWebhook:
    """Tests for POST /telegram/webhook."""

    def test_valid_message_returns_200(self, sync_client: tuple) -> None:
        """A valid Telegram message should return HTTP 200."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        resp = client.post("/telegram/webhook", json=_telegram_update())
        assert resp.status_code == 200

    def test_valid_message_returns_ok_true(self, sync_client: tuple) -> None:
        """A valid Telegram message should return {ok: true} in the body."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        data = client.post("/telegram/webhook", json=_telegram_update()).json()
        assert data["ok"] is True

    def test_valid_message_returns_task_id(self, sync_client: tuple) -> None:
        """A valid Telegram message should return a task_id in the response."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        data = client.post("/telegram/webhook", json=_telegram_update()).json()
        assert "task_id" in data
        assert isinstance(data["task_id"], str)
        assert len(data["task_id"]) == 36  # UUID

    def test_valid_message_calls_queue_enqueue(self, sync_client: tuple) -> None:
        """A valid message must call queue.enqueue exactly once."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update())
        app.state.queue.enqueue.assert_awaited_once()

    def test_valid_message_calls_send_ack(self, sync_client: tuple) -> None:
        """An ack must be sent to the user after successful enqueueing."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update())
        app.state.messenger.send_ack.assert_awaited_once()

    def test_ack_uses_correct_platform_and_chat(self, sync_client: tuple) -> None:
        """The ack should use Platform.TELEGRAM and the correct chat_id."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update(chat_id=100))

        call_kwargs = app.state.messenger.send_ack.call_args.kwargs
        assert call_kwargs.get("platform") == Platform.TELEGRAM
        assert call_kwargs.get("chat_id") == "100"

    def test_no_message_field_returns_ok_ignored(self, sync_client: tuple) -> None:
        """An update without a 'message' field should be silently ignored."""
        client, _app = sync_client
        payload = {"update_id": 99}  # no 'message'
        resp = client.post("/telegram/webhook", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data.get("ignored") is True

    def test_empty_text_is_ignored(self, sync_client: tuple) -> None:
        """A message with empty text should be ignored without enqueueing."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()

        resp = client.post("/telegram/webhook", json=_telegram_update(text=""))
        assert resp.status_code == 200
        app.state.queue.enqueue.assert_not_awaited()

    def test_whitespace_only_text_is_ignored(self, sync_client: tuple) -> None:
        """A message with only whitespace should be ignored."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()

        resp = client.post("/telegram/webhook", json=_telegram_update(text="   "))
        assert resp.status_code == 200
        app.state.queue.enqueue.assert_not_awaited()

    def test_disallowed_user_returns_ok_ignored(self, sync_client: tuple) -> None:
        """A user not in the allowlist should be rejected (ok=True, ignored=True)."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"telegram_allowed_users": [999]}
        )
        app.state.messenger.send_not_authorised = AsyncMock()
        app.state.queue.enqueue = AsyncMock()

        resp = client.post(
            "/telegram/webhook", json=_telegram_update(user_id=42, text="hello")
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data.get("ignored") is True

    def test_disallowed_user_does_not_enqueue(self, sync_client: tuple) -> None:
        """A rejected user's message must never be enqueued."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"telegram_allowed_users": [999]}
        )
        app.state.messenger.send_not_authorised = AsyncMock()
        app.state.queue.enqueue = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update(user_id=42))
        app.state.queue.enqueue.assert_not_awaited()

    def test_disallowed_user_sends_not_authorised(self, sync_client: tuple) -> None:
        """A rejected user should receive a not-authorised message."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"telegram_allowed_users": [999]}
        )
        app.state.messenger.send_not_authorised = AsyncMock()
        app.state.queue.enqueue = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update(user_id=42))
        app.state.messenger.send_not_authorised.assert_awaited_once()

    def test_allowed_user_passes_through(self, sync_client: tuple) -> None:
        """A user explicitly in the allowlist should be accepted."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"telegram_allowed_users": [42]}
        )
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        resp = client.post(
            "/telegram/webhook", json=_telegram_update(user_id=42, text="hello")
        )
        assert resp.status_code == 200
        app.state.queue.enqueue.assert_awaited_once()

    def test_empty_allowlist_accepts_all_users(self, sync_client: tuple) -> None:
        """An empty allowlist should allow any user through."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"telegram_allowed_users": []}
        )
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        resp = client.post(
            "/telegram/webhook", json=_telegram_update(user_id=9999, text="allowed")
        )
        assert resp.status_code == 200
        app.state.queue.enqueue.assert_awaited_once()

    # ---- /status command ----

    def test_status_command_no_id_sends_usage_hint(self, sync_client: tuple) -> None:
        """The /status command without a task ID should send a usage hint."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        resp = client.post("/telegram/webhook", json=_telegram_update(text="/status"))
        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited()
        call_text = app.state.messenger.send_telegram.call_args.kwargs.get("text", "")
        assert "/status" in call_text or "usage" in call_text.lower()

    def test_status_command_unknown_task_sends_not_found(self, sync_client: tuple) -> None:
        """The /status command with an unknown task ID should send a not-found message."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        resp = client.post(
            "/telegram/webhook",
            json=_telegram_update(text="/status nonexistent-task-id"),
        )
        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited()

    def test_status_command_existing_task_calls_send_status(self, sync_client: tuple) -> None:
        """The /status command for a real task should call messenger.send_status."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        app.state.messenger.send_status = AsyncMock()
        app.state.messenger.send_telegram = AsyncMock()

        # Create a task
        create_resp = client.post(
            "/telegram/webhook",
            json=_telegram_update(text="Write something", update_id=1),
        )
        task_id = create_resp.json().get("task_id", "")
        assert task_id

        # Query its status
        client.post(
            "/telegram/webhook",
            json=_telegram_update(text=f"/status {task_id}", update_id=2),
        )
        app.state.messenger.send_status.assert_awaited_once()

    # ---- /tasks command ----

    def test_tasks_command_empty_sends_message(self, sync_client: tuple) -> None:
        """The /tasks command when no tasks exist should send an empty-list message."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        resp = client.post("/telegram/webhook", json=_telegram_update(text="/tasks"))
        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited()

    def test_tasks_command_with_tasks_sends_list(self, sync_client: tuple) -> None:
        """The /tasks command with existing tasks should send a formatted list."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        # Create a task first
        client.post("/telegram/webhook", json=_telegram_update(text="My task", update_id=1))

        app.state.messenger.send_telegram = AsyncMock()
        resp = client.post("/telegram/webhook", json=_telegram_update(text="/tasks", update_id=2))
        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited()

    # ---- /help and /start commands ----

    def test_help_command_sends_help_message(self, sync_client: tuple) -> None:
        """The /help command should send exactly one help message."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        resp = client.post("/telegram/webhook", json=_telegram_update(text="/help"))
        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited_once()

    def test_start_command_sends_help_message(self, sync_client: tuple) -> None:
        """The /start command should send exactly one help message."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        resp = client.post("/telegram/webhook", json=_telegram_update(text="/start"))
        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited_once()

    def test_help_command_does_not_enqueue(self, sync_client: tuple) -> None:
        """The /help command must not enqueue an AI task."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()
        app.state.queue.enqueue = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update(text="/help"))
        app.state.queue.enqueue.assert_not_awaited()

    # ---- Task stored in DB ----

    def test_enqueued_task_stored_in_db(self, sync_client: tuple) -> None:
        """The task created from a Telegram message must be persisted in the DB."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        resp = client.post(
            "/telegram/webhook",
            json=_telegram_update(text="Store me in DB"),
        )
        task_id = resp.json()["task_id"]

        # Retrieve via REST API
        get_resp = client.get(f"/tasks/{task_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["platform"] == "telegram"
        assert data["prompt"] == "Store me in DB"


# ---------------------------------------------------------------------------
# Discord webhook
# ---------------------------------------------------------------------------


class TestDiscordWebhook:
    """Tests for POST /discord/webhook."""

    def test_valid_message_returns_200(self, sync_client: tuple) -> None:
        """A valid Discord message should return HTTP 200."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        resp = client.post("/discord/webhook", json=_discord_payload())
        assert resp.status_code == 200

    def test_valid_message_returns_ok_true(self, sync_client: tuple) -> None:
        """A valid Discord message should return {ok: true}."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        data = client.post("/discord/webhook", json=_discord_payload()).json()
        assert data["ok"] is True

    def test_valid_message_returns_task_id(self, sync_client: tuple) -> None:
        """A valid Discord message should return a task_id."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        data = client.post("/discord/webhook", json=_discord_payload()).json()
        assert "task_id" in data

    def test_valid_message_calls_queue_enqueue(self, sync_client: tuple) -> None:
        """A valid Discord message must call queue.enqueue."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/discord/webhook", json=_discord_payload())
        app.state.queue.enqueue.assert_awaited_once()

    def test_ack_sent_after_enqueue(self, sync_client: tuple) -> None:
        """An ack must be sent after successfully enqueueing a Discord task."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/discord/webhook", json=_discord_payload())
        app.state.messenger.send_ack.assert_awaited_once()

    def test_ack_uses_correct_platform_and_channel(self, sync_client: tuple) -> None:
        """The ack must use Platform.DISCORD and the correct channel_id."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/discord/webhook", json=_discord_payload(channel_id="999"))

        call_kwargs = app.state.messenger.send_ack.call_args.kwargs
        assert call_kwargs.get("platform") == Platform.DISCORD
        assert call_kwargs.get("chat_id") == "999"

    def test_bot_author_ignored(self, sync_client: tuple) -> None:
        """Messages from bots should be silently ignored."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()

        resp = client.post("/discord/webhook", json=_discord_payload(is_bot=True))
        assert resp.status_code == 200
        assert resp.json().get("ignored") is True
        app.state.queue.enqueue.assert_not_awaited()

    def test_empty_content_ignored(self, sync_client: tuple) -> None:
        """Messages with empty content should be ignored without enqueueing."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()

        resp = client.post("/discord/webhook", json=_discord_payload(content=""))
        assert resp.status_code == 200
        assert resp.json().get("ignored") is True
        app.state.queue.enqueue.assert_not_awaited()

    def test_whitespace_content_ignored(self, sync_client: tuple) -> None:
        """Messages with whitespace-only content should be ignored."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()

        resp = client.post("/discord/webhook", json=_discord_payload(content="   "))
        assert resp.status_code == 200
        app.state.queue.enqueue.assert_not_awaited()

    def test_disallowed_user_returns_ok_ignored(self, sync_client: tuple) -> None:
        """Discord users not in the allowlist should be rejected."""
        client, app = sync_client
        # user id 333 is the default in _discord_payload; only allow 999
        app.state.settings = app.state.settings.model_copy(
            update={"discord_allowed_users": [999999]}
        )
        app.state.messenger.send_not_authorised = AsyncMock()
        app.state.queue.enqueue = AsyncMock()

        resp = client.post("/discord/webhook", json=_discord_payload(user_id="333"))
        assert resp.status_code == 200
        assert resp.json().get("ignored") is True
        app.state.queue.enqueue.assert_not_awaited()

    def test_disallowed_guild_returns_ok_ignored(self, sync_client: tuple) -> None:
        """Messages from guilds not in the allowlist should be rejected."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"discord_allowed_guilds": [111111]}
        )
        app.state.queue.enqueue = AsyncMock()

        resp = client.post(
            "/discord/webhook",
            json=_discord_payload(content="hello", guild_id="999999"),
        )
        assert resp.status_code == 200
        assert resp.json().get("ignored") is True
        app.state.queue.enqueue.assert_not_awaited()

    def test_allowed_guild_passes_through(self, sync_client: tuple) -> None:
        """Messages from guilds in the allowlist should be accepted."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"discord_allowed_guilds": [12345]}
        )
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        resp = client.post(
            "/discord/webhook",
            json=_discord_payload(content="in allowed guild", guild_id="12345"),
        )
        assert resp.status_code == 200
        app.state.queue.enqueue.assert_awaited_once()

    # ---- /status command ----

    def test_status_command_no_id_sends_usage_hint(self, sync_client: tuple) -> None:
        """The /status command without a task ID should send usage instructions."""
        client, app = sync_client
        app.state.messenger.send_discord = AsyncMock()

        resp = client.post("/discord/webhook", json=_discord_payload(content="/status"))
        assert resp.status_code == 200
        app.state.messenger.send_discord.assert_awaited()

    def test_status_command_unknown_task(self, sync_client: tuple) -> None:
        """The /status command with an unknown ID should send a not-found message."""
        client, app = sync_client
        app.state.messenger.send_discord = AsyncMock()

        resp = client.post(
            "/discord/webhook",
            json=_discord_payload(content="/status nonexistent-id"),
        )
        assert resp.status_code == 200
        app.state.messenger.send_discord.assert_awaited()

    def test_status_command_existing_task_calls_send_status(self, sync_client: tuple) -> None:
        """The /status command for an existing task should call send_status."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        app.state.messenger.send_status = AsyncMock()
        app.state.messenger.send_discord = AsyncMock()

        # Create a task
        create_resp = client.post(
            "/discord/webhook",
            json=_discord_payload(content="Do work"),
        )
        task_id = create_resp.json().get("task_id", "")
        assert task_id

        # Query its status
        client.post(
            "/discord/webhook",
            json=_discord_payload(content=f"/status {task_id}"),
        )
        app.state.messenger.send_status.assert_awaited_once()

    # ---- /tasks command ----

    def test_tasks_command_empty_sends_message(self, sync_client: tuple) -> None:
        """The /tasks command with no tasks should send a response."""
        client, app = sync_client
        app.state.messenger.send_discord = AsyncMock()

        resp = client.post("/discord/webhook", json=_discord_payload(content="/tasks"))
        assert resp.status_code == 200
        app.state.messenger.send_discord.assert_awaited()

    # ---- /help command ----

    def test_help_command_sends_help_message(self, sync_client: tuple) -> None:
        """The /help command should send exactly one Discord message."""
        client, app = sync_client
        app.state.messenger.send_discord = AsyncMock()

        resp = client.post("/discord/webhook", json=_discord_payload(content="/help"))
        assert resp.status_code == 200
        app.state.messenger.send_discord.assert_awaited_once()

    def test_help_command_does_not_enqueue(self, sync_client: tuple) -> None:
        """The /help command must not enqueue an AI task."""
        client, app = sync_client
        app.state.messenger.send_discord = AsyncMock()
        app.state.queue.enqueue = AsyncMock()

        client.post("/discord/webhook", json=_discord_payload(content="/help"))
        app.state.queue.enqueue.assert_not_awaited()

    # ---- Task stored in DB ----

    def test_enqueued_task_stored_in_db(self, sync_client: tuple) -> None:
        """The task created from a Discord message must be persisted in the DB."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        resp = client.post(
            "/discord/webhook",
            json=_discord_payload(content="Discord task content", channel_id="chan42"),
        )
        task_id = resp.json()["task_id"]

        get_resp = client.get(f"/tasks/{task_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["platform"] == "discord"
        assert data["prompt"] == "Discord task content"
        assert data["chat_id"] == "chan42"


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------


class TestListTasksEndpoint:
    """Tests for GET /tasks."""

    def test_empty_database_returns_empty_list(self, sync_client: tuple) -> None:
        """No tasks in the DB should return an empty list with total=0."""
        client, _app = sync_client
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks"] == []
        assert data["total"] == 0

    def test_list_tasks_after_creation(self, sync_client: tuple) -> None:
        """After creating a task via Telegram, it should appear in the list."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update(text="List test"))

        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["tasks"]) >= 1

    def test_list_tasks_filter_by_status_pending(self, sync_client: tuple) -> None:
        """Filtering by status=pending should return only pending tasks."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update(text="Filter test"))

        resp = client.get("/tasks?status=pending")
        assert resp.status_code == 200
        data = resp.json()
        for task in data["tasks"]:
            assert task["status"] == "pending"

    def test_list_tasks_filter_by_platform_telegram(self, sync_client: tuple) -> None:
        """Filtering by platform=telegram should return only Telegram tasks."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update(text="TG task"))

        resp = client.get("/tasks?platform=telegram")
        assert resp.status_code == 200
        data = resp.json()
        for task in data["tasks"]:
            assert task["platform"] == "telegram"

    def test_list_tasks_invalid_status_returns_400(self, sync_client: tuple) -> None:
        """An invalid status filter value should return HTTP 400."""
        client, _app = sync_client
        resp = client.get("/tasks?status=invalid_status")
        assert resp.status_code == 400

    def test_list_tasks_invalid_platform_returns_400(self, sync_client: tuple) -> None:
        """An invalid platform filter value should return HTTP 400."""
        client, _app = sync_client
        resp = client.get("/tasks?platform=myspace")
        assert resp.status_code == 400

    def test_list_tasks_pagination_accepted(self, sync_client: tuple) -> None:
        """Pagination parameters limit and offset should be accepted."""
        client, _app = sync_client
        resp = client.get("/tasks?limit=10&offset=0")
        assert resp.status_code == 200

    def test_list_tasks_limit_capped(self, sync_client: tuple) -> None:
        """A limit greater than 200 should return HTTP 422 (validation error)."""
        client, _app = sync_client
        resp = client.get("/tasks?limit=9999")
        assert resp.status_code == 422

    def test_list_tasks_response_has_total_field(self, sync_client: tuple) -> None:
        """The response must include a 'total' integer field."""
        client, _app = sync_client
        data = client.get("/tasks").json()
        assert "total" in data
        assert isinstance(data["total"], int)

    def test_list_tasks_response_has_tasks_field(self, sync_client: tuple) -> None:
        """The response must include a 'tasks' list field."""
        client, _app = sync_client
        data = client.get("/tasks").json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}
# ---------------------------------------------------------------------------


class TestGetTaskEndpoint:
    """Tests for GET /tasks/{task_id}."""

    def test_get_nonexistent_task_returns_404(self, sync_client: tuple) -> None:
        """Requesting a non-existent task should return HTTP 404."""
        client, _app = sync_client
        resp = client.get("/tasks/nonexistent-uuid")
        assert resp.status_code == 404

    def test_get_existing_task_returns_200(self, sync_client: tuple) -> None:
        """Requesting an existing task should return HTTP 200."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Get test task")
        )
        task_id = create_resp.json()["task_id"]

        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200

    def test_get_existing_task_returns_correct_id(self, sync_client: tuple) -> None:
        """The returned task must have the correct ID."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="ID check")
        )
        task_id = create_resp.json()["task_id"]

        data = client.get(f"/tasks/{task_id}").json()
        assert data["id"] == task_id

    def test_get_existing_task_returns_correct_prompt(self, sync_client: tuple) -> None:
        """The returned task must have the correct prompt text."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook",
            json=_telegram_update(text="Unique prompt text here"),
        )
        task_id = create_resp.json()["task_id"]

        data = client.get(f"/tasks/{task_id}").json()
        assert data["prompt"] == "Unique prompt text here"

    def test_get_existing_task_returns_correct_platform(self, sync_client: tuple) -> None:
        """The returned task must have platform='telegram'."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Platform check")
        )
        task_id = create_resp.json()["task_id"]

        data = client.get(f"/tasks/{task_id}").json()
        assert data["platform"] == "telegram"

    def test_get_task_response_fields_present(self, sync_client: tuple) -> None:
        """The task response must include all required fields."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Field check")
        )
        task_id = create_resp.json()["task_id"]

        data = client.get(f"/tasks/{task_id}").json()
        required_fields = {
            "id", "platform", "chat_id", "user_id",
            "prompt", "status", "created_at", "updated_at",
        }
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# DELETE /tasks/{task_id}
# ---------------------------------------------------------------------------


class TestDeleteTaskEndpoint:
    """Tests for DELETE /tasks/{task_id}."""

    def test_delete_nonexistent_task_returns_404(self, sync_client: tuple) -> None:
        """Deleting a non-existent task should return HTTP 404."""
        client, _app = sync_client
        resp = client.delete("/tasks/nonexistent-uuid")
        assert resp.status_code == 404

    def test_delete_existing_task_returns_204(self, sync_client: tuple) -> None:
        """Deleting an existing task should return HTTP 204."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Delete me")
        )
        task_id = create_resp.json()["task_id"]

        resp = client.delete(f"/tasks/{task_id}")
        assert resp.status_code == 204

    def test_delete_task_removes_from_db(self, sync_client: tuple) -> None:
        """After deletion, GET /tasks/{id} should return 404."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="To be deleted")
        )
        task_id = create_resp.json()["task_id"]

        client.delete(f"/tasks/{task_id}")

        get_resp = client.get(f"/tasks/{task_id}")
        assert get_resp.status_code == 404

    def test_delete_task_reduces_total_count(self, sync_client: tuple) -> None:
        """Deleting a task should reduce the list total by one."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        client.post("/telegram/webhook", json=_telegram_update(text="Task A", update_id=1))
        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Task B", update_id=2)
        )
        task_id = create_resp.json()["task_id"]

        before = client.get("/tasks").json()["total"]
        client.delete(f"/tasks/{task_id}")
        after = client.get("/tasks").json()["total"]

        assert after == before - 1

    def test_delete_does_not_affect_other_tasks(self, sync_client: tuple) -> None:
        """Deleting one task must not remove other tasks."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        resp1 = client.post(
            "/telegram/webhook", json=_telegram_update(text="Keep me", update_id=1)
        )
        resp2 = client.post(
            "/telegram/webhook", json=_telegram_update(text="Delete me", update_id=2)
        )
        keep_id = resp1.json()["task_id"]
        del_id = resp2.json()["task_id"]

        client.delete(f"/tasks/{del_id}")

        assert client.get(f"/tasks/{keep_id}").status_code == 200
        assert client.get(f"/tasks/{del_id}").status_code == 404


# ---------------------------------------------------------------------------
# POST /tasks/{task_id}/deliver
# ---------------------------------------------------------------------------


class TestDeliverTaskEndpoint:
    """Tests for POST /tasks/{task_id}/deliver."""

    def test_deliver_nonexistent_task_returns_404(self, sync_client: tuple) -> None:
        """Delivering a non-existent task should return HTTP 404."""
        client, _app = sync_client
        resp = client.post("/tasks/nonexistent-uuid/deliver")
        assert resp.status_code == 404

    def test_deliver_pending_task_returns_409(self, sync_client: tuple) -> None:
        """Delivering a pending task should return HTTP 409 Conflict."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Pending deliver")
        )
        task_id = create_resp.json()["task_id"]

        resp = client.post(f"/tasks/{task_id}/deliver")
        assert resp.status_code == 409

    def test_deliver_running_task_returns_409(self, sync_client: tuple) -> None:
        """Delivering a running task should return HTTP 409 Conflict."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Running deliver")
        )
        task_id = create_resp.json()["task_id"]

        # Set to RUNNING
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            app.state.db.update_task_status(
                task_id, TaskStatusUpdate(status=TaskStatus.RUNNING)
            )
        )
        loop.close()

        resp = client.post(f"/tasks/{task_id}/deliver")
        assert resp.status_code == 409

    def test_deliver_done_task_returns_200(self, sync_client: tuple) -> None:
        """Delivering a done task should return HTTP 200."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        app.state.messenger.send = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Deliver done")
        )
        task_id = create_resp.json()["task_id"]
        _mark_task_done(app, task_id)

        resp = client.post(f"/tasks/{task_id}/deliver")
        assert resp.status_code == 200

    def test_deliver_done_task_returns_ok_true(self, sync_client: tuple) -> None:
        """Delivering a done task should return {ok: true}."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        app.state.messenger.send = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Done deliver ok")
        )
        task_id = create_resp.json()["task_id"]
        _mark_task_done(app, task_id)

        data = client.post(f"/tasks/{task_id}/deliver").json()
        assert data["ok"] is True

    def test_deliver_done_task_calls_messenger_send(self, sync_client: tuple) -> None:
        """Delivering a done task should call messenger.send exactly once."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        app.state.messenger.send = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Messenger send check")
        )
        task_id = create_resp.json()["task_id"]
        _mark_task_done(app, task_id)

        client.post(f"/tasks/{task_id}/deliver")
        app.state.messenger.send.assert_awaited_once()

    def test_deliver_failed_task_also_works(self, sync_client: tuple) -> None:
        """Delivering a failed task should also return HTTP 200."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        app.state.messenger.send = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Failed deliver")
        )
        task_id = create_resp.json()["task_id"]

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            app.state.db.update_task_status(
                task_id,
                TaskStatusUpdate(status=TaskStatus.FAILED, error="ai died"),
            )
        )
        loop.close()

        resp = client.post(f"/tasks/{task_id}/deliver")
        assert resp.status_code == 200

    def test_deliver_messenger_error_returns_502(self, sync_client: tuple) -> None:
        """A MessengerError during delivery should return HTTP 502."""
        from agent_bridge.messenger import MessengerError

        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        app.state.messenger.send = AsyncMock(
            side_effect=MessengerError("delivery failed")
        )

        create_resp = client.post(
            "/telegram/webhook", json=_telegram_update(text="Deliver fail test")
        )
        task_id = create_resp.json()["task_id"]
        _mark_task_done(app, task_id)

        resp = client.post(f"/tasks/{task_id}/deliver")
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /tasks/{task_id}/callback
# ---------------------------------------------------------------------------


class TestTriggerCallbackEndpoint:
    """Tests for POST /tasks/{task_id}/callback."""

    def test_no_callback_url_returns_503(self, sync_client: tuple) -> None:
        """Without a CALLBACK_URL, the endpoint should return HTTP 503."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": None}
        )
        resp = client.post("/tasks/some-id/callback")
        assert resp.status_code == 503

    def test_task_not_found_returns_404(self, sync_client: tuple) -> None:
        """A non-existent task with a configured CALLBACK_URL should return 404."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": "http://callback.example.com/done"}
        )
        resp = client.post("/tasks/nonexistent-uuid/callback")
        assert resp.status_code == 404

    def test_pending_task_returns_409(self, sync_client: tuple) -> None:
        """Triggering a callback for a pending task should return HTTP 409."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": "http://callback.example.com/done"}
        )
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook",
            json=_telegram_update(text="Pending callback test"),
        )
        task_id = create_resp.json()["task_id"]

        resp = client.post(f"/tasks/{task_id}/callback")
        assert resp.status_code == 409

    def test_done_task_posts_to_callback_url(self, sync_client: tuple) -> None:
        """Triggering a callback for a done task should POST to the callback URL."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": "http://callback.example.com/done"}
        )
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook",
            json=_telegram_update(text="Callback trigger test"),
        )
        task_id = create_resp.json()["task_id"]
        _mark_task_done(app, task_id, result="callback ok")

        with respx.mock:
            callback_route = respx.post("http://callback.example.com/done").mock(
                return_value=httpx.Response(200, json={"ok": True})
            )
            resp = client.post(f"/tasks/{task_id}/callback")

        assert resp.status_code == 200
        assert callback_route.called

    def test_done_task_callback_returns_ok_true(self, sync_client: tuple) -> None:
        """A successful callback trigger should return {ok: true}."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": "http://callback.example.com/done"}
        )
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook",
            json=_telegram_update(text="Callback ok test"),
        )
        task_id = create_resp.json()["task_id"]
        _mark_task_done(app, task_id)

        with respx.mock:
            respx.post("http://callback.example.com/done").mock(
                return_value=httpx.Response(200, json={"ok": True})
            )
            data = client.post(f"/tasks/{task_id}/callback").json()

        assert data["ok"] is True
        assert data["task_id"] == task_id

    def test_failed_task_also_triggers_callback(self, sync_client: tuple) -> None:
        """Triggering a callback for a failed task should also work."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": "http://callback.example.com/done"}
        )
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        create_resp = client.post(
            "/telegram/webhook",
            json=_telegram_update(text="Failed callback test"),
        )
        task_id = create_resp.json()["task_id"]

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            app.state.db.update_task_status(
                task_id,
                TaskStatusUpdate(status=TaskStatus.FAILED, error="ai failed"),
            )
        )
        loop.close()

        with respx.mock:
            callback_route = respx.post("http://callback.example.com/done").mock(
                return_value=httpx.Response(200, json={"ok": True})
            )
            resp = client.post(f"/tasks/{task_id}/callback")

        assert resp.status_code == 200
        assert callback_route.called


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """create_app() must return a FastAPI application instance."""
        from fastapi import FastAPI

        from agent_bridge.app import create_app

        with patch.dict(
            os.environ,
            {"AGENT_BRIDGE_TESTING": "1", "DATABASE_URL": ":memory:"},
        ):
            get_settings.cache_clear()
            _app = create_app(testing=True)
        assert isinstance(_app, FastAPI)
        get_settings.cache_clear()

    def test_create_app_has_health_route(self) -> None:
        """The created app must expose a /health route."""
        from agent_bridge.app import create_app

        with patch.dict(
            os.environ,
            {"AGENT_BRIDGE_TESTING": "1", "DATABASE_URL": ":memory:"},
        ):
            get_settings.cache_clear()
            _app = create_app(testing=True)

        routes = [r.path for r in _app.routes]
        assert "/health" in routes
        get_settings.cache_clear()

    def test_create_app_has_telegram_webhook_route(self) -> None:
        """The created app must expose a /telegram/webhook route."""
        from agent_bridge.app import create_app

        with patch.dict(
            os.environ,
            {"AGENT_BRIDGE_TESTING": "1", "DATABASE_URL": ":memory:"},
        ):
            get_settings.cache_clear()
            _app = create_app(testing=True)

        routes = [r.path for r in _app.routes]
        assert "/telegram/webhook" in routes
        get_settings.cache_clear()

    def test_create_app_has_discord_webhook_route(self) -> None:
        """The created app must expose a /discord/webhook route."""
        from agent_bridge.app import create_app

        with patch.dict(
            os.environ,
            {"AGENT_BRIDGE_TESTING": "1", "DATABASE_URL": ":memory:"},
        ):
            get_settings.cache_clear()
            _app = create_app(testing=True)

        routes = [r.path for r in _app.routes]
        assert "/discord/webhook" in routes
        get_settings.cache_clear()

    def test_create_app_has_tasks_route(self) -> None:
        """The created app must expose a /tasks route."""
        from agent_bridge.app import create_app

        with patch.dict(
            os.environ,
            {"AGENT_BRIDGE_TESTING": "1", "DATABASE_URL": ":memory:"},
        ):
            get_settings.cache_clear()
            _app = create_app(testing=True)

        routes = [r.path for r in _app.routes]
        assert "/tasks" in routes
        get_settings.cache_clear()

    def test_lifespan_initialises_state_components(self, sync_client: tuple) -> None:
        """After startup, app.state must have db, queue, messenger, and settings."""
        _client, app = sync_client
        assert hasattr(app.state, "db")
        assert hasattr(app.state, "queue")
        assert hasattr(app.state, "messenger")
        assert hasattr(app.state, "settings")

    def test_lifespan_db_is_database_instance(self, sync_client: tuple) -> None:
        """app.state.db must be an instance of Database."""
        _client, app = sync_client
        assert isinstance(app.state.db, Database)

    def test_lifespan_queue_is_task_queue_instance(self, sync_client: tuple) -> None:
        """app.state.queue must be an instance of TaskQueue."""
        _client, app = sync_client
        assert isinstance(app.state.queue, TaskQueue)

    def test_lifespan_messenger_is_messenger_instance(self, sync_client: tuple) -> None:
        """app.state.messenger must be an instance of Messenger."""
        _client, app = sync_client
        assert isinstance(app.state.messenger, Messenger)

    def test_lifespan_queue_is_running(self, sync_client: tuple) -> None:
        """The task queue must be running after application startup."""
        _client, app = sync_client
        assert app.state.queue.is_running is True

    def test_create_app_with_custom_settings(self) -> None:
        """create_app() should accept a pre-built Settings instance."""
        from fastapi import FastAPI

        from agent_bridge.app import create_app

        with patch.dict(
            os.environ,
            {"AGENT_BRIDGE_TESTING": "1", "DATABASE_URL": ":memory:"},
        ):
            get_settings.cache_clear()
            settings = Settings()
            _app = create_app(settings=settings, testing=True)

        assert isinstance(_app, FastAPI)
        get_settings.cache_clear()
