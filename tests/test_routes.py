"""Integration tests for agent_bridge HTTP routes using FastAPI TestClient.

Tests cover:
- Health check endpoint
- Telegram webhook ingestion (valid, invalid user, no text, commands)
- Discord webhook ingestion (valid, invalid user, bot author, commands)
- Task REST API (list, get, delete)
- Manual deliver and callback trigger endpoints
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

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
# Settings fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
        },
    ):
        get_settings.cache_clear()
        yield Settings()


# ---------------------------------------------------------------------------
# App fixture with mocked dependencies
# ---------------------------------------------------------------------------


@pytest.fixture
async def app_client(test_settings: Settings):
    """Provide a TestClient backed by an in-memory DB and mocked queue/messenger."""
    from agent_bridge.app import create_app

    # Use in-memory DB for tests
    test_settings_copy = test_settings.model_copy(
        update={"database_url": ":memory:"}
    )

    with patch.object(
        test_settings_copy,
        "database_url",
        ":memory:",
        create=True,
    ):
        pass  # model_copy already applied

    _app = create_app(settings=test_settings_copy, testing=True)

    async with AsyncClient(app=_app, base_url="http://test") as client:
        # The lifespan has set up app.state; yield both
        yield client, _app


@pytest.fixture
def sync_client(test_settings: Settings):
    """Synchronous TestClient for simpler non-async tests."""
    from agent_bridge.app import create_app

    # Override database URL to in-memory
    env_overrides = {
        "DATABASE_URL": ":memory:",
        "AGENT_BRIDGE_TESTING": "1",
        "TELEGRAM_BOT_TOKEN": "123456:TestToken",
        "DISCORD_BOT_TOKEN": "discord-test-token",
        "AI_BACKEND_TYPE": "ollama",
        "AI_BACKEND_URL": "http://localhost:11434",
        "AI_MODEL": "llama3",
        "SECRET_TOKEN": "test-secret",
        "MAX_CONCURRENT_TASKS": "2",
        "TASK_TIMEOUT_SECONDS": "30",
    }
    get_settings.cache_clear()
    with patch.dict(os.environ, env_overrides):
        get_settings.cache_clear()
        settings = Settings()
        _app = create_app(settings=settings, testing=True)
        with TestClient(_app, raise_server_exceptions=True) as client:
            yield client, _app
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Helper functions
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


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health_returns_ok(self, sync_client):
        client, _app = sync_client
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_version_matches_package(self, sync_client):
        from agent_bridge import __version__

        client, _app = sync_client
        resp = client.get("/health")
        assert resp.json()["version"] == __version__


# ---------------------------------------------------------------------------
# Telegram webhook
# ---------------------------------------------------------------------------


class TestTelegramWebhook:
    def test_valid_message_enqueues_task(self, sync_client):
        """A valid Telegram message should return ok and a task_id."""
        client, app = sync_client

        # Mock the queue's enqueue method
        app.state.queue.enqueue = AsyncMock()
        # Mock messenger ack
        app.state.messenger.send_ack = AsyncMock()

        payload = _telegram_update(text="Write a sorting algorithm")
        resp = client.post("/telegram/webhook", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "task_id" in data
        app.state.queue.enqueue.assert_awaited_once()

    def test_no_message_returns_ok_ignored(self, sync_client):
        """An update with no message should be silently ignored."""
        client, app = sync_client
        payload = {"update_id": 99}  # no 'message'
        resp = client.post("/telegram/webhook", json=payload)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json().get("ignored") is True

    def test_empty_text_is_ignored(self, sync_client):
        """A message with empty text should be ignored."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        payload = _telegram_update(text="")
        resp = client.post("/telegram/webhook", json=payload)
        assert resp.status_code == 200
        app.state.queue.enqueue.assert_not_awaited()

    def test_disallowed_user_rejected(self, sync_client):
        """A user not in the allowlist should receive a not-authorised response."""
        client, app = sync_client

        # Configure an allowlist that excludes our test user (id=42)
        app.state.settings = app.state.settings.model_copy(
            update={"telegram_allowed_users": [999]}
        )
        app.state.messenger.send_not_authorised = AsyncMock()
        app.state.queue.enqueue = AsyncMock()

        payload = _telegram_update(user_id=42, text="hello")
        resp = client.post("/telegram/webhook", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data.get("ignored") is True
        app.state.queue.enqueue.assert_not_awaited()

    def test_status_command(self, sync_client):
        """The /status command should reply with task info or not-found."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        payload = _telegram_update(text="/status nonexistent-task-id")
        resp = client.post("/telegram/webhook", json=payload)

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # Should attempt to send a 'not found' message
        app.state.messenger.send_telegram.assert_awaited()

    def test_status_command_no_id(self, sync_client):
        """The /status command without an ID should send usage hint."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        payload = _telegram_update(text="/status")
        resp = client.post("/telegram/webhook", json=payload)

        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited()
        call_text = app.state.messenger.send_telegram.call_args.kwargs.get("text", "")
        assert "usage" in call_text.lower() or "/status" in call_text

    def test_tasks_command_empty(self, sync_client):
        """The /tasks command with no tasks should send an empty message."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        payload = _telegram_update(text="/tasks")
        resp = client.post("/telegram/webhook", json=payload)

        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited()

    def test_help_command(self, sync_client):
        """The /help command should send a help message."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        payload = _telegram_update(text="/help")
        resp = client.post("/telegram/webhook", json=payload)

        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited_once()

    def test_start_command(self, sync_client):
        """The /start command should send a help message."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()

        payload = _telegram_update(text="/start")
        resp = client.post("/telegram/webhook", json=payload)

        assert resp.status_code == 200
        app.state.messenger.send_telegram.assert_awaited_once()

    def test_send_ack_called_after_enqueue(self, sync_client):
        """An ack message should be sent to the user after enqueueing."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        payload = _telegram_update(text="Write a test")
        client.post("/telegram/webhook", json=payload)

        app.state.messenger.send_ack.assert_awaited_once()
        call_kwargs = app.state.messenger.send_ack.call_args.kwargs
        assert call_kwargs.get("platform") == Platform.TELEGRAM
        assert call_kwargs.get("chat_id") == "100"

    def test_status_command_with_existing_task(self, sync_client):
        """The /status command for an existing task should call send_status."""
        client, app = sync_client
        app.state.messenger.send_telegram = AsyncMock()
        app.state.messenger.send_status = AsyncMock()

        # First create a task
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        enqueue_payload = _telegram_update(text="Write something", update_id=1)
        resp = client.post("/telegram/webhook", json=enqueue_payload)
        task_id = resp.json().get("task_id", "")

        if task_id:
            status_payload = _telegram_update(text=f"/status {task_id}", update_id=2)
            resp2 = client.post("/telegram/webhook", json=status_payload)
            assert resp2.status_code == 200
            app.state.messenger.send_status.assert_awaited_once()


# ---------------------------------------------------------------------------
# Discord webhook
# ---------------------------------------------------------------------------


class TestDiscordWebhook:
    def test_valid_message_enqueues_task(self, sync_client):
        """A valid Discord message should return ok and a task_id."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        payload = _discord_payload(content="Write a sorting algorithm")
        resp = client.post("/discord/webhook", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "task_id" in data
        app.state.queue.enqueue.assert_awaited_once()

    def test_bot_author_ignored(self, sync_client):
        """Messages from bots should be silently ignored."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()

        payload = _discord_payload(is_bot=True)
        resp = client.post("/discord/webhook", json=payload)

        assert resp.status_code == 200
        assert resp.json().get("ignored") is True
        app.state.queue.enqueue.assert_not_awaited()

    def test_empty_content_ignored(self, sync_client):
        """Messages with empty content should be ignored."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()

        payload = _discord_payload(content="")
        resp = client.post("/discord/webhook", json=payload)

        assert resp.status_code == 200
        assert resp.json().get("ignored") is True
        app.state.queue.enqueue.assert_not_awaited()

    def test_disallowed_user_rejected(self, sync_client):
        """Discord users not in the allowlist should be rejected."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"discord_allowed_users": [999999]}
        )
        app.state.messenger.send_not_authorised = AsyncMock()
        app.state.queue.enqueue = AsyncMock()

        payload = _discord_payload(user_id="333", content="hello")  # user 333 not allowed
        resp = client.post("/discord/webhook", json=payload)

        assert resp.status_code == 200
        assert resp.json().get("ignored") is True
        app.state.queue.enqueue.assert_not_awaited()

    def test_disallowed_guild_rejected(self, sync_client):
        """Messages from guilds not in the allowlist should be rejected."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"discord_allowed_guilds": [111111]}
        )
        app.state.queue.enqueue = AsyncMock()

        payload = _discord_payload(content="hello", guild_id="999999")  # not allowed
        resp = client.post("/discord/webhook", json=payload)

        assert resp.status_code == 200
        assert resp.json().get("ignored") is True
        app.state.queue.enqueue.assert_not_awaited()

    def test_status_command(self, sync_client):
        """The /status command should look up the task."""
        client, app = sync_client
        app.state.messenger.send_discord = AsyncMock()

        payload = _discord_payload(content="/status nonexistent-id")
        resp = client.post("/discord/webhook", json=payload)

        assert resp.status_code == 200
        app.state.messenger.send_discord.assert_awaited()

    def test_tasks_command(self, sync_client):
        """The /tasks command should respond with the recent task list."""
        client, app = sync_client
        app.state.messenger.send_discord = AsyncMock()

        payload = _discord_payload(content="/tasks")
        resp = client.post("/discord/webhook", json=payload)

        assert resp.status_code == 200
        app.state.messenger.send_discord.assert_awaited()

    def test_help_command(self, sync_client):
        """The /help command should send a help message."""
        client, app = sync_client
        app.state.messenger.send_discord = AsyncMock()

        payload = _discord_payload(content="/help")
        resp = client.post("/discord/webhook", json=payload)

        assert resp.status_code == 200
        app.state.messenger.send_discord.assert_awaited_once()

    def test_ack_sent_after_enqueue(self, sync_client):
        """An ack message should be sent after enqueueing a Discord task."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        payload = _discord_payload(content="Do something")
        client.post("/discord/webhook", json=payload)

        app.state.messenger.send_ack.assert_awaited_once()
        call_kwargs = app.state.messenger.send_ack.call_args.kwargs
        assert call_kwargs.get("platform") == Platform.DISCORD


# ---------------------------------------------------------------------------
# Task list endpoint
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_list_tasks_empty(self, sync_client):
        """GET /tasks should return an empty list when no tasks exist."""
        client, app = sync_client
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tasks"] == []
        assert data["total"] == 0

    def test_list_tasks_after_creation(self, sync_client):
        """After a Telegram message is ingested, it should appear in the task list."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        # Create a task via the Telegram webhook
        payload = _telegram_update(text="List this task")
        client.post("/telegram/webhook", json=payload)

        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["tasks"]) >= 1

    def test_list_tasks_filter_by_status(self, sync_client):
        """GET /tasks?status=pending should only return pending tasks."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        payload = _telegram_update(text="Filter test")
        client.post("/telegram/webhook", json=payload)

        resp = client.get("/tasks?status=pending")
        assert resp.status_code == 200
        data = resp.json()
        for task in data["tasks"]:
            assert task["status"] == "pending"

    def test_list_tasks_invalid_status_returns_400(self, sync_client):
        """An invalid status filter should return HTTP 400."""
        client, _app = sync_client
        resp = client.get("/tasks?status=invalid_status")
        assert resp.status_code == 400

    def test_list_tasks_invalid_platform_returns_400(self, sync_client):
        """An invalid platform filter should return HTTP 400."""
        client, _app = sync_client
        resp = client.get("/tasks?platform=myspace")
        assert resp.status_code == 400

    def test_list_tasks_pagination(self, sync_client):
        """Pagination parameters should be accepted."""
        client, _app = sync_client
        resp = client.get("/tasks?limit=10&offset=0")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Get single task endpoint
# ---------------------------------------------------------------------------


class TestGetTask:
    def test_get_task_not_found(self, sync_client):
        """GET /tasks/<nonexistent> should return 404."""
        client, _app = sync_client
        resp = client.get("/tasks/nonexistent-uuid")
        assert resp.status_code == 404

    def test_get_task_found(self, sync_client):
        """GET /tasks/<id> should return the task record."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        # Create via webhook
        payload = _telegram_update(text="Get this task")
        create_resp = client.post("/telegram/webhook", json=payload)
        task_id = create_resp.json().get("task_id")
        assert task_id is not None

        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == task_id
        assert data["prompt"] == "Get this task"
        assert data["platform"] == "telegram"


# ---------------------------------------------------------------------------
# Delete task endpoint
# ---------------------------------------------------------------------------


class TestDeleteTask:
    def test_delete_task_not_found(self, sync_client):
        """DELETE /tasks/<nonexistent> should return 404."""
        client, _app = sync_client
        resp = client.delete("/tasks/nonexistent-uuid")
        assert resp.status_code == 404

    def test_delete_task_success(self, sync_client):
        """DELETE /tasks/<id> should return 204 and remove the task."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        payload = _telegram_update(text="Delete me")
        create_resp = client.post("/telegram/webhook", json=payload)
        task_id = create_resp.json().get("task_id")
        assert task_id is not None

        del_resp = client.delete(f"/tasks/{task_id}")
        assert del_resp.status_code == 204

        get_resp = client.get(f"/tasks/{task_id}")
        assert get_resp.status_code == 404


# ---------------------------------------------------------------------------
# Manual deliver endpoint
# ---------------------------------------------------------------------------


class TestDeliverTask:
    def test_deliver_task_not_found(self, sync_client):
        """POST /tasks/<nonexistent>/deliver should return 404."""
        client, _app = sync_client
        resp = client.post("/tasks/nonexistent-uuid/deliver")
        assert resp.status_code == 404

    def test_deliver_pending_task_returns_409(self, sync_client):
        """Attempting to deliver a pending task should return 409."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        payload = _telegram_update(text="pending task")
        create_resp = client.post("/telegram/webhook", json=payload)
        task_id = create_resp.json().get("task_id")
        assert task_id is not None

        resp = client.post(f"/tasks/{task_id}/deliver")
        assert resp.status_code == 409

    def test_deliver_done_task_success(self, sync_client):
        """Delivering a done task should call messenger.send and return ok."""
        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        app.state.messenger.send = AsyncMock()

        # Create and manually mark done
        payload = _telegram_update(text="complete me")
        create_resp = client.post("/telegram/webhook", json=payload)
        task_id = create_resp.json().get("task_id")
        assert task_id is not None

        # Directly update DB status to DONE
        import asyncio

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            app.state.db.update_task_status(
                task_id,
                __import__("agent_bridge.models", fromlist=["TaskStatusUpdate"]).TaskStatusUpdate(
                    status=TaskStatus.DONE, result="Great result"
                ),
            )
        )
        loop.close()

        resp = client.post(f"/tasks/{task_id}/deliver")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        app.state.messenger.send.assert_awaited_once()

    def test_deliver_messenger_error_returns_502(self, sync_client):
        """A messenger error during delivery should return 502."""
        from agent_bridge.messenger import MessengerError

        client, app = sync_client
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()
        app.state.messenger.send = AsyncMock(
            side_effect=MessengerError("delivery failed")
        )

        payload = _telegram_update(text="deliver fail test")
        create_resp = client.post("/telegram/webhook", json=payload)
        task_id = create_resp.json().get("task_id")
        assert task_id is not None

        import asyncio

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            app.state.db.update_task_status(
                task_id,
                __import__("agent_bridge.models", fromlist=["TaskStatusUpdate"]).TaskStatusUpdate(
                    status=TaskStatus.DONE, result="ok"
                ),
            )
        )
        loop.close()

        resp = client.post(f"/tasks/{task_id}/deliver")
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Callback trigger endpoint
# ---------------------------------------------------------------------------


class TestTriggerCallback:
    def test_trigger_callback_no_url_configured(self, sync_client):
        """Without a CALLBACK_URL configured, endpoint should return 503."""
        client, app = sync_client
        # Ensure no callback URL
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": None}
        )
        resp = client.post("/tasks/some-id/callback")
        assert resp.status_code == 503

    def test_trigger_callback_task_not_found(self, sync_client):
        """404 if task doesn't exist and callback URL is configured."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": "http://callback.example.com/done"}
        )
        resp = client.post("/tasks/nonexistent-uuid/callback")
        assert resp.status_code == 404

    def test_trigger_callback_pending_task_returns_409(self, sync_client):
        """A pending task should return 409 for callback trigger."""
        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": "http://callback.example.com/done"}
        )
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        payload = _telegram_update(text="pending callback test")
        create_resp = client.post("/telegram/webhook", json=payload)
        task_id = create_resp.json().get("task_id")
        assert task_id is not None

        resp = client.post(f"/tasks/{task_id}/callback")
        assert resp.status_code == 409

    def test_trigger_callback_done_task_success(self, sync_client):
        """Triggering a callback for a done task should POST to the callback URL."""
        import asyncio

        import respx
        import httpx

        client, app = sync_client
        app.state.settings = app.state.settings.model_copy(
            update={"callback_url": "http://callback.example.com/done"}
        )
        app.state.queue.enqueue = AsyncMock()
        app.state.messenger.send_ack = AsyncMock()

        payload = _telegram_update(text="callback trigger test")
        create_resp = client.post("/telegram/webhook", json=payload)
        task_id = create_resp.json().get("task_id")
        assert task_id is not None

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            app.state.db.update_task_status(
                task_id,
                __import__("agent_bridge.models", fromlist=["TaskStatusUpdate"]).TaskStatusUpdate(
                    status=TaskStatus.DONE, result="callback ok"
                ),
            )
        )
        loop.close()

        with respx.mock:
            callback_route = respx.post("http://callback.example.com/done").mock(
                return_value=httpx.Response(200, json={"ok": True})
            )
            resp = client.post(f"/tasks/{task_id}/callback")

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert callback_route.called


# ---------------------------------------------------------------------------
# Application factory tests
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_create_app_returns_fastapi_instance(self):
        """create_app() should return a FastAPI application."""
        from fastapi import FastAPI

        from agent_bridge.app import create_app

        with patch.dict(
            os.environ,
            {
                "AGENT_BRIDGE_TESTING": "1",
                "DATABASE_URL": ":memory:",
            },
        ):
            get_settings.cache_clear()
            _app = create_app(testing=True)
            assert isinstance(_app, FastAPI)
            get_settings.cache_clear()

    def test_create_app_includes_health_route(self):
        """The created app should have a /health route registered."""
        from agent_bridge.app import create_app

        with patch.dict(
            os.environ,
            {
                "AGENT_BRIDGE_TESTING": "1",
                "DATABASE_URL": ":memory:",
            },
        ):
            get_settings.cache_clear()
            _app = create_app(testing=True)

        routes = [r.path for r in _app.routes]
        assert "/health" in routes
        get_settings.cache_clear()

    def test_create_app_has_telegram_and_discord_routes(self):
        """The app should expose both /telegram/webhook and /discord/webhook."""
        from agent_bridge.app import create_app

        with patch.dict(
            os.environ,
            {
                "AGENT_BRIDGE_TESTING": "1",
                "DATABASE_URL": ":memory:",
            },
        ):
            get_settings.cache_clear()
            _app = create_app(testing=True)

        routes = [r.path for r in _app.routes]
        assert "/telegram/webhook" in routes
        assert "/discord/webhook" in routes
        get_settings.cache_clear()

    def test_lifespan_initialises_state(self, sync_client):
        """After startup, app.state should have db, queue, messenger, settings."""
        _client, app = sync_client
        assert hasattr(app.state, "db")
        assert hasattr(app.state, "queue")
        assert hasattr(app.state, "messenger")
        assert hasattr(app.state, "settings")
