"""Tests for agent_bridge.models – Pydantic model validation and helpers."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from agent_bridge.models import (
    CallbackPayload,
    DiscordInboundPayload,
    DiscordMessage,
    DiscordUser,
    ErrorResponse,
    HealthResponse,
    Platform,
    TaskCreate,
    TaskCreatedResponse,
    TaskListResponse,
    TaskRecord,
    TaskResponse,
    TaskStatus,
    TaskStatusUpdate,
    TelegramChat,
    TelegramMessage,
    TelegramUpdate,
    TelegramUser,
)


# ---------------------------------------------------------------------------
# TaskStatus enum
# ---------------------------------------------------------------------------


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.DONE.value == "done"
        assert TaskStatus.FAILED.value == "failed"

    def test_from_string(self):
        assert TaskStatus("pending") is TaskStatus.PENDING
        assert TaskStatus("done") is TaskStatus.DONE


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


class TestPlatform:
    def test_values(self):
        assert Platform.TELEGRAM.value == "telegram"
        assert Platform.DISCORD.value == "discord"


# ---------------------------------------------------------------------------
# Telegram models
# ---------------------------------------------------------------------------


class TestTelegramUser:
    def test_minimal(self):
        user = TelegramUser(id=12345, first_name="Alice")
        assert user.id == 12345
        assert user.is_bot is False

    def test_full(self):
        user = TelegramUser(
            id=99,
            is_bot=False,
            first_name="Bob",
            last_name="Smith",
            username="bobsmith",
            language_code="en",
        )
        assert user.username == "bobsmith"
        assert user.language_code == "en"


class TestTelegramUpdate:
    def _make_update(self, text: str = "Hello") -> TelegramUpdate:
        return TelegramUpdate(
            update_id=1,
            message=TelegramMessage(
                message_id=10,
                **{"from": {"id": 42, "first_name": "Alice", "is_bot": False}},
                chat=TelegramChat(id=100, type="private"),
                date=1700000000,
                text=text,
            ),
        )

    def test_sender_id(self):
        update = self._make_update()
        assert update.sender_id == 42

    def test_chat_id(self):
        update = self._make_update()
        assert update.chat_id == 100

    def test_text(self):
        update = self._make_update("Write a function")
        assert update.text == "Write a function"

    def test_effective_message_prefers_message(self):
        update = self._make_update()
        assert update.effective_message is update.message

    def test_no_message_returns_none(self):
        update = TelegramUpdate(update_id=5)
        assert update.effective_message is None
        assert update.sender_id is None
        assert update.chat_id is None
        assert update.text is None

    def test_edited_message_fallback(self):
        edited = TelegramMessage(
            message_id=20,
            chat=TelegramChat(id=200, type="private"),
            date=1700000100,
            text="edited text",
        )
        update = TelegramUpdate(update_id=6, edited_message=edited)
        assert update.effective_message is edited
        assert update.text == "edited text"


# ---------------------------------------------------------------------------
# Discord models
# ---------------------------------------------------------------------------


class TestDiscordUser:
    def test_id_int_property(self):
        user = DiscordUser(
            id="123456789012345678",
            username="devuser",
        )
        assert user.id_int == 123456789012345678

    def test_defaults(self):
        user = DiscordUser(id="1", username="u")
        assert user.discriminator == "0"
        assert user.bot is False
        assert user.global_name is None


class TestDiscordInboundPayload:
    def test_round_trip(self):
        payload = DiscordInboundPayload(
            message=DiscordMessage(
                id="111",
                channel_id="222",
                author=DiscordUser(id="333", username="alice"),
                content="do something",
                timestamp="2024-01-01T00:00:00Z",
            )
        )
        assert payload.message.content == "do something"
        assert payload.message.author.username == "alice"


# ---------------------------------------------------------------------------
# TaskCreate
# ---------------------------------------------------------------------------


class TestTaskCreate:
    def test_valid(self):
        tc = TaskCreate(
            platform=Platform.TELEGRAM,
            chat_id="100",
            user_id="42",
            prompt="Write a hello world function",
        )
        assert tc.prompt == "Write a hello world function"

    def test_prompt_stripped(self):
        tc = TaskCreate(
            platform=Platform.DISCORD,
            chat_id="chan1",
            user_id="usr1",
            prompt="  refactor this code  ",
        )
        assert tc.prompt == "refactor this code"

    def test_blank_prompt_raises(self):
        with pytest.raises(ValidationError, match="blank"):
            TaskCreate(
                platform=Platform.TELEGRAM,
                chat_id="100",
                user_id="42",
                prompt="   ",
            )

    def test_empty_prompt_raises(self):
        with pytest.raises(ValidationError):
            TaskCreate(
                platform=Platform.TELEGRAM,
                chat_id="100",
                user_id="42",
                prompt="",
            )


# ---------------------------------------------------------------------------
# TaskRecord
# ---------------------------------------------------------------------------


class TestTaskRecord:
    def test_defaults(self):
        record = TaskRecord(
            id="abc-123",
            platform=Platform.TELEGRAM,
            chat_id="100",
            user_id="42",
            prompt="hello",
        )
        assert record.status == TaskStatus.PENDING
        assert record.result is None
        assert record.error is None
        assert isinstance(record.created_at, datetime)
        assert isinstance(record.updated_at, datetime)


# ---------------------------------------------------------------------------
# TaskResponse
# ---------------------------------------------------------------------------


class TestTaskResponse:
    def test_from_record(self):
        record = TaskRecord(
            id="xyz-456",
            platform=Platform.DISCORD,
            chat_id="chan99",
            user_id="usr99",
            prompt="generate tests",
            status=TaskStatus.DONE,
            result="here are the tests",
        )
        resp = TaskResponse.from_record(record)
        assert resp.id == "xyz-456"
        assert resp.platform == "discord"
        assert resp.status == "done"
        assert resp.result == "here are the tests"

    def test_from_record_failed(self):
        record = TaskRecord(
            id="fail-1",
            platform=Platform.TELEGRAM,
            chat_id="100",
            user_id="42",
            prompt="bad prompt",
            status=TaskStatus.FAILED,
            error="timeout",
        )
        resp = TaskResponse.from_record(record)
        assert resp.status == "failed"
        assert resp.error == "timeout"
        assert resp.result is None


# ---------------------------------------------------------------------------
# TaskListResponse
# ---------------------------------------------------------------------------


class TestTaskListResponse:
    def test_empty(self):
        resp = TaskListResponse(tasks=[], total=0)
        assert resp.total == 0
        assert resp.tasks == []


# ---------------------------------------------------------------------------
# CallbackPayload
# ---------------------------------------------------------------------------


class TestCallbackPayload:
    def test_from_record(self):
        record = TaskRecord(
            id="cb-1",
            platform=Platform.TELEGRAM,
            chat_id="100",
            user_id="42",
            prompt="fix the bug",
            status=TaskStatus.DONE,
            result="bug fixed",
        )
        payload = CallbackPayload.from_record(record)
        assert payload.task_id == "cb-1"
        assert payload.platform == "telegram"
        assert payload.status == "done"
        assert payload.result == "bug fixed"
        assert payload.error is None

    def test_from_record_failed(self):
        record = TaskRecord(
            id="cb-2",
            platform=Platform.DISCORD,
            chat_id="chan",
            user_id="usr",
            prompt="hard task",
            status=TaskStatus.FAILED,
            error="AI backend unavailable",
        )
        payload = CallbackPayload.from_record(record)
        assert payload.status == "failed"
        assert payload.error == "AI backend unavailable"


# ---------------------------------------------------------------------------
# ErrorResponse and HealthResponse
# ---------------------------------------------------------------------------


class TestErrorResponse:
    def test_fields(self):
        err = ErrorResponse(error="not_found", detail="Task does not exist")
        assert err.error == "not_found"
        assert err.detail == "Task does not exist"


class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse(version="0.1.0")
        assert h.status == "ok"
        assert h.version == "0.1.0"
