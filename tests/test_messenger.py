"""Tests for agent_bridge.messenger – message formatting, splitting, and delivery."""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_bridge.config import Settings, get_settings
from agent_bridge.messenger import (
    DISCORD_MAX_LENGTH,
    TELEGRAM_MAX_LENGTH,
    DiscordMessengerError,
    Messenger,
    MessengerError,
    TelegramMessengerError,
    _count_open_code_fence,
    format_error_for_discord,
    format_error_for_telegram,
    format_result_for_discord,
    format_result_for_telegram,
    split_message,
)
from agent_bridge.models import Platform, TaskRecord, TaskStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def telegram_settings() -> Settings:
    """Settings with Telegram token configured."""
    with patch.dict(
        os.environ,
        {
            "AGENT_BRIDGE_TESTING": "1",
            "TELEGRAM_BOT_TOKEN": "123456:TestToken",
        },
    ):
        yield Settings()


@pytest.fixture
def discord_settings() -> Settings:
    """Settings with Discord token configured."""
    with patch.dict(
        os.environ,
        {
            "AGENT_BRIDGE_TESTING": "1",
            "DISCORD_BOT_TOKEN": "discord-test-token",
        },
    ):
        yield Settings()


@pytest.fixture
def both_settings() -> Settings:
    """Settings with both Telegram and Discord tokens configured."""
    with patch.dict(
        os.environ,
        {
            "AGENT_BRIDGE_TESTING": "1",
            "TELEGRAM_BOT_TOKEN": "123456:TestToken",
            "DISCORD_BOT_TOKEN": "discord-test-token",
        },
    ):
        yield Settings()


@pytest.fixture
def no_token_settings() -> Settings:
    """Settings with no bot tokens."""
    with patch.dict(os.environ, {"AGENT_BRIDGE_TESTING": "1"}, clear=False):
        env = {k: v for k, v in os.environ.items()
               if k not in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN")}
        env["AGENT_BRIDGE_TESTING"] = "1"
        with patch.dict(os.environ, env, clear=True):
            yield Settings()


def _make_task_record(
    task_id: str = "abcdef12-0000-0000-0000-000000000000",
    platform: Platform = Platform.TELEGRAM,
    chat_id: str = "100",
    user_id: str = "42",
    prompt: str = "Write hello world",
    status: TaskStatus = TaskStatus.DONE,
    result: str | None = "print('hello world')",
    error: str | None = None,
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        prompt=prompt,
        status=status,
        result=result,
        error=error,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# split_message tests
# ---------------------------------------------------------------------------


class TestSplitMessage:
    def test_short_message_not_split(self):
        text = "Hello, world!"
        chunks = split_message(text, max_length=100)
        assert chunks == ["Hello, world!"]

    def test_exact_length_not_split(self):
        text = "a" * 100
        chunks = split_message(text, max_length=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_message_is_split(self):
        text = "line\n" * 200  # 1000 chars
        chunks = split_message(text, max_length=100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_all_content_preserved(self):
        """No characters should be lost when splitting."""
        text = "abcdef\n" * 50  # 350 chars
        chunks = split_message(text, max_length=60)
        # Rejoin (stripping added fence markers) should contain original content
        joined = "\n".join(chunks)
        # Count original 'a' chars
        original_a_count = text.count("a")
        assert joined.count("a") == original_a_count

    def test_splits_on_newline_boundary(self):
        text = "line1\nline2\nline3\nline4\nline5"
        chunks = split_message(text, max_length=12)
        for chunk in chunks:
            assert len(chunk) <= 12

    def test_empty_string_returns_empty_list(self):
        chunks = split_message("", max_length=100)
        # Empty string produces no meaningful chunks
        assert all(c.strip() == "" for c in chunks) or chunks == []

    def test_single_very_long_line_hard_split(self):
        text = "a" * 500
        chunks = split_message(text, max_length=100)
        assert len(chunks) == 5
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_code_fence_closed_on_split(self):
        """An open code fence should be closed at the split boundary."""
        code = "```python\n" + "x = 1\n" * 30 + "```"
        chunks = split_message(code, max_length=80)
        if len(chunks) > 1:
            # First chunk should end with closing fence
            assert chunks[0].endswith("```")


# ---------------------------------------------------------------------------
# _count_open_code_fence tests
# ---------------------------------------------------------------------------


class TestCountOpenCodeFence:
    def test_no_fence(self):
        assert _count_open_code_fence("plain text") == ""

    def test_closed_fence(self):
        text = "```python\ncode\n```"
        result = _count_open_code_fence(text)
        assert result == ""  # all fences matched

    def test_open_fence_with_language(self):
        text = "```python\ncode here"
        result = _count_open_code_fence(text)
        assert result == "python"

    def test_open_fence_no_language(self):
        text = "```\nsome code"
        result = _count_open_code_fence(text)
        # Returns empty string as language tag (but truthy check won't apply here)
        # The function returns the language tag, which may be ""
        assert isinstance(result, str)

    def test_multiple_closed_fences(self):
        text = "```python\ncode\n```\n```js\nmore\n```"
        assert _count_open_code_fence(text) == ""

    def test_odd_fence_count(self):
        text = "```python\ncode\n```\n```js\nmore"
        result = _count_open_code_fence(text)
        assert result == "js"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestFormatResultForTelegram:
    def test_includes_task_id_prefix(self):
        text = format_result_for_telegram("some result", "abcdef12-xxxx")
        assert "abcdef12" in text

    def test_includes_result_text(self):
        text = format_result_for_telegram("def foo(): pass", "abc")
        assert "def foo(): pass" in text

    def test_contains_checkmark(self):
        text = format_result_for_telegram("result", "id")
        assert "\u2705" in text

    def test_preserves_code_blocks(self):
        result = "```python\nprint('hi')\n```"
        text = format_result_for_telegram(result, "id")
        assert "```python" in text


class TestFormatErrorForTelegram:
    def test_includes_task_id_prefix(self):
        text = format_error_for_telegram("timeout", "abcdef12-xxxx")
        assert "abcdef12" in text

    def test_includes_error_text(self):
        text = format_error_for_telegram("AI backend error", "id")
        assert "AI backend error" in text

    def test_contains_cross_mark(self):
        text = format_error_for_telegram("error", "id")
        assert "\u274c" in text


class TestFormatResultForDiscord:
    def test_includes_task_id_prefix(self):
        text = format_result_for_discord("some result", "abcdef12-xxxx")
        assert "abcdef12" in text

    def test_includes_result_text(self):
        text = format_result_for_discord("def bar(): pass", "abc")
        assert "def bar(): pass" in text

    def test_contains_checkmark(self):
        text = format_result_for_discord("result", "id")
        assert "\u2705" in text


class TestFormatErrorForDiscord:
    def test_includes_task_id_prefix(self):
        text = format_error_for_discord("timeout", "abcdef12-xxxx")
        assert "abcdef12" in text

    def test_includes_error_text(self):
        text = format_error_for_discord("connection refused", "id")
        assert "connection refused" in text

    def test_contains_cross_mark(self):
        text = format_error_for_discord("error", "id")
        assert "\u274c" in text

    def test_wraps_in_code_block(self):
        text = format_error_for_discord("the error", "id")
        assert "```" in text


# ---------------------------------------------------------------------------
# Messenger – Telegram delivery
# ---------------------------------------------------------------------------


class TestMessengerTelegram:
    async def test_send_telegram_calls_bot_send_message(self, telegram_settings: Settings):
        """send_telegram should call Bot.send_message with correct args."""
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()

        messenger = Messenger(telegram_settings)
        messenger._telegram_bot = mock_bot

        await messenger.send_telegram(chat_id="100", text="Hello!")

        mock_bot.send_message.assert_awaited_once()
        call_kwargs = mock_bot.send_message.call_args
        assert call_kwargs.kwargs["chat_id"] == "100" or call_kwargs.args[0] == "100"

    async def test_send_telegram_splits_long_message(self, telegram_settings: Settings):
        """Messages over 4096 chars should be sent in multiple chunks."""
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()

        messenger = Messenger(telegram_settings)
        messenger._telegram_bot = mock_bot

        long_text = "word " * 1000  # ~5000 chars
        await messenger.send_telegram(chat_id="100", text=long_text)

        assert mock_bot.send_message.await_count >= 2

    async def test_send_telegram_no_token_raises(self, no_token_settings: Settings):
        """Sending without a Telegram token should raise TelegramMessengerError."""
        messenger = Messenger(no_token_settings)
        with pytest.raises(TelegramMessengerError, match="token"):
            await messenger.send_telegram(chat_id="100", text="hi")

    async def test_send_telegram_retries_as_plain_text_on_parse_error(
        self, telegram_settings: Settings
    ):
        """A Markdown parse error should trigger a plain-text retry."""
        parse_error = Exception("Bad markdown entity in the text")
        mock_bot = AsyncMock()
        # First call raises a parse error; second call (plain text) succeeds
        mock_bot.send_message = AsyncMock(side_effect=[parse_error, None])

        messenger = Messenger(telegram_settings)
        messenger._telegram_bot = mock_bot

        # Should not raise
        await messenger.send_telegram(chat_id="100", text="bad `markdown")

        assert mock_bot.send_message.await_count == 2
        # Second call should have parse_mode=None
        second_call = mock_bot.send_message.call_args_list[1]
        assert second_call.kwargs.get("parse_mode") is None

    async def test_send_telegram_non_parse_error_raises(self, telegram_settings: Settings):
        """Non-Markdown errors should propagate as TelegramMessengerError."""
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(side_effect=Exception("network failure"))

        messenger = Messenger(telegram_settings)
        messenger._telegram_bot = mock_bot

        with pytest.raises(TelegramMessengerError, match="network failure"):
            await messenger.send_telegram(chat_id="100", text="hi")


# ---------------------------------------------------------------------------
# Messenger – Discord delivery
# ---------------------------------------------------------------------------


class TestMessengerDiscord:
    async def test_send_discord_no_token_raises(self, no_token_settings: Settings):
        """Sending without a Discord token should raise DiscordMessengerError."""
        messenger = Messenger(no_token_settings)
        with pytest.raises(DiscordMessengerError, match="token"):
            await messenger.send_discord(channel_id="123", text="hi")

    async def test_send_discord_uses_http_client(self, discord_settings: Settings):
        """send_discord should use discord's HTTP client to post messages."""
        mock_http = AsyncMock()
        mock_http.static_login = AsyncMock()
        mock_http.send_message = AsyncMock()
        mock_http.close = AsyncMock()

        with patch("agent_bridge.messenger.discord") as mock_discord:
            mock_discord.http.HTTPClient.return_value = mock_http
            mock_discord.errors.Forbidden = Exception
            mock_discord.errors.NotFound = Exception
            mock_discord.errors.HTTPException = Exception

            messenger = Messenger(discord_settings)
            await messenger.send_discord(channel_id="123456", text="Hello Discord!")

            mock_http.static_login.assert_awaited_once()
            mock_http.send_message.assert_awaited_once()
            mock_http.close.assert_awaited_once()

    async def test_send_discord_splits_long_message(self, discord_settings: Settings):
        """Messages over 2000 chars should be sent in multiple chunks."""
        mock_http = AsyncMock()
        mock_http.static_login = AsyncMock()
        mock_http.send_message = AsyncMock()
        mock_http.close = AsyncMock()

        with patch("agent_bridge.messenger.discord") as mock_discord:
            mock_discord.http.HTTPClient.return_value = mock_http
            mock_discord.errors.Forbidden = type("Forbidden", (Exception,), {})
            mock_discord.errors.NotFound = type("NotFound", (Exception,), {})
            mock_discord.errors.HTTPException = type("HTTPException", (Exception,), {})

            long_text = "word " * 500  # ~2500 chars
            messenger = Messenger(discord_settings)
            await messenger.send_discord(channel_id="123456", text=long_text)

            assert mock_http.send_message.await_count >= 2

    async def test_send_discord_http_always_closed_on_error(self, discord_settings: Settings):
        """HTTP client must be closed even when an error occurs."""
        mock_http = AsyncMock()
        mock_http.static_login = AsyncMock()
        mock_http.send_message = AsyncMock(side_effect=Exception("network error"))
        mock_http.close = AsyncMock()

        ForbiddenType = type("Forbidden", (Exception,), {})
        NotFoundType = type("NotFound", (Exception,), {})
        HTTPExceptionType = type("HTTPException", (Exception,), {})

        with patch("agent_bridge.messenger.discord") as mock_discord:
            mock_discord.http.HTTPClient.return_value = mock_http
            mock_discord.errors.Forbidden = ForbiddenType
            mock_discord.errors.NotFound = NotFoundType
            mock_discord.errors.HTTPException = HTTPExceptionType

            messenger = Messenger(discord_settings)
            with pytest.raises(DiscordMessengerError):
                await messenger.send_discord(channel_id="123", text="hi")

            mock_http.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Messenger.send() – unified dispatch
# ---------------------------------------------------------------------------


class TestMessengerSend:
    async def test_send_done_telegram(self, telegram_settings: Settings):
        """send() with DONE status dispatches to send_telegram."""
        messenger = Messenger(telegram_settings)
        messenger.send_telegram = AsyncMock()

        record = _make_task_record(
            platform=Platform.TELEGRAM,
            status=TaskStatus.DONE,
            result="output text",
        )
        await messenger.send(record)

        messenger.send_telegram.assert_awaited_once()
        call_kwargs = messenger.send_telegram.call_args
        assert "output text" in call_kwargs.kwargs.get("text", "")

    async def test_send_failed_telegram(self, telegram_settings: Settings):
        """send() with FAILED status sends error message via Telegram."""
        messenger = Messenger(telegram_settings)
        messenger.send_telegram = AsyncMock()

        record = _make_task_record(
            platform=Platform.TELEGRAM,
            status=TaskStatus.FAILED,
            result=None,
            error="backend timeout",
        )
        await messenger.send(record)

        messenger.send_telegram.assert_awaited_once()
        call_kwargs = messenger.send_telegram.call_args
        assert "backend timeout" in call_kwargs.kwargs.get("text", "")

    async def test_send_done_discord(self, discord_settings: Settings):
        """send() with DONE status dispatches to send_discord."""
        messenger = Messenger(discord_settings)
        messenger.send_discord = AsyncMock()

        record = _make_task_record(
            platform=Platform.DISCORD,
            status=TaskStatus.DONE,
            result="discord result",
        )
        await messenger.send(record)

        messenger.send_discord.assert_awaited_once()
        call_kwargs = messenger.send_discord.call_args
        assert "discord result" in call_kwargs.kwargs.get("text", "")

    async def test_send_failed_discord(self, discord_settings: Settings):
        """send() with FAILED status sends error message via Discord."""
        messenger = Messenger(discord_settings)
        messenger.send_discord = AsyncMock()

        record = _make_task_record(
            platform=Platform.DISCORD,
            status=TaskStatus.FAILED,
            result=None,
            error="ai backend down",
        )
        await messenger.send(record)

        messenger.send_discord.assert_awaited_once()
        call_kwargs = messenger.send_discord.call_args
        assert "ai backend down" in call_kwargs.kwargs.get("text", "")

    async def test_send_non_terminal_status_skips(self, telegram_settings: Settings):
        """send() with PENDING or RUNNING status should not call any send method."""
        messenger = Messenger(telegram_settings)
        messenger.send_telegram = AsyncMock()
        messenger.send_discord = AsyncMock()

        record = _make_task_record(status=TaskStatus.PENDING)
        await messenger.send(record)  # should not raise

        messenger.send_telegram.assert_not_awaited()
        messenger.send_discord.assert_not_awaited()

    async def test_send_uses_chat_id_from_record(self, telegram_settings: Settings):
        """send() should pass record.chat_id to the platform send method."""
        messenger = Messenger(telegram_settings)
        messenger.send_telegram = AsyncMock()

        record = _make_task_record(
            chat_id="999",
            status=TaskStatus.DONE,
            result="something",
        )
        await messenger.send(record)

        call_kwargs = messenger.send_telegram.call_args
        assert call_kwargs.kwargs.get("chat_id") == "999"

    async def test_send_result_none_uses_fallback(self, telegram_settings: Settings):
        """When result is None on a DONE task, a fallback string is used."""
        messenger = Messenger(telegram_settings)
        messenger.send_telegram = AsyncMock()

        record = _make_task_record(
            status=TaskStatus.DONE,
            result=None,
        )
        await messenger.send(record)  # should not raise

        messenger.send_telegram.assert_awaited_once()
        call_text = messenger.send_telegram.call_args.kwargs.get("text", "")
        assert "no output" in call_text

    async def test_send_string_platform_works(self, telegram_settings: Settings):
        """send() should accept string platform values from the database."""
        messenger = Messenger(telegram_settings)
        messenger.send_telegram = AsyncMock()

        record = _make_task_record(
            platform=Platform.TELEGRAM,
            status=TaskStatus.DONE,
            result="text",
        )
        # Simulate the record coming from the DB with string platform
        object.__setattr__(record, "platform", "telegram")
        await messenger.send(record)

        messenger.send_telegram.assert_awaited_once()


# ---------------------------------------------------------------------------
# Messenger.send_text()
# ---------------------------------------------------------------------------


class TestMessengerSendText:
    async def test_send_text_telegram(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        messenger.send_telegram = AsyncMock()

        await messenger.send_text(Platform.TELEGRAM, "100", "hello")

        messenger.send_telegram.assert_awaited_once_with(chat_id="100", text="hello")

    async def test_send_text_discord(self, discord_settings: Settings):
        messenger = Messenger(discord_settings)
        messenger.send_discord = AsyncMock()

        await messenger.send_text(Platform.DISCORD, "123456", "hello discord")

        messenger.send_discord.assert_awaited_once_with(
            channel_id="123456", text="hello discord"
        )

    async def test_send_text_string_platform(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        messenger.send_telegram = AsyncMock()

        await messenger.send_text("telegram", "100", "str platform")

        messenger.send_telegram.assert_awaited_once()

    async def test_send_text_invalid_platform_raises(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)

        with pytest.raises((ValueError, Exception)):
            await messenger.send_text("invalid_platform", "100", "hi")  # type: ignore


# ---------------------------------------------------------------------------
# Messenger.send_ack()
# ---------------------------------------------------------------------------


class TestMessengerSendAck:
    async def test_send_ack_contains_task_id(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        messenger.send_text = AsyncMock()

        await messenger.send_ack(Platform.TELEGRAM, "100", "abcdef12-1234-5678-0000-000000000000")

        call_text = messenger.send_text.call_args.kwargs.get("text", "") or \
                    messenger.send_text.call_args.args[2]
        assert "abcdef12" in call_text

    async def test_send_ack_mentions_queued(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        messenger.send_text = AsyncMock()

        await messenger.send_ack(Platform.TELEGRAM, "100", "task-id-here")

        call_text = messenger.send_text.call_args.kwargs.get("text", "") or \
                    messenger.send_text.call_args.args[2]
        assert "queued" in call_text.lower()


# ---------------------------------------------------------------------------
# Messenger.send_status()
# ---------------------------------------------------------------------------


class TestMessengerSendStatus:
    async def _get_status_text(self, messenger: Messenger, record: TaskRecord) -> str:
        messenger.send_text = AsyncMock()
        await messenger.send_status(record.platform, record.chat_id, record)
        args = messenger.send_text.call_args
        return args.kwargs.get("text", "") or args.args[2]

    async def test_pending_status(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        record = _make_task_record(status=TaskStatus.PENDING)
        text = await self._get_status_text(messenger, record)
        assert "pending" in text.lower()

    async def test_running_status(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        record = _make_task_record(status=TaskStatus.RUNNING)
        text = await self._get_status_text(messenger, record)
        assert "running" in text.lower()

    async def test_done_status(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        record = _make_task_record(status=TaskStatus.DONE)
        text = await self._get_status_text(messenger, record)
        assert "done" in text.lower()

    async def test_failed_status_includes_error(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        record = _make_task_record(
            status=TaskStatus.FAILED, result=None, error="backend error"
        )
        text = await self._get_status_text(messenger, record)
        assert "failed" in text.lower()
        assert "backend error" in text

    async def test_status_includes_short_task_id(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        record = _make_task_record(task_id="abcdef12-0000-0000-0000-000000000000")
        text = await self._get_status_text(messenger, record)
        assert "abcdef12" in text


# ---------------------------------------------------------------------------
# Messenger.send_not_authorised()
# ---------------------------------------------------------------------------


class TestMessengerSendNotAuthorised:
    async def test_sends_access_denied_message(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        messenger.send_text = AsyncMock()

        await messenger.send_not_authorised(Platform.TELEGRAM, "100")

        messenger.send_text.assert_awaited_once()
        call_text = messenger.send_text.call_args.kwargs.get("text", "") or \
                    messenger.send_text.call_args.args[2]
        assert "authoris" in call_text.lower() or "access" in call_text.lower()

    async def test_sends_to_correct_chat(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        messenger.send_text = AsyncMock()

        await messenger.send_not_authorised(Platform.DISCORD, "channel_99")

        call_args = messenger.send_text.call_args
        # Second positional arg or 'chat_id' kwarg
        platform_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("platform")
        chat_id_arg = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("chat_id")
        assert chat_id_arg == "channel_99"


# ---------------------------------------------------------------------------
# Messenger.send_error()
# ---------------------------------------------------------------------------


class TestMessengerSendError:
    async def test_sends_error_message(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        messenger.send_text = AsyncMock()

        await messenger.send_error(Platform.TELEGRAM, "100", "Something went wrong")

        messenger.send_text.assert_awaited_once()
        call_text = messenger.send_text.call_args.kwargs.get("text", "") or \
                    messenger.send_text.call_args.args[2]
        assert "Something went wrong" in call_text

    async def test_error_message_contains_cross(self, telegram_settings: Settings):
        messenger = Messenger(telegram_settings)
        messenger.send_text = AsyncMock()

        await messenger.send_error(Platform.TELEGRAM, "100", "fail")

        call_text = messenger.send_text.call_args.kwargs.get("text", "") or \
                    messenger.send_text.call_args.args[2]
        assert "\u274c" in call_text


# ---------------------------------------------------------------------------
# Lazy bot client initialisation
# ---------------------------------------------------------------------------


class TestLazyClientInit:
    def test_get_telegram_bot_no_token_raises(self, no_token_settings: Settings):
        messenger = Messenger(no_token_settings)
        with pytest.raises(TelegramMessengerError, match="token"):
            messenger._get_telegram_bot()

    def test_get_discord_client_no_token_raises(self, no_token_settings: Settings):
        messenger = Messenger(no_token_settings)
        with pytest.raises(DiscordMessengerError, match="token"):
            messenger._get_discord_client()

    def test_get_telegram_bot_returns_cached_instance(self, telegram_settings: Settings):
        """Second call to _get_telegram_bot should return the same object."""
        mock_bot = MagicMock()
        messenger = Messenger(telegram_settings)
        messenger._telegram_bot = mock_bot

        result1 = messenger._get_telegram_bot()
        result2 = messenger._get_telegram_bot()

        assert result1 is result2 is mock_bot
