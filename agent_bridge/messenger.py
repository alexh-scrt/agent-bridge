"""Unified messenger abstraction for agent_bridge.

Provides a platform-agnostic interface for delivering AI-generated text and
code results back to the originating Telegram or Discord chat. Handles
message formatting (code blocks, length limits) for each platform.

Typical usage::

    messenger = Messenger(settings)
    await messenger.send(record)  # record.platform determines the target

    # Or send directly to a specific platform:
    await messenger.send_telegram(chat_id="100", text="Hello!")
    await messenger.send_discord(channel_id="123456789", text="Hello!")
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent_bridge.config import Settings
from agent_bridge.models import Platform, TaskRecord, TaskStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform-specific message length limits
# ---------------------------------------------------------------------------

# Telegram: 4096 UTF-16 code units; we use a safe conservative limit
TELEGRAM_MAX_LENGTH: int = 4096

# Discord: 2000 characters per message
DISCORD_MAX_LENGTH: int = 2000

# Prefix added to continuation messages when splitting long responses
_CONTINUATION_PREFIX: str = "(continued)\n"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_result_for_telegram(result: str, task_id: str) -> str:
    """Format a completed AI result for Telegram using MarkdownV2 fencing.

    If the result already contains fenced code blocks (triple backtick), it is
    returned as-is (after adding a header). Otherwise the entire result is
    wrapped in a generic code block.

    Args:
        result: The raw AI-generated result string.
        task_id: The task UUID, used in the header line.

    Returns:
        A formatted string ready to send to Telegram.
    """
    header = f"\u2705 Task `{task_id[:8]}` completed:\n\n"
    # If the result already has code fences, keep the structure intact
    if "```" in result:
        return header + result
    return header + result


def format_error_for_telegram(error: str, task_id: str) -> str:
    """Format a failed task error message for Telegram.

    Args:
        error: The error description string.
        task_id: The task UUID, used in the header line.

    Returns:
        A formatted error string ready to send to Telegram.
    """
    return (
        f"\u274c Task `{task_id[:8]}` failed:\n"
        f"`{error}`"
    )


def format_result_for_discord(result: str, task_id: str) -> str:
    """Format a completed AI result for Discord using Markdown code fencing.

    Args:
        result: The raw AI-generated result string.
        task_id: The task UUID, used in the header line.

    Returns:
        A formatted string ready to send to Discord.
    """
    header = f"\u2705 Task `{task_id[:8]}` completed:\n\n"
    if "```" in result:
        return header + result
    return header + result


def format_error_for_discord(error: str, task_id: str) -> str:
    """Format a failed task error message for Discord.

    Args:
        error: The error description string.
        task_id: The task UUID, used in the header line.

    Returns:
        A formatted error string ready to send to Discord.
    """
    return (
        f"\u274c Task `{task_id[:8]}` failed:\n"
        f"```\n{error}\n```"
    )


def split_message(text: str, max_length: int) -> list[str]:
    """Split a long message into chunks that fit within the platform's limit.

    Attempts to split on newline boundaries to preserve readability. If a
    single line exceeds ``max_length``, it is split at the character limit.

    Code blocks that span a split boundary are closed and re-opened in the
    next chunk to maintain valid Markdown formatting.

    Args:
        text: The full message text to split.
        max_length: Maximum character count per chunk.

    Returns:
        A list of one or more non-empty strings, each within ``max_length``.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_length:
        # Try to split on a newline boundary within the limit
        split_at = remaining.rfind("\n", 0, max_length)
        if split_at <= 0:
            # No newline found – hard split at max_length
            split_at = max_length

        chunk = remaining[:split_at]
        remaining = remaining[split_at:].lstrip("\n")

        # Detect if we're inside an open code fence and close/reopen it
        open_fence = _count_open_code_fence(chunk)
        if open_fence:
            chunk += "\n```"
            remaining = f"```{open_fence}\n" + remaining

        chunks.append(chunk)

    if remaining:
        chunks.append(remaining)

    return [c for c in chunks if c.strip()]


def _count_open_code_fence(text: str) -> str:
    """Return the language tag of an unclosed code fence, or empty string.

    Scans ``text`` for triple-backtick fences and returns the language tag
    of the last unclosed opening fence, or an empty string if all fences
    are matched.

    Args:
        text: The text to inspect.

    Returns:
        Language tag (may be empty string) if a fence is open, else ``""``
        (but distinguishable: returns ``None`` when closed – actually returns
        the language tag or empty string when open, ``""`` ... see below).

    Note:
        Returns the language specifier string (possibly "") when inside a code
        block, or ``None`` when outside. Callers check truthiness for the
        ``\`\`\`` re-open case and use the returned string for the language tag.
    """
    # Find all ``` occurrences with optional language tag
    pattern = re.compile(r"```(\w*)")
    matches = pattern.findall(text)
    # Every two matches = open + close; odd count = unclosed
    if len(matches) % 2 == 0:
        return ""  # all closed
    # Return the language tag of the last opening fence
    return matches[-1] if matches else ""


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MessengerError(Exception):
    """Raised when a message cannot be delivered to the target platform."""


class TelegramMessengerError(MessengerError):
    """Raised for Telegram-specific delivery errors."""


class DiscordMessengerError(MessengerError):
    """Raised for Discord-specific delivery errors."""


# ---------------------------------------------------------------------------
# Main Messenger class
# ---------------------------------------------------------------------------


class Messenger:
    """Unified messenger that delivers results to Telegram or Discord.

    Wraps the ``python-telegram-bot`` and ``discord.py`` libraries behind a
    single async interface. Lazy-initialises bot clients on first use so that
    the application can start without both tokens being present.

    Args:
        settings: Application configuration containing bot tokens.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._telegram_bot: Any | None = None
        self._discord_client: Any | None = None

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------

    def _get_telegram_bot(self) -> Any:
        """Lazily construct and return the Telegram Bot instance.

        Returns:
            A :class:`telegram.Bot` instance.

        Raises:
            TelegramMessengerError: If the Telegram bot token is not configured.
        """
        if self._telegram_bot is not None:
            return self._telegram_bot

        if not self._settings.telegram_enabled:
            raise TelegramMessengerError(
                "Telegram bot token is not configured. "
                "Set the TELEGRAM_BOT_TOKEN environment variable."
            )

        try:
            from telegram import Bot  # type: ignore[import]
            token = self._settings.telegram_bot_token.get_secret_value()  # type: ignore[union-attr]
            self._telegram_bot = Bot(token=token)
        except ImportError as exc:
            raise TelegramMessengerError(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            ) from exc

        return self._telegram_bot

    async def send_telegram(
        self,
        chat_id: str | int,
        text: str,
        parse_mode: str = "Markdown",
    ) -> None:
        """Send a text message to a Telegram chat.

        Long messages are automatically split into multiple sends, each within
        Telegram's 4096-character limit. If Markdown parsing fails (e.g. due
        to malformed entities), the message is retried as plain text.

        Args:
            chat_id: The Telegram chat ID (int or string representation).
            text: The message text to send.
            parse_mode: Telegram parse mode (``'Markdown'``, ``'HTML'``, or
                ``None`` for plain text).

        Raises:
            TelegramMessengerError: If the message cannot be delivered.
        """
        bot = self._get_telegram_bot()
        chunks = split_message(text, TELEGRAM_MAX_LENGTH)

        for i, chunk in enumerate(chunks):
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                )
                logger.debug(
                    "Sent Telegram message chunk %d/%d to chat %s",
                    i + 1,
                    len(chunks),
                    chat_id,
                )
            except Exception as exc:
                # Attempt to detect Markdown parse errors and fall back to plain text
                error_str = str(exc).lower()
                if "parse" in error_str or "markdown" in error_str or "entity" in error_str:
                    logger.warning(
                        "Telegram Markdown parse error for chat %s, retrying as plain text: %s",
                        chat_id,
                        exc,
                    )
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            parse_mode=None,
                        )
                        continue
                    except Exception as plain_exc:
                        raise TelegramMessengerError(
                            f"Failed to send plain-text fallback to Telegram chat {chat_id}: {plain_exc}"
                        ) from plain_exc
                raise TelegramMessengerError(
                    f"Failed to send message to Telegram chat {chat_id}: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Discord
    # ------------------------------------------------------------------

    def _get_discord_client(self) -> Any:
        """Lazily construct and return the Discord client instance.

        Returns:
            A :class:`discord.Client` instance.

        Raises:
            DiscordMessengerError: If the Discord bot token is not configured.
        """
        if self._discord_client is not None:
            return self._discord_client

        if not self._settings.discord_enabled:
            raise DiscordMessengerError(
                "Discord bot token is not configured. "
                "Set the DISCORD_BOT_TOKEN environment variable."
            )

        try:
            import discord  # type: ignore[import]

            intents = discord.Intents.default()
            self._discord_client = discord.Client(intents=intents)
        except ImportError as exc:
            raise DiscordMessengerError(
                "discord.py is not installed. "
                "Install it with: pip install discord.py"
            ) from exc

        return self._discord_client

    async def send_discord(
        self,
        channel_id: str | int,
        text: str,
    ) -> None:
        """Send a text message to a Discord channel.

        Long messages are automatically split into multiple sends, each within
        Discord's 2000-character limit.

        Uses the Discord HTTP API directly via :mod:`discord` to fetch the
        channel and send the message without requiring a running gateway
        connection.

        Args:
            channel_id: The Discord channel snowflake ID (int or string).
            text: The message text to send.

        Raises:
            DiscordMessengerError: If the message cannot be delivered.
        """
        if not self._settings.discord_enabled:
            raise DiscordMessengerError(
                "Discord bot token is not configured. "
                "Set the DISCORD_BOT_TOKEN environment variable."
            )

        try:
            import discord  # type: ignore[import]
        except ImportError as exc:
            raise DiscordMessengerError(
                "discord.py is not installed."
            ) from exc

        token = self._settings.discord_bot_token.get_secret_value()  # type: ignore[union-attr]
        channel_id_int = int(channel_id)
        chunks = split_message(text, DISCORD_MAX_LENGTH)

        # Use discord.py's HTTP client directly to avoid needing a running event loop
        # gateway connection.  We create a minimal HTTP client, fetch the channel,
        # and send messages.
        http = discord.http.HTTPClient()
        try:
            await http.static_login(token)

            for i, chunk in enumerate(chunks):
                try:
                    await http.send_message(channel_id_int, content=chunk)
                    logger.debug(
                        "Sent Discord message chunk %d/%d to channel %s",
                        i + 1,
                        len(chunks),
                        channel_id,
                    )
                except discord.errors.Forbidden as exc:
                    raise DiscordMessengerError(
                        f"Bot lacks permission to send to Discord channel {channel_id}: {exc}"
                    ) from exc
                except discord.errors.NotFound as exc:
                    raise DiscordMessengerError(
                        f"Discord channel {channel_id} not found: {exc}"
                    ) from exc
                except discord.errors.HTTPException as exc:
                    raise DiscordMessengerError(
                        f"Discord HTTP error sending to channel {channel_id}: {exc}"
                    ) from exc
        except DiscordMessengerError:
            raise
        except Exception as exc:
            raise DiscordMessengerError(
                f"Unexpected error sending to Discord channel {channel_id}: {exc}"
            ) from exc
        finally:
            try:
                await http.close()
            except Exception:
                pass  # best-effort cleanup

    # ------------------------------------------------------------------
    # Unified send
    # ------------------------------------------------------------------

    async def send(self, record: TaskRecord) -> None:
        """Deliver a completed or failed task result to the originating chat.

        Dispatches to the appropriate platform-specific send method based on
        ``record.platform``. The message is formatted differently for success
        (DONE) and failure (FAILED) states.

        Args:
            record: The final :class:`~agent_bridge.models.TaskRecord` with
                status :attr:`TaskStatus.DONE` or :attr:`TaskStatus.FAILED`.

        Raises:
            MessengerError: If the message cannot be delivered to the platform.
            ValueError: If the record's platform is not supported.
        """
        platform = record.platform
        # Normalise to enum if it came in as a string
        if isinstance(platform, str):
            platform = Platform(platform)

        status = record.status
        if isinstance(status, str):
            status = TaskStatus(status)

        if status == TaskStatus.DONE:
            result_text = record.result or "(no output)"
            if platform == Platform.TELEGRAM:
                message = format_result_for_telegram(result_text, record.id)
                await self.send_telegram(chat_id=record.chat_id, text=message)
            elif platform == Platform.DISCORD:
                message = format_result_for_discord(result_text, record.id)
                await self.send_discord(channel_id=record.chat_id, text=message)
            else:
                raise ValueError(f"Unsupported platform: {platform!r}")

        elif status == TaskStatus.FAILED:
            error_text = record.error or "Unknown error"
            if platform == Platform.TELEGRAM:
                message = format_error_for_telegram(error_text, record.id)
                await self.send_telegram(chat_id=record.chat_id, text=message)
            elif platform == Platform.DISCORD:
                message = format_error_for_discord(error_text, record.id)
                await self.send_discord(channel_id=record.chat_id, text=message)
            else:
                raise ValueError(f"Unsupported platform: {platform!r}")

        else:
            logger.warning(
                "send() called with non-terminal task status %r for task %s – skipping",
                status,
                record.id,
            )

    async def send_text(
        self,
        platform: Platform | str,
        chat_id: str | int,
        text: str,
    ) -> None:
        """Send an arbitrary text message to a platform chat.

        A lower-level method that bypasses task record formatting and sends
        ``text`` directly to the specified chat.

        Args:
            platform: Target platform (:attr:`Platform.TELEGRAM` or
                :attr:`Platform.DISCORD`).
            chat_id: Platform-specific chat or channel identifier.
            text: The message text to deliver.

        Raises:
            MessengerError: If the message cannot be delivered.
            ValueError: If the platform is not supported.
        """
        if isinstance(platform, str):
            platform = Platform(platform)

        if platform == Platform.TELEGRAM:
            await self.send_telegram(chat_id=chat_id, text=text)
        elif platform == Platform.DISCORD:
            await self.send_discord(channel_id=chat_id, text=text)
        else:
            raise ValueError(f"Unsupported platform: {platform!r}")

    # ------------------------------------------------------------------
    # Status / acknowledgement helpers
    # ------------------------------------------------------------------

    async def send_ack(
        self,
        platform: Platform | str,
        chat_id: str | int,
        task_id: str,
    ) -> None:
        """Send an acknowledgement message that a task has been queued.

        Args:
            platform: Target platform.
            chat_id: Platform-specific chat or channel identifier.
            task_id: The UUID of the queued task.

        Raises:
            MessengerError: If the acknowledgement cannot be delivered.
        """
        short_id = task_id[:8]
        text = (
            f"\u23f3 Your request has been queued (task `{short_id}`).\n"
            f"I'll send the result here when it's ready."
        )
        await self.send_text(platform=platform, chat_id=chat_id, text=text)

    async def send_status(
        self,
        platform: Platform | str,
        chat_id: str | int,
        record: TaskRecord,
    ) -> None:
        """Send a task status summary message to a chat.

        Formats a human-readable status update and delivers it to the
        specified chat. Does not include the full result text.

        Args:
            platform: Target platform.
            chat_id: Platform-specific chat or channel identifier.
            record: The task record whose status to report.

        Raises:
            MessengerError: If the status message cannot be delivered.
        """
        status = record.status
        if isinstance(status, str):
            status = TaskStatus(status)

        short_id = record.id[:8]

        if status == TaskStatus.PENDING:
            icon = "\u23f3"
            status_text = "pending (queued)"
        elif status == TaskStatus.RUNNING:
            icon = "\u26a1"
            status_text = "running"
        elif status == TaskStatus.DONE:
            icon = "\u2705"
            status_text = "done"
        elif status == TaskStatus.FAILED:
            icon = "\u274c"
            status_text = f"failed: {record.error or 'unknown error'}"
        else:
            icon = "\u2754"
            status_text = str(status)

        text = f"{icon} Task `{short_id}` status: **{status_text}**"
        await self.send_text(platform=platform, chat_id=chat_id, text=text)

    async def send_error(
        self,
        platform: Platform | str,
        chat_id: str | int,
        error_message: str,
    ) -> None:
        """Send a generic error message to a chat.

        Args:
            platform: Target platform.
            chat_id: Platform-specific chat or channel identifier.
            error_message: Human-readable error description.

        Raises:
            MessengerError: If the error message cannot be delivered.
        """
        text = f"\u274c Error: {error_message}"
        await self.send_text(platform=platform, chat_id=chat_id, text=text)

    async def send_not_authorised(
        self,
        platform: Platform | str,
        chat_id: str | int,
    ) -> None:
        """Send an 'access denied' message to an unauthorised user.

        Args:
            platform: Target platform.
            chat_id: Platform-specific chat or channel identifier.

        Raises:
            MessengerError: If the message cannot be delivered.
        """
        text = (
            "\U0001f512 You are not authorised to use this bot. "
            "Contact the administrator to request access."
        )
        await self.send_text(platform=platform, chat_id=chat_id, text=text)
