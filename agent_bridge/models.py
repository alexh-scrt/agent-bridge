"""Pydantic models for agent_bridge.

Defines data models for:
- Incoming webhook payloads (Telegram, Discord)
- Task records with status tracking
- API responses for task queries and creation
- Callback payloads for task completion notifications
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Lifecycle states for an AI task."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Platform(str, Enum):
    """Source platform that originated a task."""

    TELEGRAM = "telegram"
    DISCORD = "discord"


# ---------------------------------------------------------------------------
# Telegram webhook payload models
# ---------------------------------------------------------------------------


class TelegramUser(BaseModel):
    """Represents the sender of a Telegram message."""

    id: int = Field(..., description="Telegram user ID")
    is_bot: bool = Field(default=False, description="True if the sender is a bot")
    first_name: str = Field(default="", description="User's first name")
    last_name: str | None = Field(default=None, description="User's last name")
    username: str | None = Field(default=None, description="User's @username")
    language_code: str | None = Field(default=None, description="IETF language tag")

    model_config = {"extra": "allow"}


class TelegramChat(BaseModel):
    """Represents a Telegram chat (private, group, supergroup, or channel)."""

    id: int = Field(..., description="Unique identifier for this chat")
    type: str = Field(..., description="Type of chat: private, group, supergroup, or channel")
    title: str | None = Field(default=None, description="Title, for supergroups and channels")
    username: str | None = Field(default=None, description="Username for private chats or channels")

    model_config = {"extra": "allow"}


class TelegramMessage(BaseModel):
    """Represents an incoming Telegram message."""

    message_id: int = Field(..., description="Unique message identifier within the chat")
    from_user: TelegramUser | None = Field(
        default=None, alias="from", description="Sender of the message"
    )
    chat: TelegramChat = Field(..., description="Conversation the message belongs to")
    date: int = Field(..., description="Unix timestamp of the message")
    text: str | None = Field(default=None, description="Text content of the message")

    model_config = {"extra": "allow", "populate_by_name": True}


class TelegramUpdate(BaseModel):
    """Top-level Telegram webhook update payload."""

    update_id: int = Field(..., description="Unique update identifier")
    message: TelegramMessage | None = Field(
        default=None, description="New incoming message"
    )
    edited_message: TelegramMessage | None = Field(
        default=None, description="Edited message"
    )

    model_config = {"extra": "allow"}

    @property
    def effective_message(self) -> TelegramMessage | None:
        """Return the most relevant message from this update."""
        return self.message or self.edited_message

    @property
    def sender_id(self) -> int | None:
        """Return the sender's Telegram user ID, if available."""
        msg = self.effective_message
        if msg and msg.from_user:
            return msg.from_user.id
        return None

    @property
    def chat_id(self) -> int | None:
        """Return the chat ID from the effective message."""
        msg = self.effective_message
        if msg:
            return msg.chat.id
        return None

    @property
    def text(self) -> str | None:
        """Return the text from the effective message."""
        msg = self.effective_message
        if msg:
            return msg.text
        return None


# ---------------------------------------------------------------------------
# Discord webhook payload models
# ---------------------------------------------------------------------------


class DiscordUser(BaseModel):
    """Represents a Discord user."""

    id: str = Field(..., description="Discord user snowflake ID (as string)")
    username: str = Field(..., description="User's Discord username")
    discriminator: str = Field(default="0", description="Four-digit discriminator tag")
    global_name: str | None = Field(default=None, description="Global display name")
    bot: bool = Field(default=False, description="True if the user is a bot")

    model_config = {"extra": "allow"}

    @property
    def id_int(self) -> int:
        """Return the user ID as an integer."""
        return int(self.id)


class DiscordMessage(BaseModel):
    """Represents an incoming Discord message event payload.

    This model is used by agent_bridge's internal webhook ingestion endpoint
    rather than the raw Discord gateway payload (which is handled by discord.py).
    """

    id: str = Field(..., description="Discord message snowflake ID")
    channel_id: str = Field(..., description="Channel the message was sent in")
    guild_id: str | None = Field(default=None, description="Guild (server) ID, if applicable")
    author: DiscordUser = Field(..., description="The user who sent the message")
    content: str = Field(default="", description="Text content of the message")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the message")

    model_config = {"extra": "allow"}


class DiscordInboundPayload(BaseModel):
    """Payload posted to the agent_bridge Discord webhook ingestion endpoint."""

    message: DiscordMessage = Field(..., description="The Discord message to process")


# ---------------------------------------------------------------------------
# Internal task record models
# ---------------------------------------------------------------------------


class TaskRecord(BaseModel):
    """Full representation of a task stored in the database."""

    id: str = Field(..., description="UUID4 task identifier")
    platform: Platform = Field(..., description="Originating platform")
    chat_id: str = Field(..., description="Platform-specific chat/channel identifier")
    user_id: str = Field(..., description="Platform-specific user identifier")
    prompt: str = Field(..., description="The user's original prompt text")
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current lifecycle status"
    )
    result: str | None = Field(
        default=None, description="AI-generated result text (set when status=done)"
    )
    error: str | None = Field(
        default=None, description="Error description (set when status=failed)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="UTC timestamp when task was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the last status update",
    )

    model_config = {"use_enum_values": False}


class TaskCreate(BaseModel):
    """Input model for creating a new task."""

    platform: Platform = Field(..., description="Originating platform")
    chat_id: str = Field(..., description="Platform-specific chat/channel identifier")
    user_id: str = Field(..., description="Platform-specific user identifier")
    prompt: str = Field(..., min_length=1, description="The user's prompt text")

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        """Ensure the prompt is not blank after stripping whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt must not be blank")
        return stripped


class TaskStatusUpdate(BaseModel):
    """Partial update payload for changing a task's status and optional result/error."""

    status: TaskStatus = Field(..., description="New status to apply")
    result: str | None = Field(default=None, description="AI result text, if completed")
    error: str | None = Field(default=None, description="Error description, if failed")


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class TaskResponse(BaseModel):
    """API response model for a single task record."""

    id: str = Field(..., description="Task UUID")
    platform: str = Field(..., description="Originating platform name")
    chat_id: str = Field(..., description="Platform chat/channel ID")
    user_id: str = Field(..., description="Platform user ID")
    prompt: str = Field(..., description="Original prompt")
    status: str = Field(..., description="Current task status")
    result: str | None = Field(default=None, description="AI result, if available")
    error: str | None = Field(default=None, description="Error detail, if failed")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")
    updated_at: datetime = Field(..., description="Last update timestamp (UTC)")

    @classmethod
    def from_record(cls, record: TaskRecord) -> "TaskResponse":
        """Construct a TaskResponse from a TaskRecord."""
        return cls(
            id=record.id,
            platform=record.platform.value if isinstance(record.platform, Platform) else record.platform,
            chat_id=record.chat_id,
            user_id=record.user_id,
            prompt=record.prompt,
            status=record.status.value if isinstance(record.status, TaskStatus) else record.status,
            result=record.result,
            error=record.error,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class TaskListResponse(BaseModel):
    """API response model for a list of tasks."""

    tasks: list[TaskResponse] = Field(default_factory=list, description="List of task records")
    total: int = Field(..., description="Total number of tasks matching the query")


class TaskCreatedResponse(BaseModel):
    """API response returned when a new task is successfully enqueued."""

    task_id: str = Field(..., description="UUID of the newly created task")
    status: str = Field(default=TaskStatus.PENDING.value, description="Initial task status")
    message: str = Field(default="Task enqueued successfully", description="Human-readable confirmation")


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(..., description="Short error type or code")
    detail: str = Field(..., description="Human-readable error description")


class HealthResponse(BaseModel):
    """Response model for the health-check endpoint."""

    status: str = Field(default="ok", description="Service health status")
    version: str = Field(..., description="agent_bridge package version")


# ---------------------------------------------------------------------------
# Callback payload
# ---------------------------------------------------------------------------


class CallbackPayload(BaseModel):
    """Payload posted to the external callback URL on task completion."""

    task_id: str = Field(..., description="UUID of the completed task")
    platform: str = Field(..., description="Originating platform")
    chat_id: str = Field(..., description="Platform chat/channel ID")
    user_id: str = Field(..., description="Platform user ID")
    status: str = Field(..., description="Final task status (done or failed)")
    result: str | None = Field(default=None, description="AI result text, if available")
    error: str | None = Field(default=None, description="Error detail, if failed")
    completed_at: datetime = Field(
        default_factory=datetime.utcnow, description="UTC timestamp when task completed"
    )

    @classmethod
    def from_record(cls, record: TaskRecord) -> "CallbackPayload":
        """Construct a CallbackPayload from a completed or failed TaskRecord."""
        return cls(
            task_id=record.id,
            platform=record.platform.value if isinstance(record.platform, Platform) else record.platform,
            chat_id=record.chat_id,
            user_id=record.user_id,
            status=record.status.value if isinstance(record.status, TaskStatus) else record.status,
            result=record.result,
            error=record.error,
            completed_at=record.updated_at,
        )
