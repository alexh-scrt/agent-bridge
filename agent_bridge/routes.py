"""HTTP route handlers for agent_bridge.

Defines all FastAPI endpoints including:
- Health check
- Telegram webhook ingestion
- Discord webhook ingestion
- Task status queries
- Manual callback triggers

All endpoints validate request authentication and user allowlists before
processing. Tasks are enqueued for async AI execution via the TaskQueue.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from agent_bridge import __version__
from agent_bridge.config import Settings
from agent_bridge.db import Database
from agent_bridge.messenger import Messenger, MessengerError
from agent_bridge.models import (
    CallbackPayload,
    DiscordInboundPayload,
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
    TelegramUpdate,
)
from agent_bridge.queue import TaskQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router instances
# ---------------------------------------------------------------------------

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def get_settings(request: Request) -> Settings:
    """Extract the Settings instance from the application state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The application :class:`~agent_bridge.config.Settings` instance.
    """
    return request.app.state.settings


def get_db(request: Request) -> Database:
    """Extract the Database instance from the application state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The application :class:`~agent_bridge.db.Database` instance.
    """
    return request.app.state.db


def get_queue(request: Request) -> TaskQueue:
    """Extract the TaskQueue instance from the application state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The application :class:`~agent_bridge.queue.TaskQueue` instance.
    """
    return request.app.state.queue


def get_messenger(request: Request) -> Messenger:
    """Extract the Messenger instance from the application state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The application :class:`~agent_bridge.messenger.Messenger` instance.
    """
    return request.app.state.messenger


def verify_secret_token(
    request: Request,
    x_secret_token: Annotated[str | None, Header()] = None,
) -> None:
    """Verify the X-Secret-Token header against the configured secret.

    This dependency raises HTTP 401 if the token is missing or incorrect.
    The check is skipped if the configured secret is the default 'changeme'
    value in order to allow easy local development (with a warning).

    Args:
        request: The incoming FastAPI request.
        x_secret_token: Value of the X-Secret-Token HTTP header.

    Raises:
        HTTPException: 401 if the token is missing or does not match.
    """
    settings: Settings = request.app.state.settings
    configured_secret = settings.secret_token.get_secret_value()

    # Skip verification in testing mode
    testing = getattr(request.app.state, "testing", False)
    if testing:
        return

    if configured_secret == "changeme":
        # Warn but allow through during development
        logger.warning(
            "SECRET_TOKEN is set to the default 'changeme'. "
            "Set a strong secret in production."
        )
        return

    if x_secret_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Secret-Token header",
        )

    if x_secret_token != configured_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Secret-Token",
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["system"],
)
async def health_check() -> HealthResponse:
    """Return the service health status and version.

    Returns:
        A :class:`~agent_bridge.models.HealthResponse` with ``status='ok'``.
    """
    return HealthResponse(status="ok", version=__version__)


# ---------------------------------------------------------------------------
# Telegram webhook
# ---------------------------------------------------------------------------


@router.post(
    "/telegram/webhook",
    summary="Telegram webhook endpoint",
    tags=["telegram"],
    status_code=status.HTTP_200_OK,
)
async def telegram_webhook(
    update: TelegramUpdate,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[Database, Depends(get_db)],
    queue: Annotated[TaskQueue, Depends(get_queue)],
    messenger: Annotated[Messenger, Depends(get_messenger)],
) -> dict[str, Any]:
    """Ingest an update from the Telegram Bot API.

    Validates that the sender is in the allowed users list (if configured),
    extracts the message text, and enqueues an AI task. Sends an
    acknowledgement back to the user's chat immediately.

    Special commands:
    - ``/status <task_id>``: Query the status of a specific task.
    - ``/tasks``: List recent tasks for the current chat.

    Args:
        update: Parsed Telegram update payload.
        request: The raw FastAPI request (used for state access).
        settings: Application settings.
        db: Task database.
        queue: The async task queue.
        messenger: The messenger abstraction for sending replies.

    Returns:
        A ``{"ok": true}`` JSON response on success.

    Raises:
        HTTPException: 400 if the update contains no usable message.
    """
    logger.debug("Received Telegram update_id=%d", update.update_id)

    effective_message = update.effective_message
    if effective_message is None:
        # Telegram sends non-message updates (e.g. callback queries); ignore them
        return {"ok": True, "ignored": True}

    sender_id = update.sender_id
    chat_id = update.chat_id
    text = update.text

    if chat_id is None:
        return {"ok": True, "ignored": True}

    # Check user allowlist
    if settings.telegram_allowed_users and sender_id not in settings.telegram_allowed_users:
        logger.warning(
            "Telegram user %s is not in the allowlist – rejecting", sender_id
        )
        try:
            await messenger.send_not_authorised(
                platform=Platform.TELEGRAM,
                chat_id=str(chat_id),
            )
        except MessengerError as exc:
            logger.error("Failed to send not-authorised message: %s", exc)
        return {"ok": True, "ignored": True}

    if not text or not text.strip():
        logger.debug("Telegram update has no text – ignoring")
        return {"ok": True, "ignored": True}

    text = text.strip()

    # ------------------------------------------------------------------ #
    # Handle special commands
    # ------------------------------------------------------------------ #
    if text.startswith("/status"):
        return await _handle_telegram_status_command(
            text=text,
            chat_id=str(chat_id),
            db=db,
            messenger=messenger,
        )

    if text.startswith("/tasks"):
        return await _handle_telegram_tasks_command(
            chat_id=str(chat_id),
            user_id=str(sender_id),
            db=db,
            messenger=messenger,
        )

    if text.startswith("/start") or text.startswith("/help"):
        help_text = (
            "\U0001f916 *agent_bridge* AI Coding Assistant\n\n"
            "Send me any coding question or task and I'll get back to you "
            "with an AI-generated response.\n\n"
            "*Commands:*\n"
            "`/status <task_id>` – Check the status of a task\n"
            "`/tasks` – List your recent tasks\n"
            "`/help` – Show this message"
        )
        try:
            await messenger.send_telegram(chat_id=str(chat_id), text=help_text)
        except MessengerError as exc:
            logger.error("Failed to send help message: %s", exc)
        return {"ok": True}

    # ------------------------------------------------------------------ #
    # Enqueue the AI task
    # ------------------------------------------------------------------ #
    task_create = TaskCreate(
        platform=Platform.TELEGRAM,
        chat_id=str(chat_id),
        user_id=str(sender_id) if sender_id is not None else "unknown",
        prompt=text,
    )

    try:
        task_id = await db.create_task(task_create)
        task_record = await db.get_task(task_id)
        assert task_record is not None

        await queue.enqueue(task_record)
        logger.info(
            "Enqueued Telegram task %s for user %s in chat %s",
            task_id,
            sender_id,
            chat_id,
        )
    except Exception as exc:
        logger.error("Failed to enqueue Telegram task: %s", exc)
        try:
            await messenger.send_error(
                platform=Platform.TELEGRAM,
                chat_id=str(chat_id),
                error_message="Failed to queue your request. Please try again.",
            )
        except MessengerError:
            pass
        return {"ok": True, "error": str(exc)}

    # Send acknowledgement
    try:
        await messenger.send_ack(
            platform=Platform.TELEGRAM,
            chat_id=str(chat_id),
            task_id=task_id,
        )
    except MessengerError as exc:
        logger.error("Failed to send Telegram ack for task %s: %s", task_id, exc)

    return {"ok": True, "task_id": task_id}


async def _handle_telegram_status_command(
    text: str,
    chat_id: str,
    db: Database,
    messenger: Messenger,
) -> dict[str, Any]:
    """Handle the ``/status <task_id>`` command for Telegram.

    Args:
        text: The full command text (e.g. ``/status abc123``).
        chat_id: The Telegram chat ID to reply to.
        db: The task database.
        messenger: The messenger for sending the reply.

    Returns:
        A ``{"ok": True}`` response dict.
    """
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        try:
            await messenger.send_telegram(
                chat_id=chat_id,
                text="Usage: `/status <task_id>`",
            )
        except MessengerError as exc:
            logger.error("Failed to send status usage hint: %s", exc)
        return {"ok": True}

    task_id = parts[1].strip()
    record = await db.get_task(task_id)

    if record is None:
        try:
            await messenger.send_telegram(
                chat_id=chat_id,
                text=f"\u274c Task `{task_id[:8]}` not found.",
            )
        except MessengerError as exc:
            logger.error("Failed to send task not found message: %s", exc)
        return {"ok": True}

    try:
        await messenger.send_status(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            record=record,
        )
    except MessengerError as exc:
        logger.error("Failed to send task status: %s", exc)

    return {"ok": True}


async def _handle_telegram_tasks_command(
    chat_id: str,
    user_id: str,
    db: Database,
    messenger: Messenger,
) -> dict[str, Any]:
    """Handle the ``/tasks`` command for Telegram, listing recent tasks.

    Args:
        chat_id: The Telegram chat ID.
        user_id: The Telegram user ID.
        db: The task database.
        messenger: The messenger for sending the reply.

    Returns:
        A ``{"ok": True}`` response dict.
    """
    tasks = await db.list_tasks(chat_id=chat_id, limit=5)

    if not tasks:
        try:
            await messenger.send_telegram(
                chat_id=chat_id,
                text="You have no recent tasks.",
            )
        except MessengerError as exc:
            logger.error("Failed to send tasks list: %s", exc)
        return {"ok": True}

    lines = ["*Your recent tasks:*\n"]
    for record in tasks:
        status_icon = {
            TaskStatus.PENDING: "\u23f3",
            TaskStatus.RUNNING: "\u26a1",
            TaskStatus.DONE: "\u2705",
            TaskStatus.FAILED: "\u274c",
        }.get(record.status, "\u2754")
        short_id = record.id[:8]
        prompt_preview = record.prompt[:40] + "..." if len(record.prompt) > 40 else record.prompt
        lines.append(f"{status_icon} `{short_id}` – {prompt_preview}")

    message = "\n".join(lines)
    try:
        await messenger.send_telegram(chat_id=chat_id, text=message)
    except MessengerError as exc:
        logger.error("Failed to send tasks list: %s", exc)

    return {"ok": True}


# ---------------------------------------------------------------------------
# Discord webhook / ingestion
# ---------------------------------------------------------------------------


@router.post(
    "/discord/webhook",
    summary="Discord message ingestion endpoint",
    tags=["discord"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_secret_token)],
)
async def discord_webhook(
    payload: DiscordInboundPayload,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[Database, Depends(get_db)],
    queue: Annotated[TaskQueue, Depends(get_queue)],
    messenger: Annotated[Messenger, Depends(get_messenger)],
) -> dict[str, Any]:
    """Ingest a Discord message from the agent_bridge Discord ingestion endpoint.

    Unlike the Telegram webhook (which is called directly by Telegram's servers),
    this endpoint is called by the agent_bridge Discord bot component (or any
    intermediary) after filtering gateway events. It validates the user allowlist,
    extracts the message content, and enqueues an AI task.

    Special commands (prefix ``/``):
    - ``/status <task_id>``: Query a specific task.
    - ``/tasks``: List recent tasks.

    Args:
        payload: The Discord message payload.
        request: The raw FastAPI request.
        settings: Application settings.
        db: Task database.
        queue: The async task queue.
        messenger: The messenger abstraction.

    Returns:
        A ``{"ok": true}`` JSON response on success.
    """
    message = payload.message
    author = message.author

    logger.debug(
        "Received Discord message id=%s from user=%s",
        message.id,
        author.id,
    )

    # Ignore bot messages
    if author.bot:
        return {"ok": True, "ignored": True}

    # Check user allowlist
    if settings.discord_allowed_users and author.id_int not in settings.discord_allowed_users:
        logger.warning(
            "Discord user %s is not in the allowlist – rejecting", author.id
        )
        try:
            await messenger.send_not_authorised(
                platform=Platform.DISCORD,
                chat_id=message.channel_id,
            )
        except MessengerError as exc:
            logger.error("Failed to send not-authorised message to Discord: %s", exc)
        return {"ok": True, "ignored": True}

    # Check guild allowlist
    if (
        settings.discord_allowed_guilds
        and message.guild_id is not None
        and int(message.guild_id) not in settings.discord_allowed_guilds
    ):
        logger.warning(
            "Discord guild %s is not in the allowlist – rejecting", message.guild_id
        )
        return {"ok": True, "ignored": True}

    content = message.content.strip()
    if not content:
        return {"ok": True, "ignored": True}

    # ------------------------------------------------------------------ #
    # Handle special commands
    # ------------------------------------------------------------------ #
    if content.startswith("/status"):
        return await _handle_discord_status_command(
            text=content,
            channel_id=message.channel_id,
            db=db,
            messenger=messenger,
        )

    if content.startswith("/tasks"):
        return await _handle_discord_tasks_command(
            channel_id=message.channel_id,
            db=db,
            messenger=messenger,
        )

    if content.startswith("/help"):
        help_text = (
            "\U0001f916 **agent_bridge** AI Coding Assistant\n\n"
            "Send me any coding question or task and I'll get back to you "
            "with an AI-generated response.\n\n"
            "**Commands:**\n"
            "`/status <task_id>` – Check the status of a task\n"
            "`/tasks` – List your recent tasks\n"
            "`/help` – Show this message"
        )
        try:
            await messenger.send_discord(channel_id=message.channel_id, text=help_text)
        except MessengerError as exc:
            logger.error("Failed to send Discord help message: %s", exc)
        return {"ok": True}

    # ------------------------------------------------------------------ #
    # Enqueue the AI task
    # ------------------------------------------------------------------ #
    task_create = TaskCreate(
        platform=Platform.DISCORD,
        chat_id=message.channel_id,
        user_id=author.id,
        prompt=content,
    )

    try:
        task_id = await db.create_task(task_create)
        task_record = await db.get_task(task_id)
        assert task_record is not None

        await queue.enqueue(task_record)
        logger.info(
            "Enqueued Discord task %s for user %s in channel %s",
            task_id,
            author.id,
            message.channel_id,
        )
    except Exception as exc:
        logger.error("Failed to enqueue Discord task: %s", exc)
        try:
            await messenger.send_error(
                platform=Platform.DISCORD,
                chat_id=message.channel_id,
                error_message="Failed to queue your request. Please try again.",
            )
        except MessengerError:
            pass
        return {"ok": True, "error": str(exc)}

    # Send acknowledgement
    try:
        await messenger.send_ack(
            platform=Platform.DISCORD,
            chat_id=message.channel_id,
            task_id=task_id,
        )
    except MessengerError as exc:
        logger.error("Failed to send Discord ack for task %s: %s", task_id, exc)

    return {"ok": True, "task_id": task_id}


async def _handle_discord_status_command(
    text: str,
    channel_id: str,
    db: Database,
    messenger: Messenger,
) -> dict[str, Any]:
    """Handle the ``/status <task_id>`` command for Discord.

    Args:
        text: The full command text.
        channel_id: The Discord channel ID to reply to.
        db: The task database.
        messenger: The messenger for sending the reply.

    Returns:
        A ``{"ok": True}`` response dict.
    """
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        try:
            await messenger.send_discord(
                channel_id=channel_id,
                text="Usage: `/status <task_id>`",
            )
        except MessengerError as exc:
            logger.error("Failed to send Discord status usage hint: %s", exc)
        return {"ok": True}

    task_id = parts[1].strip()
    record = await db.get_task(task_id)

    if record is None:
        try:
            await messenger.send_discord(
                channel_id=channel_id,
                text=f"\u274c Task `{task_id[:8]}` not found.",
            )
        except MessengerError as exc:
            logger.error("Failed to send Discord task-not-found message: %s", exc)
        return {"ok": True}

    try:
        await messenger.send_status(
            platform=Platform.DISCORD,
            chat_id=channel_id,
            record=record,
        )
    except MessengerError as exc:
        logger.error("Failed to send Discord task status: %s", exc)

    return {"ok": True}


async def _handle_discord_tasks_command(
    channel_id: str,
    db: Database,
    messenger: Messenger,
) -> dict[str, Any]:
    """Handle the ``/tasks`` command for Discord.

    Args:
        channel_id: The Discord channel ID.
        db: The task database.
        messenger: The messenger for sending the reply.

    Returns:
        A ``{"ok": True}`` response dict.
    """
    tasks = await db.list_tasks(chat_id=channel_id, limit=5)

    if not tasks:
        try:
            await messenger.send_discord(
                channel_id=channel_id,
                text="No recent tasks found for this channel.",
            )
        except MessengerError as exc:
            logger.error("Failed to send Discord tasks list: %s", exc)
        return {"ok": True}

    lines = ["**Recent tasks:**\n"]
    for record in tasks:
        status_icon = {
            TaskStatus.PENDING: "\u23f3",
            TaskStatus.RUNNING: "\u26a1",
            TaskStatus.DONE: "\u2705",
            TaskStatus.FAILED: "\u274c",
        }.get(record.status, "\u2754")
        short_id = record.id[:8]
        prompt_preview = record.prompt[:40] + "..." if len(record.prompt) > 40 else record.prompt
        lines.append(f"{status_icon} `{short_id}` – {prompt_preview}")

    message = "\n".join(lines)
    try:
        await messenger.send_discord(channel_id=channel_id, text=message)
    except MessengerError as exc:
        logger.error("Failed to send Discord tasks list: %s", exc)

    return {"ok": True}


# ---------------------------------------------------------------------------
# Task REST API
# ---------------------------------------------------------------------------


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="List tasks",
    tags=["tasks"],
    dependencies=[Depends(verify_secret_token)],
)
async def list_tasks(
    db: Annotated[Database, Depends(get_db)],
    platform: str | None = Query(default=None, description="Filter by platform"),
    chat_id: str | None = Query(default=None, description="Filter by chat ID"),
    user_id: str | None = Query(default=None, description="Filter by user ID"),
    task_status: str | None = Query(
        default=None, alias="status", description="Filter by status"
    ),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
) -> TaskListResponse:
    """Return a paginated, filtered list of task records.

    All query parameters are optional. Results are ordered from newest to oldest.

    Args:
        db: The task database (injected).
        platform: Optional platform filter (``telegram`` or ``discord``).
        chat_id: Optional chat/channel ID filter.
        user_id: Optional user ID filter.
        task_status: Optional status filter (``pending``, ``running``, ``done``, ``failed``).
        limit: Maximum number of records to return.
        offset: Number of records to skip.

    Returns:
        A :class:`~agent_bridge.models.TaskListResponse` with matching records.

    Raises:
        HTTPException: 400 if an invalid platform or status value is provided.
    """
    platform_enum: Platform | None = None
    if platform is not None:
        try:
            platform_enum = Platform(platform.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform '{platform}'. Must be 'telegram' or 'discord'.",
            )

    status_enum: TaskStatus | None = None
    if task_status is not None:
        try:
            status_enum = TaskStatus(task_status.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid status '{task_status}'. "
                    "Must be one of: pending, running, done, failed."
                ),
            )

    tasks = await db.list_tasks(
        platform=platform_enum,
        chat_id=chat_id,
        user_id=user_id,
        status=status_enum,
        limit=limit,
        offset=offset,
    )
    total = await db.count_tasks(
        platform=platform_enum,
        chat_id=chat_id,
        user_id=user_id,
        status=status_enum,
    )

    return TaskListResponse(
        tasks=[TaskResponse.from_record(r) for r in tasks],
        total=total,
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    summary="Get a single task",
    tags=["tasks"],
    dependencies=[Depends(verify_secret_token)],
)
async def get_task(
    task_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> TaskResponse:
    """Retrieve a single task by its UUID.

    Args:
        task_id: The UUID of the task to retrieve.
        db: The task database (injected).

    Returns:
        A :class:`~agent_bridge.models.TaskResponse` for the requested task.

    Raises:
        HTTPException: 404 if the task is not found.
    """
    record = await db.get_task(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    return TaskResponse.from_record(record)


@router.delete(
    "/tasks/{task_id}",
    summary="Delete a task",
    tags=["tasks"],
    dependencies=[Depends(verify_secret_token)],
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_task(
    task_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> None:
    """Delete a task record by its UUID.

    Args:
        task_id: The UUID of the task to delete.
        db: The task database (injected).

    Raises:
        HTTPException: 404 if the task is not found.
    """
    deleted = await db.delete_task(task_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )


# ---------------------------------------------------------------------------
# Manual callback trigger
# ---------------------------------------------------------------------------


@router.post(
    "/tasks/{task_id}/deliver",
    summary="Manually re-deliver a completed task result",
    tags=["tasks"],
    dependencies=[Depends(verify_secret_token)],
    status_code=status.HTTP_200_OK,
)
async def deliver_task_result(
    task_id: str,
    db: Annotated[Database, Depends(get_db)],
    messenger: Annotated[Messenger, Depends(get_messenger)],
) -> dict[str, Any]:
    """Re-send the result of a completed or failed task to the originating chat.

    This is useful if the initial delivery failed (e.g. the bot was offline)
    or for manual re-triggering during development.

    Args:
        task_id: The UUID of the task whose result should be delivered.
        db: The task database (injected).
        messenger: The messenger abstraction (injected).

    Returns:
        A ``{"ok": true, "task_id": task_id, "status": ...}`` JSON response.

    Raises:
        HTTPException: 404 if the task is not found.
        HTTPException: 409 if the task is still pending or running.
        HTTPException: 502 if the messenger fails to deliver the result.
    """
    record = await db.get_task(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    if record.status not in (TaskStatus.DONE, TaskStatus.FAILED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Task '{task_id}' has status '{record.status.value}'. "
                "Only 'done' or 'failed' tasks can be re-delivered."
            ),
        )

    try:
        await messenger.send(record)
        logger.info("Manually delivered result for task %s", task_id)
    except MessengerError as exc:
        logger.error("Manual delivery failed for task %s: %s", task_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to deliver task result: {exc}",
        ) from exc

    return {
        "ok": True,
        "task_id": task_id,
        "status": record.status.value if hasattr(record.status, "value") else record.status,
    }


@router.post(
    "/tasks/{task_id}/callback",
    summary="Manually trigger the external callback for a task",
    tags=["tasks"],
    dependencies=[Depends(verify_secret_token)],
    status_code=status.HTTP_200_OK,
)
async def trigger_callback(
    task_id: str,
    request: Request,
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, Any]:
    """Manually POST the task completion payload to the configured callback URL.

    This is useful for re-triggering webhook notifications if the initial
    delivery failed.

    Args:
        task_id: The UUID of the task to send the callback for.
        request: The raw FastAPI request (for accessing app state).
        db: The task database (injected).

    Returns:
        A ``{"ok": true, "task_id": task_id}`` JSON response.

    Raises:
        HTTPException: 404 if the task is not found.
        HTTPException: 409 if the task is not in a terminal state.
        HTTPException: 503 if no callback URL is configured.
        HTTPException: 502 if the callback delivery fails.
    """
    settings: Settings = request.app.state.settings
    queue: TaskQueue = request.app.state.queue

    if not settings.callback_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No CALLBACK_URL is configured.",
        )

    record = await db.get_task(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    if record.status not in (TaskStatus.DONE, TaskStatus.FAILED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Task '{task_id}' has status '{record.status.value if hasattr(record.status, 'value') else record.status}'. "
                "Only 'done' or 'failed' tasks can trigger a callback."
            ),
        )

    try:
        await queue._send_external_callback(record)
        logger.info("Manually triggered callback for task %s", task_id)
    except Exception as exc:
        logger.error("Manual callback failed for task %s: %s", task_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Callback delivery failed: {exc}",
        ) from exc

    return {"ok": True, "task_id": task_id}
