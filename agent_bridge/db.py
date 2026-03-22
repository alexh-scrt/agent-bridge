"""SQLite-backed task store for agent_bridge.

Provides a fully async database layer using aiosqlite for persisting task
history, status updates, and results. All operations are wrapped in a
:class:`Database` class that manages the connection lifecycle.

Typical usage::

    db = Database("agent_bridge.db")
    await db.initialize()

    task_id = await db.create_task(TaskCreate(...))
    record = await db.get_task(task_id)
    await db.update_task_status(task_id, TaskStatusUpdate(status=TaskStatus.DONE, result="..."))

    await db.close()
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import aiosqlite

from agent_bridge.models import (
    Platform,
    TaskCreate,
    TaskRecord,
    TaskStatus,
    TaskStatusUpdate,
)

logger = logging.getLogger(__name__)

# SQLite timestamp format used for storage and retrieval
_DT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
_DT_FORMAT_SHORT = "%Y-%m-%d %H:%M:%S"

_CREATE_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    platform    TEXT NOT NULL,
    chat_id     TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    result      TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

_CREATE_IDX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
"""

_CREATE_IDX_PLATFORM = """
CREATE INDEX IF NOT EXISTS idx_tasks_platform ON tasks (platform);
"""

_CREATE_IDX_CHAT = """
CREATE INDEX IF NOT EXISTS idx_tasks_chat_id ON tasks (chat_id);
"""


def _parse_dt(value: str) -> datetime:
    """Parse a datetime string stored in SQLite, handling both formats.

    Args:
        value: ISO-like datetime string from the database.

    Returns:
        A :class:`datetime` object.

    Raises:
        ValueError: If the string cannot be parsed in any expected format.
    """
    for fmt in (_DT_FORMAT, _DT_FORMAT_SHORT):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # Fallback: try fromisoformat (Python 3.11+)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"Cannot parse datetime string: {value!r}")


def _format_dt(dt: datetime) -> str:
    """Format a datetime for SQLite storage.

    Args:
        dt: The datetime to format.

    Returns:
        ISO-like string with microseconds.
    """
    return dt.strftime(_DT_FORMAT)


def _row_to_task_record(row: aiosqlite.Row) -> TaskRecord:
    """Convert a raw SQLite row into a :class:`TaskRecord`.

    Args:
        row: A row fetched from the ``tasks`` table.

    Returns:
        A fully populated :class:`TaskRecord` instance.
    """
    return TaskRecord(
        id=row["id"],
        platform=Platform(row["platform"]),
        chat_id=row["chat_id"],
        user_id=row["user_id"],
        prompt=row["prompt"],
        status=TaskStatus(row["status"]),
        result=row["result"],
        error=row["error"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


class Database:
    """Async SQLite task store backed by aiosqlite.

    Manages the full lifecycle of task records: creation, status updates,
    retrieval by ID, listing with optional filters, and deletion.

    Args:
        db_path: Path to the SQLite database file. Use ``":memory:"`` for
            an ephemeral in-process database (useful in tests).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the database connection and create the schema if needed.

        This method is idempotent – calling it multiple times is safe.

        Raises:
            aiosqlite.Error: If the database file cannot be opened or the
                schema cannot be created.
        """
        if self._conn is not None:
            return

        logger.debug("Opening database connection: %s", self._db_path)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrent read performance
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")

        # Create schema
        await self._conn.execute(_CREATE_TASKS_TABLE)
        await self._conn.execute(_CREATE_IDX_STATUS)
        await self._conn.execute(_CREATE_IDX_PLATFORM)
        await self._conn.execute(_CREATE_IDX_CHAT)
        await self._conn.commit()

        logger.info("Database initialised at %s", self._db_path)

    async def close(self) -> None:
        """Close the underlying database connection.

        Safe to call even if :meth:`initialize` has not been called.
        """
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    def _require_connection(self) -> aiosqlite.Connection:
        """Return the active connection or raise if not initialised.

        Returns:
            The active :class:`aiosqlite.Connection`.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._conn is None:
            raise RuntimeError(
                "Database is not initialised. Call await db.initialize() first."
            )
        return self._conn

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def create_task(self, task_create: TaskCreate) -> str:
        """Persist a new task and return its generated UUID.

        The task is created with :attr:`TaskStatus.PENDING` status.

        Args:
            task_create: Validated input data for the new task.

        Returns:
            The UUID string assigned to the new task.

        Raises:
            RuntimeError: If the database is not initialised.
            aiosqlite.Error: On any database error.
        """
        conn = self._require_connection()
        task_id = str(uuid.uuid4())
        now = _format_dt(datetime.utcnow())
        platform_value = (
            task_create.platform.value
            if isinstance(task_create.platform, Platform)
            else task_create.platform
        )

        await conn.execute(
            """
            INSERT INTO tasks (id, platform, chat_id, user_id, prompt, status, result, error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
            """,
            (
                task_id,
                platform_value,
                task_create.chat_id,
                task_create.user_id,
                task_create.prompt,
                TaskStatus.PENDING.value,
                now,
                now,
            ),
        )
        await conn.commit()
        logger.debug("Created task %s for user %s", task_id, task_create.user_id)
        return task_id

    async def get_task(self, task_id: str) -> TaskRecord | None:
        """Retrieve a single task by its UUID.

        Args:
            task_id: The UUID of the task to retrieve.

        Returns:
            A :class:`TaskRecord` if found, or ``None`` if no such task exists.

        Raises:
            RuntimeError: If the database is not initialised.
            aiosqlite.Error: On any database error.
        """
        conn = self._require_connection()
        async with conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None
        return _row_to_task_record(row)

    async def update_task_status(
        self, task_id: str, update: TaskStatusUpdate
    ) -> TaskRecord | None:
        """Apply a status update to an existing task.

        Updates ``status``, ``result`` (if provided), ``error`` (if provided),
        and ``updated_at`` timestamp.

        Args:
            task_id: UUID of the task to update.
            update: The new status and optional result/error.

        Returns:
            The updated :class:`TaskRecord`, or ``None`` if the task was not found.

        Raises:
            RuntimeError: If the database is not initialised.
            aiosqlite.Error: On any database error.
        """
        conn = self._require_connection()
        now = _format_dt(datetime.utcnow())
        status_value = (
            update.status.value
            if isinstance(update.status, TaskStatus)
            else update.status
        )

        await conn.execute(
            """
            UPDATE tasks
               SET status     = ?,
                   result     = ?,
                   error      = ?,
                   updated_at = ?
             WHERE id = ?
            """,
            (status_value, update.result, update.error, now, task_id),
        )
        await conn.commit()

        updated = await self.get_task(task_id)
        if updated is None:
            logger.warning("update_task_status: task %s not found", task_id)
        else:
            logger.debug(
                "Task %s status -> %s", task_id, status_value
            )
        return updated

    async def list_tasks(
        self,
        *,
        platform: Platform | None = None,
        chat_id: str | None = None,
        user_id: str | None = None,
        status: TaskStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TaskRecord]:
        """Return a filtered, paginated list of tasks.

        All filter parameters are optional; omitting them returns all tasks.
        Results are ordered from newest to oldest (``created_at DESC``).

        Args:
            platform: Filter by originating platform.
            chat_id: Filter by chat/channel ID.
            user_id: Filter by user ID.
            status: Filter by task status.
            limit: Maximum number of records to return (default 50).
            offset: Number of records to skip for pagination (default 0).

        Returns:
            A list of :class:`TaskRecord` instances.

        Raises:
            RuntimeError: If the database is not initialised.
            aiosqlite.Error: On any database error.
        """
        conn = self._require_connection()
        conditions: list[str] = []
        params: list[Any] = []

        if platform is not None:
            platform_value = platform.value if isinstance(platform, Platform) else platform
            conditions.append("platform = ?")
            params.append(platform_value)

        if chat_id is not None:
            conditions.append("chat_id = ?")
            params.append(chat_id)

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        if status is not None:
            status_value = status.value if isinstance(status, TaskStatus) else status
            conditions.append("status = ?")
            params.append(status_value)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"""
            SELECT * FROM tasks
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [_row_to_task_record(row) for row in rows]

    async def count_tasks(
        self,
        *,
        platform: Platform | None = None,
        chat_id: str | None = None,
        user_id: str | None = None,
        status: TaskStatus | None = None,
    ) -> int:
        """Return the total count of tasks matching the given filters.

        Accepts the same filter parameters as :meth:`list_tasks`.

        Args:
            platform: Filter by originating platform.
            chat_id: Filter by chat/channel ID.
            user_id: Filter by user ID.
            status: Filter by task status.

        Returns:
            Integer count of matching tasks.

        Raises:
            RuntimeError: If the database is not initialised.
            aiosqlite.Error: On any database error.
        """
        conn = self._require_connection()
        conditions: list[str] = []
        params: list[Any] = []

        if platform is not None:
            platform_value = platform.value if isinstance(platform, Platform) else platform
            conditions.append("platform = ?")
            params.append(platform_value)

        if chat_id is not None:
            conditions.append("chat_id = ?")
            params.append(chat_id)

        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        if status is not None:
            status_value = status.value if isinstance(status, TaskStatus) else status
            conditions.append("status = ?")
            params.append(status_value)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT COUNT(*) FROM tasks {where_clause}"

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()

        return int(row[0]) if row else 0

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task record by its UUID.

        Args:
            task_id: UUID of the task to delete.

        Returns:
            ``True`` if a row was deleted, ``False`` if the task was not found.

        Raises:
            RuntimeError: If the database is not initialised.
            aiosqlite.Error: On any database error.
        """
        conn = self._require_connection()
        async with conn.execute(
            "DELETE FROM tasks WHERE id = ?", (task_id,)
        ) as cursor:
            deleted = cursor.rowcount > 0
        await conn.commit()

        if deleted:
            logger.debug("Deleted task %s", task_id)
        else:
            logger.debug("delete_task: task %s not found", task_id)

        return deleted

    async def delete_tasks_by_status(self, status: TaskStatus) -> int:
        """Delete all tasks with a given status.

        Useful for housekeeping – for example, purging old ``done`` or
        ``failed`` records.

        Args:
            status: The status of tasks to delete.

        Returns:
            The number of rows deleted.

        Raises:
            RuntimeError: If the database is not initialised.
            aiosqlite.Error: On any database error.
        """
        conn = self._require_connection()
        status_value = status.value if isinstance(status, TaskStatus) else status
        async with conn.execute(
            "DELETE FROM tasks WHERE status = ?", (status_value,)
        ) as cursor:
            count = cursor.rowcount
        await conn.commit()
        logger.info("Purged %d tasks with status '%s'", count, status_value)
        return count

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "Database":
        """Support usage as an async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the connection when exiting the context manager."""
        await self.close()
