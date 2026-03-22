"""Async task queue for agent_bridge.

Provides an asyncio-based task queue that accepts AI jobs, executes them
concurrently up to a configurable limit, persists status transitions to
the database, and invokes completion callbacks when tasks finish.

Typical usage::

    queue = TaskQueue(settings=settings, db=db, messenger=messenger)
    await queue.start()

    task_id = await queue.enqueue(task_record)

    # Later, during application shutdown:
    await queue.stop()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import httpx

from agent_bridge.ai_client import AIClient, AIClientError
from agent_bridge.config import Settings
from agent_bridge.db import Database
from agent_bridge.models import (
    CallbackPayload,
    TaskRecord,
    TaskStatus,
    TaskStatusUpdate,
)

logger = logging.getLogger(__name__)

# Type alias for the completion callback signature.
# The callback receives the final TaskRecord (done or failed).
CompletionCallback = Callable[[TaskRecord], Awaitable[None]]


class TaskQueueError(Exception):
    """Raised when the task queue encounters an unrecoverable error."""


class TaskQueue:
    """Async task queue with concurrency control and completion callbacks.

    Jobs submitted via :meth:`enqueue` are placed in an internal asyncio
    queue. A pool of worker coroutines picks them up and executes them
    against the AI backend, limited to ``settings.max_concurrent_tasks``
    simultaneous executions. Upon completion (success or failure), the
    task's status is updated in the database and optional callbacks are fired.

    Args:
        settings: Application configuration.
        db: Initialised :class:`~agent_bridge.db.Database` instance.
        messenger: Optional async callable invoked with the completed
            :class:`~agent_bridge.models.TaskRecord` to deliver results back
            to the user. Signature: ``async def messenger(record: TaskRecord) -> None``.
        http_client: Optional :class:`httpx.AsyncClient` for dependency
            injection (AI client HTTP + callback HTTP). If ``None``, the
            queue creates and owns its own client.
    """

    def __init__(
        self,
        settings: Settings,
        db: Database,
        messenger: CompletionCallback | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._messenger = messenger
        self._http_client = http_client
        self._owns_http_client = http_client is None

        # Internal asyncio primitives – created in start()
        self._queue: asyncio.Queue[TaskRecord] = asyncio.Queue()
        self._semaphore: asyncio.Semaphore | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._additional_callbacks: list[CompletionCallback] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the queue's worker pool.

        Creates the concurrency semaphore and spawns worker coroutines.
        Call this once during application startup (e.g. in the FastAPI
        lifespan handler).

        Raises:
            TaskQueueError: If :meth:`start` is called while the queue is
                already running.
        """
        if self._running:
            raise TaskQueueError("TaskQueue is already running. Call stop() first.")

        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent_tasks)

        if self._owns_http_client:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self._settings.ai_timeout_seconds,
                    write=30.0,
                    pool=10.0,
                )
            )

        self._running = True
        num_workers = max(self._settings.max_concurrent_tasks, 1)
        for i in range(num_workers):
            worker = asyncio.create_task(
                self._worker_loop(), name=f"agent_bridge_worker_{i}"
            )
            self._workers.append(worker)

        logger.info(
            "TaskQueue started with %d workers (max_concurrent=%d)",
            num_workers,
            self._settings.max_concurrent_tasks,
        )

    async def stop(self, timeout: float = 30.0) -> None:
        """Gracefully stop the queue, waiting for in-flight tasks to complete.

        Signals all workers to shut down by sending sentinel values into the
        queue. Waits up to ``timeout`` seconds for workers to finish.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown before
                cancelling remaining workers.
        """
        if not self._running:
            return

        self._running = False
        logger.info("TaskQueue stopping – sending shutdown signals to %d workers", len(self._workers))

        # Send a sentinel ``None`` for each worker to unblock them
        for _ in self._workers:
            await self._queue.put(None)  # type: ignore[arg-type]

        # Wait for all workers to finish
        if self._workers:
            done, pending = await asyncio.wait(
                self._workers,
                timeout=timeout,
            )
            for task in pending:
                logger.warning("Worker %s did not stop in time – cancelling", task.get_name())
                task.cancel()

        self._workers.clear()

        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("TaskQueue stopped")

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def add_completion_callback(self, callback: CompletionCallback) -> None:
        """Register an additional callback to invoke on task completion.

        Callbacks are invoked after the task status is persisted to the
        database and after the messenger (if configured) has been called.
        Multiple callbacks are invoked in registration order.

        Args:
            callback: An async callable that receives the final
                :class:`~agent_bridge.models.TaskRecord`.
        """
        self._additional_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Enqueueing
    # ------------------------------------------------------------------

    async def enqueue(self, task: TaskRecord) -> None:
        """Add a task to the queue for asynchronous execution.

        The task's status must be :attr:`TaskStatus.PENDING`. After
        enqueueing, the task will be picked up by the next available worker.

        Args:
            task: A :class:`~agent_bridge.models.TaskRecord` to execute.

        Raises:
            TaskQueueError: If the queue has not been started.
            ValueError: If the task status is not PENDING.
        """
        if not self._running:
            raise TaskQueueError(
                "TaskQueue is not running. Call start() before enqueue()."
            )
        if task.status not in (TaskStatus.PENDING, "pending"):
            raise ValueError(
                f"Only PENDING tasks can be enqueued; got status={task.status!r}"
            )

        await self._queue.put(task)
        logger.debug(
            "Enqueued task %s (prompt_len=%d)", task.id, len(task.prompt)
        )

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Internal worker coroutine that processes tasks from the queue.

        Runs indefinitely until a sentinel ``None`` value is received from
        the queue, which signals that the worker should exit.
        """
        worker_name = asyncio.current_task().get_name() if asyncio.current_task() else "worker"
        logger.debug("%s started", worker_name)

        while True:
            try:
                task = await self._queue.get()
            except asyncio.CancelledError:
                logger.debug("%s cancelled while waiting for task", worker_name)
                break

            # Sentinel value signals shutdown
            if task is None:
                self._queue.task_done()
                logger.debug("%s received shutdown signal", worker_name)
                break

            logger.debug("%s picked up task %s", worker_name, task.id)
            try:
                await self._execute_task(task)
            except Exception:
                # _execute_task has its own error handling; log unexpected errors
                logger.exception(
                    "%s encountered unexpected error processing task %s",
                    worker_name,
                    task.id,
                )
            finally:
                self._queue.task_done()

        logger.debug("%s exiting", worker_name)

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def _execute_task(self, task: TaskRecord) -> None:
        """Execute a single AI task, updating status and firing callbacks.

        Acquires the concurrency semaphore before execution and releases it
        afterwards. Wraps the AI call in the task timeout.

        Args:
            task: The task to execute.
        """
        assert self._semaphore is not None  # always set after start()

        async with self._semaphore:
            # Mark the task as running in the database
            running_record = await self._db.update_task_status(
                task.id,
                TaskStatusUpdate(status=TaskStatus.RUNNING),
            )
            if running_record is None:
                logger.warning(
                    "Task %s disappeared from DB before execution", task.id
                )
                return

            logger.info(
                "Executing task %s (platform=%s, user=%s)",
                task.id,
                task.platform,
                task.user_id,
            )

            result_record: TaskRecord | None = None
            try:
                result_text = await asyncio.wait_for(
                    self._run_ai(task.prompt),
                    timeout=self._settings.task_timeout_seconds,
                )
                result_record = await self._db.update_task_status(
                    task.id,
                    TaskStatusUpdate(
                        status=TaskStatus.DONE,
                        result=result_text,
                    ),
                )
                logger.info("Task %s completed successfully", task.id)

            except asyncio.TimeoutError:
                error_msg = (
                    f"Task timed out after {self._settings.task_timeout_seconds}s"
                )
                logger.error("Task %s timed out", task.id)
                result_record = await self._db.update_task_status(
                    task.id,
                    TaskStatusUpdate(
                        status=TaskStatus.FAILED,
                        error=error_msg,
                    ),
                )

            except AIClientError as exc:
                error_msg = f"AI backend error: {exc}"
                logger.error("Task %s failed with AI error: %s", task.id, exc)
                result_record = await self._db.update_task_status(
                    task.id,
                    TaskStatusUpdate(
                        status=TaskStatus.FAILED,
                        error=error_msg,
                    ),
                )

            except Exception as exc:
                error_msg = f"Unexpected error: {exc}"
                logger.exception("Task %s failed with unexpected error", task.id)
                result_record = await self._db.update_task_status(
                    task.id,
                    TaskStatusUpdate(
                        status=TaskStatus.FAILED,
                        error=error_msg,
                    ),
                )

            finally:
                if result_record is not None:
                    await self._fire_callbacks(result_record)

    async def _run_ai(self, prompt: str) -> str:
        """Invoke the AI client to generate a completion for the given prompt.

        Reuses the shared HTTP client if available, otherwise creates a
        temporary AIClient.

        Args:
            prompt: The user's prompt text.

        Returns:
            The AI-generated response string.

        Raises:
            AIClientError: On any AI backend communication error.
        """
        ai_client = AIClient(
            settings=self._settings,
            http_client=self._http_client,
        )
        if self._http_client is not None:
            # Use the shared HTTP client directly – skip context manager lifecycle
            return await ai_client.complete(prompt)
        else:
            async with ai_client:
                return await ai_client.complete(prompt)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    async def _fire_callbacks(self, record: TaskRecord) -> None:
        """Invoke all registered completion callbacks for a finished task.

        Calls the messenger (if configured) and then any additional
        registered callbacks. Errors in individual callbacks are logged but
        do not propagate.

        Also sends the optional external webhook callback if configured in
        settings.

        Args:
            record: The final :class:`~agent_bridge.models.TaskRecord` with
                status DONE or FAILED.
        """
        # 1. Send result back via messenger
        if self._messenger is not None:
            try:
                await self._messenger(record)
            except Exception as exc:
                logger.error(
                    "Messenger callback failed for task %s: %s", record.id, exc
                )

        # 2. Fire any additional registered callbacks
        for callback in self._additional_callbacks:
            try:
                await callback(record)
            except Exception as exc:
                logger.error(
                    "Completion callback %s failed for task %s: %s",
                    getattr(callback, "__name__", repr(callback)),
                    record.id,
                    exc,
                )

        # 3. Send external webhook callback if configured
        if self._settings.callback_url:
            await self._send_external_callback(record)

    async def _send_external_callback(self, record: TaskRecord) -> None:
        """POST the task completion payload to the configured external callback URL.

        Args:
            record: The completed or failed task record.
        """
        if not self._settings.callback_url:
            return

        payload = CallbackPayload.from_record(record)
        headers: dict[str, str] = {"Content-Type": "application/json"}

        if self._settings.callback_secret is not None:
            headers["X-Callback-Secret"] = (
                self._settings.callback_secret.get_secret_value()
            )

        payload_json = payload.model_dump(mode="json", default=str)

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    self._settings.callback_url,
                    headers=headers,
                    json=payload_json,
                    timeout=10.0,
                )
            else:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        self._settings.callback_url,
                        headers=headers,
                        json=payload_json,
                    )

            if response.status_code >= 400:
                logger.warning(
                    "External callback for task %s returned HTTP %d",
                    record.id,
                    response.status_code,
                )
            else:
                logger.debug(
                    "External callback for task %s delivered (HTTP %d)",
                    record.id,
                    response.status_code,
                )
        except httpx.RequestError as exc:
            logger.error(
                "External callback request failed for task %s: %s", record.id, exc
            )
        except Exception as exc:
            logger.error(
                "Unexpected error sending external callback for task %s: %s",
                record.id,
                exc,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the queue's worker pool is active."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Return the number of tasks currently waiting in the queue."""
        return self._queue.qsize()
