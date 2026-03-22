"""Tests for agent_bridge.queue – async task queue, concurrency, and callbacks."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from agent_bridge.ai_client import AIClientError
from agent_bridge.config import Settings
from agent_bridge.db import Database
from agent_bridge.models import (
    Platform,
    TaskCreate,
    TaskRecord,
    TaskStatus,
    TaskStatusUpdate,
)
from agent_bridge.queue import TaskQueue, TaskQueueError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from agent_bridge.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Return test settings with a mock AI backend."""
    with patch.dict(
        os.environ,
        {
            "AGENT_BRIDGE_TESTING": "1",
            "AI_BACKEND_TYPE": "ollama",
            "AI_BACKEND_URL": "http://localhost:11434",
            "AI_MODEL": "llama3",
            "AI_MAX_TOKENS": "100",
            "AI_TEMPERATURE": "0.2",
            "AI_TIMEOUT_SECONDS": "10",
            "MAX_CONCURRENT_TASKS": "3",
            "TASK_TIMEOUT_SECONDS": "30",
        },
    ):
        yield Settings()


@pytest.fixture
async def db() -> Database:
    """Return an in-memory database for each test."""
    async with Database(":memory:") as database:
        yield database


@pytest.fixture
async def queue(settings: Settings, db: Database) -> TaskQueue:
    """Return a started TaskQueue with a mocked AI backend."""
    tq = TaskQueue(settings=settings, db=db)
    await tq.start()
    yield tq
    await tq.stop(timeout=5.0)


def _make_task_record(
    task_id: str = "test-task-1",
    platform: Platform = Platform.TELEGRAM,
    chat_id: str = "100",
    user_id: str = "42",
    prompt: str = "Write hello world",
    status: TaskStatus = TaskStatus.PENDING,
) -> TaskRecord:
    """Create a TaskRecord for testing."""
    return TaskRecord(
        id=task_id,
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        prompt=prompt,
        status=status,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


async def _create_and_get_task(db: Database, prompt: str = "Test prompt") -> TaskRecord:
    """Helper: create a task in the DB and return the TaskRecord."""
    tc = TaskCreate(
        platform=Platform.TELEGRAM,
        chat_id="100",
        user_id="42",
        prompt=prompt,
    )
    task_id = await db.create_task(tc)
    record = await db.get_task(task_id)
    assert record is not None
    return record


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


async def test_queue_starts_and_stops(settings: Settings, db: Database):
    tq = TaskQueue(settings=settings, db=db)
    assert tq.is_running is False
    await tq.start()
    assert tq.is_running is True
    await tq.stop(timeout=2.0)
    assert tq.is_running is False


async def test_start_twice_raises(settings: Settings, db: Database):
    tq = TaskQueue(settings=settings, db=db)
    await tq.start()
    try:
        with pytest.raises(TaskQueueError, match="already running"):
            await tq.start()
    finally:
        await tq.stop(timeout=2.0)


async def test_stop_without_start_is_safe(settings: Settings, db: Database):
    tq = TaskQueue(settings=settings, db=db)
    await tq.stop()  # must not raise


async def test_is_running_property(settings: Settings, db: Database):
    tq = TaskQueue(settings=settings, db=db)
    assert tq.is_running is False
    await tq.start()
    assert tq.is_running is True
    await tq.stop(timeout=2.0)
    assert tq.is_running is False


# ---------------------------------------------------------------------------
# Enqueue tests
# ---------------------------------------------------------------------------


async def test_enqueue_before_start_raises(settings: Settings, db: Database):
    tq = TaskQueue(settings=settings, db=db)
    task = _make_task_record()
    with pytest.raises(TaskQueueError, match="not running"):
        await tq.enqueue(task)


async def test_enqueue_non_pending_raises(settings: Settings, db: Database):
    tq = TaskQueue(settings=settings, db=db)
    await tq.start()
    try:
        task = _make_task_record(status=TaskStatus.RUNNING)
        with pytest.raises(ValueError, match="PENDING"):
            await tq.enqueue(task)
    finally:
        await tq.stop(timeout=2.0)


async def test_queue_size_increases_on_enqueue(settings: Settings, db: Database):
    """Verify queue_size reflects pending items (before workers consume them)."""
    # Use a very slow AI backend so items stay in the queue
    tq = TaskQueue(settings=settings, db=db)
    # Don't start workers so queue accumulates
    # (we'll just check _queue.qsize() after putting)
    tq._queue = asyncio.Queue()
    tq._running = True  # bypass the running check

    task1 = _make_task_record(task_id="t1")
    task2 = _make_task_record(task_id="t2")
    await tq._queue.put(task1)
    await tq._queue.put(task2)

    assert tq.queue_size == 2
    tq._running = False


# ---------------------------------------------------------------------------
# Task execution – success path
# ---------------------------------------------------------------------------


@respx.mock
async def test_successful_task_sets_status_done(settings: Settings, db: Database):
    """Enqueue a task and verify it transitions to DONE in the DB."""
    ai_response = {
        "message": {"role": "assistant", "content": "print('hello world')"},
        "done": True,
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=ai_response)
    )

    task_record = await _create_and_get_task(db, "Write hello world in Python")

    tq = TaskQueue(settings=settings, db=db)
    await tq.start()
    try:
        await tq.enqueue(task_record)
        # Wait for the queue to drain
        await asyncio.wait_for(tq._queue.join(), timeout=10.0)
    finally:
        await tq.stop(timeout=5.0)

    final = await db.get_task(task_record.id)
    assert final is not None
    assert final.status == TaskStatus.DONE
    assert final.result is not None
    assert "hello" in final.result.lower() or "print" in final.result


@respx.mock
async def test_successful_task_stores_result(settings: Settings, db: Database):
    """Verify the AI response text is persisted to the DB."""
    result_text = "def bubble_sort(arr): pass"
    ai_response = {
        "message": {"role": "assistant", "content": result_text},
        "done": True,
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=ai_response)
    )

    task_record = await _create_and_get_task(db, "Write bubble sort")

    tq = TaskQueue(settings=settings, db=db)
    await tq.start()
    try:
        await tq.enqueue(task_record)
        await asyncio.wait_for(tq._queue.join(), timeout=10.0)
    finally:
        await tq.stop(timeout=5.0)

    final = await db.get_task(task_record.id)
    assert final is not None
    assert final.result == result_text


# ---------------------------------------------------------------------------
# Task execution – failure paths
# ---------------------------------------------------------------------------


@respx.mock
async def test_ai_error_sets_status_failed(settings: Settings, db: Database):
    """An AI backend error should set the task to FAILED with an error message."""
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    task_record = await _create_and_get_task(db, "Trigger AI error")

    tq = TaskQueue(settings=settings, db=db)
    await tq.start()
    try:
        await tq.enqueue(task_record)
        await asyncio.wait_for(tq._queue.join(), timeout=10.0)
    finally:
        await tq.stop(timeout=5.0)

    final = await db.get_task(task_record.id)
    assert final is not None
    assert final.status == TaskStatus.FAILED
    assert final.error is not None
    assert len(final.error) > 0


@respx.mock
async def test_connection_error_sets_status_failed(settings: Settings, db: Database):
    """A connection error should set the task to FAILED."""
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    task_record = await _create_and_get_task(db, "Unreachable backend")

    tq = TaskQueue(settings=settings, db=db)
    await tq.start()
    try:
        await tq.enqueue(task_record)
        await asyncio.wait_for(tq._queue.join(), timeout=10.0)
    finally:
        await tq.stop(timeout=5.0)

    final = await db.get_task(task_record.id)
    assert final is not None
    assert final.status == TaskStatus.FAILED


async def test_task_timeout_sets_status_failed(settings: Settings, db: Database):
    """A task that exceeds task_timeout_seconds should be marked FAILED."""
    # Override task timeout to a very small value
    with patch.dict(os.environ, {"TASK_TIMEOUT_SECONDS": "0.05", "AGENT_BRIDGE_TESTING": "1"}):
        fast_timeout_settings = Settings()

    async def slow_ai(*args: Any, **kwargs: Any) -> str:
        await asyncio.sleep(10)  # Much longer than timeout
        return "should not reach here"

    task_record = await _create_and_get_task(db, "Slow task")

    tq = TaskQueue(settings=fast_timeout_settings, db=db)
    # Patch the internal _run_ai to simulate slowness
    with patch.object(tq, "_run_ai", side_effect=slow_ai):
        await tq.start()
        try:
            await tq.enqueue(task_record)
            await asyncio.wait_for(tq._queue.join(), timeout=5.0)
        finally:
            await tq.stop(timeout=5.0)

    final = await db.get_task(task_record.id)
    assert final is not None
    assert final.status == TaskStatus.FAILED
    assert "timed out" in (final.error or "").lower()


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------


@respx.mock
async def test_messenger_callback_invoked_on_success(settings: Settings, db: Database):
    """The messenger callback should be called with the completed task."""
    ai_response = {
        "message": {"role": "assistant", "content": "result text"},
        "done": True,
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=ai_response)
    )

    received_records: list[TaskRecord] = []

    async def mock_messenger(record: TaskRecord) -> None:
        received_records.append(record)

    task_record = await _create_and_get_task(db)

    tq = TaskQueue(settings=settings, db=db, messenger=mock_messenger)
    await tq.start()
    try:
        await tq.enqueue(task_record)
        await asyncio.wait_for(tq._queue.join(), timeout=10.0)
    finally:
        await tq.stop(timeout=5.0)

    assert len(received_records) == 1
    assert received_records[0].status == TaskStatus.DONE


@respx.mock
async def test_messenger_callback_invoked_on_failure(settings: Settings, db: Database):
    """The messenger callback should also be called when a task fails."""
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(500, text="error")
    )

    received_records: list[TaskRecord] = []

    async def mock_messenger(record: TaskRecord) -> None:
        received_records.append(record)

    task_record = await _create_and_get_task(db)

    tq = TaskQueue(settings=settings, db=db, messenger=mock_messenger)
    await tq.start()
    try:
        await tq.enqueue(task_record)
        await asyncio.wait_for(tq._queue.join(), timeout=10.0)
    finally:
        await tq.stop(timeout=5.0)

    assert len(received_records) == 1
    assert received_records[0].status == TaskStatus.FAILED


@respx.mock
async def test_messenger_error_does_not_stop_queue(settings: Settings, db: Database):
    """A failing messenger must not crash the queue or leave tasks broken."""
    ai_response = {
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=ai_response)
    )

    async def bad_messenger(record: TaskRecord) -> None:
        raise RuntimeError("messenger blew up")

    task_record = await _create_and_get_task(db)

    tq = TaskQueue(settings=settings, db=db, messenger=bad_messenger)
    await tq.start()
    try:
        await tq.enqueue(task_record)
        await asyncio.wait_for(tq._queue.join(), timeout=10.0)
    finally:
        await tq.stop(timeout=5.0)

    # Task should still be DONE in the DB even if messenger failed
    final = await db.get_task(task_record.id)
    assert final is not None
    assert final.status == TaskStatus.DONE


@respx.mock
async def test_additional_callback_invoked(settings: Settings, db: Database):
    """add_completion_callback() callbacks should be invoked after the messenger."""
    ai_response = {
        "message": {"role": "assistant", "content": "callback test result"},
        "done": True,
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=ai_response)
    )

    call_order: list[str] = []

    async def messenger(record: TaskRecord) -> None:
        call_order.append("messenger")

    async def extra_callback(record: TaskRecord) -> None:
        call_order.append("extra")

    task_record = await _create_and_get_task(db)

    tq = TaskQueue(settings=settings, db=db, messenger=messenger)
    tq.add_completion_callback(extra_callback)
    await tq.start()
    try:
        await tq.enqueue(task_record)
        await asyncio.wait_for(tq._queue.join(), timeout=10.0)
    finally:
        await tq.stop(timeout=5.0)

    assert "messenger" in call_order
    assert "extra" in call_order
    assert call_order.index("messenger") < call_order.index("extra")


@respx.mock
async def test_multiple_additional_callbacks(settings: Settings, db: Database):
    """Multiple registered callbacks should all be invoked."""
    ai_response = {
        "message": {"role": "assistant", "content": "multi callback"},
        "done": True,
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=ai_response)
    )

    invocations: list[str] = []

    async def cb1(record: TaskRecord) -> None:
        invocations.append("cb1")

    async def cb2(record: TaskRecord) -> None:
        invocations.append("cb2")

    async def cb3(record: TaskRecord) -> None:
        invocations.append("cb3")

    task_record = await _create_and_get_task(db)

    tq = TaskQueue(settings=settings, db=db)
    tq.add_completion_callback(cb1)
    tq.add_completion_callback(cb2)
    tq.add_completion_callback(cb3)
    await tq.start()
    try:
        await tq.enqueue(task_record)
        await asyncio.wait_for(tq._queue.join(), timeout=10.0)
    finally:
        await tq.stop(timeout=5.0)

    assert "cb1" in invocations
    assert "cb2" in invocations
    assert "cb3" in invocations


# ---------------------------------------------------------------------------
# External webhook callback
# ---------------------------------------------------------------------------


@respx.mock
async def test_external_callback_posted_on_success():
    """If callback_url is set, a POST should be sent on task completion."""
    with patch.dict(
        os.environ,
        {
            "AGENT_BRIDGE_TESTING": "1",
            "AI_BACKEND_TYPE": "ollama",
            "AI_BACKEND_URL": "http://localhost:11434",
            "AI_MODEL": "llama3",
            "AI_MAX_TOKENS": "100",
            "AI_TEMPERATURE": "0.2",
            "AI_TIMEOUT_SECONDS": "10",
            "MAX_CONCURRENT_TASKS": "2",
            "TASK_TIMEOUT_SECONDS": "30",
            "CALLBACK_URL": "http://callback.example.com/done",
        },
    ):
        cb_settings = Settings()

    ai_response = {
        "message": {"role": "assistant", "content": "callback result"},
        "done": True,
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=ai_response)
    )
    callback_route = respx.post("http://callback.example.com/done").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    async with Database(":memory:") as db:
        task_record = await _create_and_get_task(db)
        tq = TaskQueue(settings=cb_settings, db=db)
        await tq.start()
        try:
            await tq.enqueue(task_record)
            await asyncio.wait_for(tq._queue.join(), timeout=10.0)
        finally:
            await tq.stop(timeout=5.0)

    assert callback_route.called


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


@respx.mock
async def test_concurrency_limit_respected(settings: Settings, db: Database):
    """No more than max_concurrent_tasks should run simultaneously."""
    max_concurrent = settings.max_concurrent_tasks
    concurrent_count = 0
    peak_concurrent = 0
    lock = asyncio.Lock()

    ai_response = {
        "message": {"role": "assistant", "content": "concurrent result"},
        "done": True,
    }

    async def slow_ai_side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal concurrent_count, peak_concurrent
        async with lock:
            concurrent_count += 1
            if concurrent_count > peak_concurrent:
                peak_concurrent = concurrent_count
        await asyncio.sleep(0.05)  # simulate AI work
        async with lock:
            concurrent_count -= 1
        return httpx.Response(200, json=ai_response)

    respx.post("http://localhost:11434/api/chat").mock(side_effect=slow_ai_side_effect)

    num_tasks = max_concurrent + 2
    task_records = [
        await _create_and_get_task(db, f"Task {i}") for i in range(num_tasks)
    ]

    tq = TaskQueue(settings=settings, db=db)
    await tq.start()
    try:
        for task in task_records:
            await tq.enqueue(task)
        await asyncio.wait_for(tq._queue.join(), timeout=15.0)
    finally:
        await tq.stop(timeout=5.0)

    assert peak_concurrent <= max_concurrent, (
        f"Peak concurrent ({peak_concurrent}) exceeded limit ({max_concurrent})"
    )

    # All tasks should be DONE
    for task in task_records:
        final = await db.get_task(task.id)
        assert final is not None
        assert final.status == TaskStatus.DONE


@respx.mock
async def test_multiple_tasks_all_complete(settings: Settings, db: Database):
    """All enqueued tasks should eventually be processed."""
    ai_response = {
        "message": {"role": "assistant", "content": "done"},
        "done": True,
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=ai_response)
    )

    num_tasks = 6
    task_records = [
        await _create_and_get_task(db, f"Task {i}") for i in range(num_tasks)
    ]

    tq = TaskQueue(settings=settings, db=db)
    await tq.start()
    try:
        for task in task_records:
            await tq.enqueue(task)
        await asyncio.wait_for(tq._queue.join(), timeout=20.0)
    finally:
        await tq.stop(timeout=5.0)

    for task in task_records:
        final = await db.get_task(task.id)
        assert final is not None
        assert final.status == TaskStatus.DONE


# ---------------------------------------------------------------------------
# Injected HTTP client
# ---------------------------------------------------------------------------


@respx.mock
async def test_injected_http_client_is_used(settings: Settings, db: Database):
    """The TaskQueue should use a provided HTTP client for AI calls."""
    ai_response = {
        "message": {"role": "assistant", "content": "injected client result"},
        "done": True,
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=ai_response)
    )

    task_record = await _create_and_get_task(db)

    external_client = httpx.AsyncClient()
    try:
        tq = TaskQueue(settings=settings, db=db, http_client=external_client)
        await tq.start()
        try:
            await tq.enqueue(task_record)
            await asyncio.wait_for(tq._queue.join(), timeout=10.0)
        finally:
            await tq.stop(timeout=5.0)
    finally:
        await external_client.aclose()

    final = await db.get_task(task_record.id)
    assert final is not None
    assert final.status == TaskStatus.DONE
