"""Tests for agent_bridge.db – SQLite task store CRUD operations."""

from __future__ import annotations

import pytest

from agent_bridge.db import Database
from agent_bridge.models import (
    Platform,
    TaskCreate,
    TaskRecord,
    TaskStatus,
    TaskStatusUpdate,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> Database:
    """Return an initialised in-memory Database for each test."""
    async with Database(":memory:") as database:
        yield database


def _make_task_create(
    platform: Platform = Platform.TELEGRAM,
    chat_id: str = "100",
    user_id: str = "42",
    prompt: str = "Write a sorting algorithm",
) -> TaskCreate:
    """Helper that returns a valid TaskCreate instance."""
    return TaskCreate(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


async def test_initialize_is_idempotent():
    """Calling initialize() twice must not raise."""
    db = Database(":memory:")
    await db.initialize()
    await db.initialize()  # second call must be safe
    await db.close()


async def test_close_without_initialize_is_safe():
    """Calling close() before initialize() must not raise."""
    db = Database(":memory:")
    await db.close()  # should not raise


async def test_context_manager(db: Database):
    """Verify the async context manager yields a ready-to-use Database."""
    assert db._conn is not None


async def test_require_connection_raises_when_not_initialised():
    """Operations on an uninitialised database must raise RuntimeError."""
    db = Database(":memory:")
    with pytest.raises(RuntimeError, match="not initialised"):
        db._require_connection()


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------


async def test_create_task_returns_uuid(db: Database):
    task_id = await db.create_task(_make_task_create())
    assert isinstance(task_id, str)
    assert len(task_id) == 36  # UUID4 canonical form


async def test_create_task_persisted(db: Database):
    task_id = await db.create_task(_make_task_create())
    record = await db.get_task(task_id)
    assert record is not None
    assert record.id == task_id
    assert record.status == TaskStatus.PENDING


async def test_create_task_stores_correct_fields(db: Database):
    tc = _make_task_create(
        platform=Platform.DISCORD,
        chat_id="chan1",
        user_id="usr1",
        prompt="Explain async generators",
    )
    task_id = await db.create_task(tc)
    record = await db.get_task(task_id)
    assert record is not None
    assert record.platform == Platform.DISCORD
    assert record.chat_id == "chan1"
    assert record.user_id == "usr1"
    assert record.prompt == "Explain async generators"
    assert record.result is None
    assert record.error is None


async def test_multiple_tasks_have_unique_ids(db: Database):
    ids = [await db.create_task(_make_task_create()) for _ in range(5)]
    assert len(set(ids)) == 5


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------


async def test_get_task_not_found_returns_none(db: Database):
    result = await db.get_task("nonexistent-uuid")
    assert result is None


async def test_get_task_returns_task_record_instance(db: Database):
    task_id = await db.create_task(_make_task_create())
    record = await db.get_task(task_id)
    assert isinstance(record, TaskRecord)


# ---------------------------------------------------------------------------
# update_task_status
# ---------------------------------------------------------------------------


async def test_update_status_to_running(db: Database):
    task_id = await db.create_task(_make_task_create())
    updated = await db.update_task_status(
        task_id, TaskStatusUpdate(status=TaskStatus.RUNNING)
    )
    assert updated is not None
    assert updated.status == TaskStatus.RUNNING


async def test_update_status_to_done_with_result(db: Database):
    task_id = await db.create_task(_make_task_create())
    updated = await db.update_task_status(
        task_id,
        TaskStatusUpdate(status=TaskStatus.DONE, result="def sort(): ..."),
    )
    assert updated is not None
    assert updated.status == TaskStatus.DONE
    assert updated.result == "def sort(): ..."
    assert updated.error is None


async def test_update_status_to_failed_with_error(db: Database):
    task_id = await db.create_task(_make_task_create())
    updated = await db.update_task_status(
        task_id,
        TaskStatusUpdate(status=TaskStatus.FAILED, error="AI backend timeout"),
    )
    assert updated is not None
    assert updated.status == TaskStatus.FAILED
    assert updated.error == "AI backend timeout"
    assert updated.result is None


async def test_update_nonexistent_task_returns_none(db: Database):
    result = await db.update_task_status(
        "ghost-id", TaskStatusUpdate(status=TaskStatus.DONE)
    )
    assert result is None


async def test_updated_at_changes_after_update(db: Database):
    import asyncio

    task_id = await db.create_task(_make_task_create())
    original = await db.get_task(task_id)
    assert original is not None

    await asyncio.sleep(0.01)  # ensure clock advances
    await db.update_task_status(
        task_id, TaskStatusUpdate(status=TaskStatus.RUNNING)
    )
    updated = await db.get_task(task_id)
    assert updated is not None
    assert updated.updated_at >= original.updated_at


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


async def test_list_tasks_empty(db: Database):
    tasks = await db.list_tasks()
    assert tasks == []


async def test_list_tasks_returns_all(db: Database):
    for _ in range(3):
        await db.create_task(_make_task_create())
    tasks = await db.list_tasks()
    assert len(tasks) == 3


async def test_list_tasks_filter_by_platform(db: Database):
    await db.create_task(_make_task_create(platform=Platform.TELEGRAM))
    await db.create_task(_make_task_create(platform=Platform.DISCORD))
    await db.create_task(_make_task_create(platform=Platform.TELEGRAM))

    tg_tasks = await db.list_tasks(platform=Platform.TELEGRAM)
    dc_tasks = await db.list_tasks(platform=Platform.DISCORD)

    assert len(tg_tasks) == 2
    assert len(dc_tasks) == 1


async def test_list_tasks_filter_by_status(db: Database):
    id1 = await db.create_task(_make_task_create())
    id2 = await db.create_task(_make_task_create())
    await db.create_task(_make_task_create())

    await db.update_task_status(id1, TaskStatusUpdate(status=TaskStatus.DONE))
    await db.update_task_status(id2, TaskStatusUpdate(status=TaskStatus.RUNNING))

    done_tasks = await db.list_tasks(status=TaskStatus.DONE)
    pending_tasks = await db.list_tasks(status=TaskStatus.PENDING)
    running_tasks = await db.list_tasks(status=TaskStatus.RUNNING)

    assert len(done_tasks) == 1
    assert len(pending_tasks) == 1
    assert len(running_tasks) == 1


async def test_list_tasks_filter_by_user_id(db: Database):
    await db.create_task(_make_task_create(user_id="user_a"))
    await db.create_task(_make_task_create(user_id="user_b"))
    await db.create_task(_make_task_create(user_id="user_a"))

    tasks_a = await db.list_tasks(user_id="user_a")
    tasks_b = await db.list_tasks(user_id="user_b")

    assert len(tasks_a) == 2
    assert len(tasks_b) == 1


async def test_list_tasks_filter_by_chat_id(db: Database):
    await db.create_task(_make_task_create(chat_id="chat_1"))
    await db.create_task(_make_task_create(chat_id="chat_2"))

    tasks_1 = await db.list_tasks(chat_id="chat_1")
    assert len(tasks_1) == 1


async def test_list_tasks_pagination(db: Database):
    for i in range(5):
        await db.create_task(_make_task_create(prompt=f"Task {i}"))

    page1 = await db.list_tasks(limit=2, offset=0)
    page2 = await db.list_tasks(limit=2, offset=2)
    page3 = await db.list_tasks(limit=2, offset=4)

    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1

    # No duplicates across pages
    all_ids = {t.id for t in page1 + page2 + page3}
    assert len(all_ids) == 5


async def test_list_tasks_ordered_newest_first(db: Database):
    import asyncio

    id1 = await db.create_task(_make_task_create(prompt="first"))
    await asyncio.sleep(0.01)
    id2 = await db.create_task(_make_task_create(prompt="second"))
    await asyncio.sleep(0.01)
    id3 = await db.create_task(_make_task_create(prompt="third"))

    tasks = await db.list_tasks()
    assert tasks[0].id == id3
    assert tasks[1].id == id2
    assert tasks[2].id == id1


# ---------------------------------------------------------------------------
# count_tasks
# ---------------------------------------------------------------------------


async def test_count_tasks_empty(db: Database):
    count = await db.count_tasks()
    assert count == 0


async def test_count_tasks_all(db: Database):
    for _ in range(4):
        await db.create_task(_make_task_create())
    assert await db.count_tasks() == 4


async def test_count_tasks_by_status(db: Database):
    id1 = await db.create_task(_make_task_create())
    await db.create_task(_make_task_create())
    await db.update_task_status(id1, TaskStatusUpdate(status=TaskStatus.DONE))

    assert await db.count_tasks(status=TaskStatus.DONE) == 1
    assert await db.count_tasks(status=TaskStatus.PENDING) == 1


async def test_count_tasks_by_platform(db: Database):
    await db.create_task(_make_task_create(platform=Platform.TELEGRAM))
    await db.create_task(_make_task_create(platform=Platform.TELEGRAM))
    await db.create_task(_make_task_create(platform=Platform.DISCORD))

    assert await db.count_tasks(platform=Platform.TELEGRAM) == 2
    assert await db.count_tasks(platform=Platform.DISCORD) == 1


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------


async def test_delete_task_returns_true(db: Database):
    task_id = await db.create_task(_make_task_create())
    result = await db.delete_task(task_id)
    assert result is True


async def test_delete_task_removes_from_db(db: Database):
    task_id = await db.create_task(_make_task_create())
    await db.delete_task(task_id)
    assert await db.get_task(task_id) is None


async def test_delete_nonexistent_task_returns_false(db: Database):
    result = await db.delete_task("does-not-exist")
    assert result is False


async def test_delete_task_does_not_affect_others(db: Database):
    id1 = await db.create_task(_make_task_create())
    id2 = await db.create_task(_make_task_create())
    await db.delete_task(id1)

    assert await db.get_task(id1) is None
    assert await db.get_task(id2) is not None


# ---------------------------------------------------------------------------
# delete_tasks_by_status
# ---------------------------------------------------------------------------


async def test_delete_tasks_by_status(db: Database):
    id1 = await db.create_task(_make_task_create())
    id2 = await db.create_task(_make_task_create())
    await db.create_task(_make_task_create())

    await db.update_task_status(id1, TaskStatusUpdate(status=TaskStatus.DONE))
    await db.update_task_status(id2, TaskStatusUpdate(status=TaskStatus.DONE))

    deleted = await db.delete_tasks_by_status(TaskStatus.DONE)
    assert deleted == 2
    assert await db.count_tasks(status=TaskStatus.DONE) == 0
    assert await db.count_tasks(status=TaskStatus.PENDING) == 1


async def test_delete_tasks_by_status_none_matching(db: Database):
    await db.create_task(_make_task_create())
    deleted = await db.delete_tasks_by_status(TaskStatus.DONE)
    assert deleted == 0
