"""Tests for agent_bridge.db – SQLite task store CRUD operations.

Covers:
- Database lifecycle (initialize, close, context manager)
- Task creation and retrieval
- Status updates with result/error fields
- Filtering and pagination in list_tasks
- Count queries
- Deletion (single and by status)
"""

from __future__ import annotations

import asyncio

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


async def test_initialize_is_idempotent() -> None:
    """Calling initialize() twice must not raise."""
    db = Database(":memory:")
    await db.initialize()
    await db.initialize()  # second call must be safe
    await db.close()


async def test_close_without_initialize_is_safe() -> None:
    """Calling close() before initialize() must not raise."""
    db = Database(":memory:")
    await db.close()  # should not raise


async def test_context_manager(db: Database) -> None:
    """Verify the async context manager yields a ready-to-use Database."""
    assert db._conn is not None


async def test_context_manager_closes_on_exit() -> None:
    """Connection should be None after exiting the context manager."""
    db = Database(":memory:")
    async with db:
        assert db._conn is not None
    assert db._conn is None


async def test_require_connection_raises_when_not_initialised() -> None:
    """Operations on an uninitialised database must raise RuntimeError."""
    db = Database(":memory:")
    with pytest.raises(RuntimeError, match="not initialised"):
        db._require_connection()


async def test_require_connection_returns_connection_when_initialised(
    db: Database,
) -> None:
    """_require_connection should return the active connection when ready."""
    conn = db._require_connection()
    assert conn is not None


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------


async def test_create_task_returns_uuid(db: Database) -> None:
    """create_task should return a UUID4 string."""
    task_id = await db.create_task(_make_task_create())
    assert isinstance(task_id, str)
    assert len(task_id) == 36  # UUID4 canonical form with hyphens


async def test_create_task_uuid_format(db: Database) -> None:
    """The returned UUID should have the correct hyphen structure."""
    task_id = await db.create_task(_make_task_create())
    parts = task_id.split("-")
    assert len(parts) == 5
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


async def test_create_task_persisted(db: Database) -> None:
    """Created task should be retrievable from the database."""
    task_id = await db.create_task(_make_task_create())
    record = await db.get_task(task_id)
    assert record is not None
    assert record.id == task_id
    assert record.status == TaskStatus.PENDING


async def test_create_task_default_status_is_pending(db: Database) -> None:
    """Newly created tasks must have PENDING status."""
    task_id = await db.create_task(_make_task_create())
    record = await db.get_task(task_id)
    assert record is not None
    assert record.status == TaskStatus.PENDING


async def test_create_task_result_and_error_null(db: Database) -> None:
    """New tasks must have NULL result and error fields."""
    task_id = await db.create_task(_make_task_create())
    record = await db.get_task(task_id)
    assert record is not None
    assert record.result is None
    assert record.error is None


async def test_create_task_stores_correct_fields(db: Database) -> None:
    """All fields from TaskCreate should be persisted correctly."""
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


async def test_create_task_telegram_platform(db: Database) -> None:
    """Platform.TELEGRAM should be stored and retrieved correctly."""
    task_id = await db.create_task(_make_task_create(platform=Platform.TELEGRAM))
    record = await db.get_task(task_id)
    assert record is not None
    assert record.platform == Platform.TELEGRAM


async def test_create_task_discord_platform(db: Database) -> None:
    """Platform.DISCORD should be stored and retrieved correctly."""
    task_id = await db.create_task(_make_task_create(platform=Platform.DISCORD))
    record = await db.get_task(task_id)
    assert record is not None
    assert record.platform == Platform.DISCORD


async def test_multiple_tasks_have_unique_ids(db: Database) -> None:
    """Each created task must receive a unique UUID."""
    ids = [await db.create_task(_make_task_create()) for _ in range(5)]
    assert len(set(ids)) == 5


async def test_create_task_sets_timestamps(db: Database) -> None:
    """created_at and updated_at should be set on creation."""
    task_id = await db.create_task(_make_task_create())
    record = await db.get_task(task_id)
    assert record is not None
    assert record.created_at is not None
    assert record.updated_at is not None


async def test_create_task_created_and_updated_at_equal(db: Database) -> None:
    """created_at and updated_at should be equal immediately after creation."""
    task_id = await db.create_task(_make_task_create())
    record = await db.get_task(task_id)
    assert record is not None
    # Allow up to 1 second difference due to micro-timing
    delta = abs((record.updated_at - record.created_at).total_seconds())
    assert delta < 1.0


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------


async def test_get_task_not_found_returns_none(db: Database) -> None:
    """get_task with a non-existent ID should return None."""
    result = await db.get_task("nonexistent-uuid")
    assert result is None


async def test_get_task_empty_string_returns_none(db: Database) -> None:
    """get_task with an empty string ID should return None."""
    result = await db.get_task("")
    assert result is None


async def test_get_task_returns_task_record_instance(db: Database) -> None:
    """get_task should return a TaskRecord instance for existing tasks."""
    task_id = await db.create_task(_make_task_create())
    record = await db.get_task(task_id)
    assert isinstance(record, TaskRecord)


async def test_get_task_all_fields_present(db: Database) -> None:
    """get_task should populate all fields in the returned TaskRecord."""
    tc = _make_task_create(
        platform=Platform.TELEGRAM,
        chat_id="200",
        user_id="99",
        prompt="Generate a fibonacci sequence",
    )
    task_id = await db.create_task(tc)
    record = await db.get_task(task_id)
    assert record is not None
    assert record.id == task_id
    assert record.platform == Platform.TELEGRAM
    assert record.chat_id == "200"
    assert record.user_id == "99"
    assert record.prompt == "Generate a fibonacci sequence"
    assert record.status == TaskStatus.PENDING
    assert record.result is None
    assert record.error is None
    assert record.created_at is not None
    assert record.updated_at is not None


# ---------------------------------------------------------------------------
# update_task_status
# ---------------------------------------------------------------------------


async def test_update_status_to_running(db: Database) -> None:
    """Updating status to RUNNING should persist correctly."""
    task_id = await db.create_task(_make_task_create())
    updated = await db.update_task_status(
        task_id, TaskStatusUpdate(status=TaskStatus.RUNNING)
    )
    assert updated is not None
    assert updated.status == TaskStatus.RUNNING


async def test_update_status_to_done_with_result(db: Database) -> None:
    """Updating status to DONE with a result should persist both fields."""
    task_id = await db.create_task(_make_task_create())
    updated = await db.update_task_status(
        task_id,
        TaskStatusUpdate(status=TaskStatus.DONE, result="def sort(): ..."),
    )
    assert updated is not None
    assert updated.status == TaskStatus.DONE
    assert updated.result == "def sort(): ..."
    assert updated.error is None


async def test_update_status_to_failed_with_error(db: Database) -> None:
    """Updating status to FAILED with an error should persist both fields."""
    task_id = await db.create_task(_make_task_create())
    updated = await db.update_task_status(
        task_id,
        TaskStatusUpdate(status=TaskStatus.FAILED, error="AI backend timeout"),
    )
    assert updated is not None
    assert updated.status == TaskStatus.FAILED
    assert updated.error == "AI backend timeout"
    assert updated.result is None


async def test_update_status_clears_previous_result(db: Database) -> None:
    """A subsequent update should overwrite the previous result field."""
    task_id = await db.create_task(_make_task_create())
    # First: mark done with a result
    await db.update_task_status(
        task_id,
        TaskStatusUpdate(status=TaskStatus.DONE, result="first result"),
    )
    # Second: mark failed with no result
    updated = await db.update_task_status(
        task_id,
        TaskStatusUpdate(status=TaskStatus.FAILED, error="oops"),
    )
    assert updated is not None
    assert updated.status == TaskStatus.FAILED
    assert updated.result is None
    assert updated.error == "oops"


async def test_update_nonexistent_task_returns_none(db: Database) -> None:
    """Updating a non-existent task ID should return None."""
    result = await db.update_task_status(
        "ghost-id", TaskStatusUpdate(status=TaskStatus.DONE)
    )
    assert result is None


async def test_updated_at_changes_after_update(db: Database) -> None:
    """updated_at timestamp should increase after a status update."""
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


async def test_update_returns_current_record(db: Database) -> None:
    """update_task_status should return the fully updated TaskRecord."""
    tc = _make_task_create(prompt="test prompt")
    task_id = await db.create_task(tc)
    result = await db.update_task_status(
        task_id,
        TaskStatusUpdate(status=TaskStatus.RUNNING),
    )
    assert result is not None
    assert result.id == task_id
    assert result.prompt == "test prompt"  # original fields preserved
    assert result.status == TaskStatus.RUNNING


async def test_update_preserves_original_fields(db: Database) -> None:
    """Status updates must not overwrite platform, chat_id, user_id, or prompt."""
    tc = _make_task_create(
        platform=Platform.DISCORD,
        chat_id="chan99",
        user_id="user99",
        prompt="preserve me",
    )
    task_id = await db.create_task(tc)
    updated = await db.update_task_status(
        task_id,
        TaskStatusUpdate(status=TaskStatus.DONE, result="done"),
    )
    assert updated is not None
    assert updated.platform == Platform.DISCORD
    assert updated.chat_id == "chan99"
    assert updated.user_id == "user99"
    assert updated.prompt == "preserve me"


async def test_update_status_without_result_or_error(db: Database) -> None:
    """Updating status with no result or error should work without error."""
    task_id = await db.create_task(_make_task_create())
    updated = await db.update_task_status(
        task_id, TaskStatusUpdate(status=TaskStatus.RUNNING)
    )
    assert updated is not None
    assert updated.status == TaskStatus.RUNNING
    assert updated.result is None
    assert updated.error is None


async def test_multiple_sequential_status_updates(db: Database) -> None:
    """A task should transition through multiple statuses correctly."""
    task_id = await db.create_task(_make_task_create())

    # PENDING -> RUNNING
    r1 = await db.update_task_status(
        task_id, TaskStatusUpdate(status=TaskStatus.RUNNING)
    )
    assert r1 is not None
    assert r1.status == TaskStatus.RUNNING

    # RUNNING -> DONE
    r2 = await db.update_task_status(
        task_id, TaskStatusUpdate(status=TaskStatus.DONE, result="final")
    )
    assert r2 is not None
    assert r2.status == TaskStatus.DONE
    assert r2.result == "final"


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


async def test_list_tasks_empty(db: Database) -> None:
    """list_tasks on an empty database should return an empty list."""
    tasks = await db.list_tasks()
    assert tasks == []


async def test_list_tasks_returns_all(db: Database) -> None:
    """list_tasks with no filters should return all tasks."""
    for _ in range(3):
        await db.create_task(_make_task_create())
    tasks = await db.list_tasks()
    assert len(tasks) == 3


async def test_list_tasks_returns_task_record_instances(db: Database) -> None:
    """list_tasks should return TaskRecord instances."""
    await db.create_task(_make_task_create())
    tasks = await db.list_tasks()
    assert len(tasks) == 1
    assert isinstance(tasks[0], TaskRecord)


async def test_list_tasks_filter_by_platform_telegram(db: Database) -> None:
    """list_tasks filtered by TELEGRAM should exclude DISCORD tasks."""
    await db.create_task(_make_task_create(platform=Platform.TELEGRAM))
    await db.create_task(_make_task_create(platform=Platform.DISCORD))
    await db.create_task(_make_task_create(platform=Platform.TELEGRAM))

    tg_tasks = await db.list_tasks(platform=Platform.TELEGRAM)
    dc_tasks = await db.list_tasks(platform=Platform.DISCORD)

    assert len(tg_tasks) == 2
    assert len(dc_tasks) == 1
    assert all(t.platform == Platform.TELEGRAM for t in tg_tasks)
    assert all(t.platform == Platform.DISCORD for t in dc_tasks)


async def test_list_tasks_filter_by_status_pending(db: Database) -> None:
    """list_tasks filtered by PENDING should return only pending tasks."""
    id1 = await db.create_task(_make_task_create())
    id2 = await db.create_task(_make_task_create())
    await db.create_task(_make_task_create())

    await db.update_task_status(id1, TaskStatusUpdate(status=TaskStatus.DONE))
    await db.update_task_status(id2, TaskStatusUpdate(status=TaskStatus.RUNNING))

    done_tasks = await db.list_tasks(status=TaskStatus.DONE)
    pending_tasks = await db.list_tasks(status=TaskStatus.PENDING)
    running_tasks = await db.list_tasks(status=TaskStatus.RUNNING)

    assert len(done_tasks) == 1
    assert done_tasks[0].status == TaskStatus.DONE

    assert len(pending_tasks) == 1
    assert pending_tasks[0].status == TaskStatus.PENDING

    assert len(running_tasks) == 1
    assert running_tasks[0].status == TaskStatus.RUNNING


async def test_list_tasks_filter_by_failed_status(db: Database) -> None:
    """list_tasks filtered by FAILED should return only failed tasks."""
    id1 = await db.create_task(_make_task_create())
    await db.create_task(_make_task_create())

    await db.update_task_status(
        id1, TaskStatusUpdate(status=TaskStatus.FAILED, error="error")
    )

    failed = await db.list_tasks(status=TaskStatus.FAILED)
    assert len(failed) == 1
    assert failed[0].status == TaskStatus.FAILED


async def test_list_tasks_filter_by_user_id(db: Database) -> None:
    """list_tasks filtered by user_id should return only that user's tasks."""
    await db.create_task(_make_task_create(user_id="user_a"))
    await db.create_task(_make_task_create(user_id="user_b"))
    await db.create_task(_make_task_create(user_id="user_a"))

    tasks_a = await db.list_tasks(user_id="user_a")
    tasks_b = await db.list_tasks(user_id="user_b")

    assert len(tasks_a) == 2
    assert all(t.user_id == "user_a" for t in tasks_a)

    assert len(tasks_b) == 1
    assert tasks_b[0].user_id == "user_b"


async def test_list_tasks_filter_by_chat_id(db: Database) -> None:
    """list_tasks filtered by chat_id should return only tasks for that chat."""
    await db.create_task(_make_task_create(chat_id="chat_1"))
    await db.create_task(_make_task_create(chat_id="chat_2"))
    await db.create_task(_make_task_create(chat_id="chat_1"))

    tasks_1 = await db.list_tasks(chat_id="chat_1")
    tasks_2 = await db.list_tasks(chat_id="chat_2")

    assert len(tasks_1) == 2
    assert all(t.chat_id == "chat_1" for t in tasks_1)
    assert len(tasks_2) == 1


async def test_list_tasks_combined_filters(db: Database) -> None:
    """Multiple filters should be ANDed together."""
    await db.create_task(
        _make_task_create(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1")
    )
    await db.create_task(
        _make_task_create(platform=Platform.DISCORD, user_id="u1", chat_id="c2")
    )
    await db.create_task(
        _make_task_create(platform=Platform.TELEGRAM, user_id="u2", chat_id="c1")
    )

    results = await db.list_tasks(
        platform=Platform.TELEGRAM,
        user_id="u1",
    )
    assert len(results) == 1
    assert results[0].platform == Platform.TELEGRAM
    assert results[0].user_id == "u1"


async def test_list_tasks_pagination_limit(db: Database) -> None:
    """limit parameter should cap the number of returned records."""
    for i in range(5):
        await db.create_task(_make_task_create(prompt=f"Task {i}"))

    tasks = await db.list_tasks(limit=3)
    assert len(tasks) == 3


async def test_list_tasks_pagination_offset(db: Database) -> None:
    """offset parameter should skip the specified number of records."""
    for i in range(5):
        await db.create_task(_make_task_create(prompt=f"Task {i}"))

    all_tasks = await db.list_tasks(limit=100)
    page1 = await db.list_tasks(limit=2, offset=0)
    page2 = await db.list_tasks(limit=2, offset=2)
    page3 = await db.list_tasks(limit=2, offset=4)

    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1

    # No duplicates across pages
    all_ids = {t.id for t in page1 + page2 + page3}
    assert len(all_ids) == 5


async def test_list_tasks_pagination_no_overlap(db: Database) -> None:
    """Paginated results should not overlap between pages."""
    for i in range(6):
        await db.create_task(_make_task_create(prompt=f"Task {i}"))

    page1 = await db.list_tasks(limit=3, offset=0)
    page2 = await db.list_tasks(limit=3, offset=3)

    ids1 = {t.id for t in page1}
    ids2 = {t.id for t in page2}
    assert ids1.isdisjoint(ids2)


async def test_list_tasks_ordered_newest_first(db: Database) -> None:
    """list_tasks should return results ordered by created_at descending."""
    id1 = await db.create_task(_make_task_create(prompt="first"))
    await asyncio.sleep(0.01)
    id2 = await db.create_task(_make_task_create(prompt="second"))
    await asyncio.sleep(0.01)
    id3 = await db.create_task(_make_task_create(prompt="third"))

    tasks = await db.list_tasks()
    assert tasks[0].id == id3
    assert tasks[1].id == id2
    assert tasks[2].id == id1


async def test_list_tasks_offset_beyond_total_returns_empty(db: Database) -> None:
    """An offset beyond the total record count should return an empty list."""
    await db.create_task(_make_task_create())
    tasks = await db.list_tasks(offset=999)
    assert tasks == []


async def test_list_tasks_filter_no_match_returns_empty(db: Database) -> None:
    """A filter that matches no tasks should return an empty list."""
    await db.create_task(_make_task_create(user_id="known_user"))
    tasks = await db.list_tasks(user_id="unknown_user")
    assert tasks == []


# ---------------------------------------------------------------------------
# count_tasks
# ---------------------------------------------------------------------------


async def test_count_tasks_empty(db: Database) -> None:
    """count_tasks on an empty database should return 0."""
    count = await db.count_tasks()
    assert count == 0


async def test_count_tasks_all(db: Database) -> None:
    """count_tasks with no filters should return the total number of tasks."""
    for _ in range(4):
        await db.create_task(_make_task_create())
    assert await db.count_tasks() == 4


async def test_count_tasks_by_status(db: Database) -> None:
    """count_tasks filtered by status should count correctly."""
    id1 = await db.create_task(_make_task_create())
    await db.create_task(_make_task_create())
    await db.update_task_status(id1, TaskStatusUpdate(status=TaskStatus.DONE))

    assert await db.count_tasks(status=TaskStatus.DONE) == 1
    assert await db.count_tasks(status=TaskStatus.PENDING) == 1
    assert await db.count_tasks(status=TaskStatus.RUNNING) == 0
    assert await db.count_tasks(status=TaskStatus.FAILED) == 0


async def test_count_tasks_by_platform(db: Database) -> None:
    """count_tasks filtered by platform should count correctly."""
    await db.create_task(_make_task_create(platform=Platform.TELEGRAM))
    await db.create_task(_make_task_create(platform=Platform.TELEGRAM))
    await db.create_task(_make_task_create(platform=Platform.DISCORD))

    assert await db.count_tasks(platform=Platform.TELEGRAM) == 2
    assert await db.count_tasks(platform=Platform.DISCORD) == 1


async def test_count_tasks_by_user_id(db: Database) -> None:
    """count_tasks filtered by user_id should count correctly."""
    await db.create_task(_make_task_create(user_id="alice"))
    await db.create_task(_make_task_create(user_id="alice"))
    await db.create_task(_make_task_create(user_id="bob"))

    assert await db.count_tasks(user_id="alice") == 2
    assert await db.count_tasks(user_id="bob") == 1
    assert await db.count_tasks(user_id="charlie") == 0


async def test_count_tasks_by_chat_id(db: Database) -> None:
    """count_tasks filtered by chat_id should count correctly."""
    await db.create_task(_make_task_create(chat_id="room_a"))
    await db.create_task(_make_task_create(chat_id="room_b"))
    await db.create_task(_make_task_create(chat_id="room_a"))

    assert await db.count_tasks(chat_id="room_a") == 2
    assert await db.count_tasks(chat_id="room_b") == 1


async def test_count_tasks_combined_filters(db: Database) -> None:
    """count_tasks with multiple filters should AND them correctly."""
    id1 = await db.create_task(
        _make_task_create(platform=Platform.TELEGRAM, user_id="u1")
    )
    await db.create_task(
        _make_task_create(platform=Platform.DISCORD, user_id="u1")
    )
    await db.update_task_status(id1, TaskStatusUpdate(status=TaskStatus.DONE))

    assert (
        await db.count_tasks(platform=Platform.TELEGRAM, status=TaskStatus.DONE) == 1
    )
    assert (
        await db.count_tasks(platform=Platform.DISCORD, status=TaskStatus.DONE) == 0
    )


async def test_count_matches_list_length(db: Database) -> None:
    """count_tasks result should always match len(list_tasks) for the same filters."""
    for i in range(7):
        tid = await db.create_task(_make_task_create(prompt=f"p{i}"))
        if i % 2 == 0:
            await db.update_task_status(
                tid, TaskStatusUpdate(status=TaskStatus.DONE)
            )

    for status_filter in list(TaskStatus):
        count = await db.count_tasks(status=status_filter)
        tasks = await db.list_tasks(status=status_filter, limit=100)
        assert count == len(tasks), (
            f"count ({count}) != len(list) ({len(tasks)}) for status={status_filter}"
        )


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------


async def test_delete_task_returns_true(db: Database) -> None:
    """Deleting an existing task should return True."""
    task_id = await db.create_task(_make_task_create())
    result = await db.delete_task(task_id)
    assert result is True


async def test_delete_task_removes_from_db(db: Database) -> None:
    """After deletion, get_task should return None for that task."""
    task_id = await db.create_task(_make_task_create())
    await db.delete_task(task_id)
    assert await db.get_task(task_id) is None


async def test_delete_task_reduces_count(db: Database) -> None:
    """Deleting a task should reduce the total count by one."""
    id1 = await db.create_task(_make_task_create())
    await db.create_task(_make_task_create())
    assert await db.count_tasks() == 2

    await db.delete_task(id1)
    assert await db.count_tasks() == 1


async def test_delete_nonexistent_task_returns_false(db: Database) -> None:
    """Deleting a non-existent task should return False."""
    result = await db.delete_task("does-not-exist")
    assert result is False


async def test_delete_task_does_not_affect_others(db: Database) -> None:
    """Deleting one task must not remove other tasks."""
    id1 = await db.create_task(_make_task_create())
    id2 = await db.create_task(_make_task_create())
    await db.delete_task(id1)

    assert await db.get_task(id1) is None
    assert await db.get_task(id2) is not None


async def test_delete_task_idempotent_returns_false_on_second_call(
    db: Database,
) -> None:
    """Deleting the same task twice should return False on the second attempt."""
    task_id = await db.create_task(_make_task_create())
    first = await db.delete_task(task_id)
    second = await db.delete_task(task_id)
    assert first is True
    assert second is False


async def test_delete_all_tasks_leaves_empty_db(db: Database) -> None:
    """Deleting all tasks should result in an empty database."""
    ids = [await db.create_task(_make_task_create()) for _ in range(3)]
    for task_id in ids:
        await db.delete_task(task_id)
    assert await db.count_tasks() == 0
    assert await db.list_tasks() == []


# ---------------------------------------------------------------------------
# delete_tasks_by_status
# ---------------------------------------------------------------------------


async def test_delete_tasks_by_status_done(db: Database) -> None:
    """delete_tasks_by_status(DONE) should remove all done tasks."""
    id1 = await db.create_task(_make_task_create())
    id2 = await db.create_task(_make_task_create())
    await db.create_task(_make_task_create())

    await db.update_task_status(id1, TaskStatusUpdate(status=TaskStatus.DONE))
    await db.update_task_status(id2, TaskStatusUpdate(status=TaskStatus.DONE))

    deleted = await db.delete_tasks_by_status(TaskStatus.DONE)
    assert deleted == 2
    assert await db.count_tasks(status=TaskStatus.DONE) == 0
    assert await db.count_tasks(status=TaskStatus.PENDING) == 1


async def test_delete_tasks_by_status_failed(db: Database) -> None:
    """delete_tasks_by_status(FAILED) should remove all failed tasks."""
    id1 = await db.create_task(_make_task_create())
    await db.create_task(_make_task_create())
    await db.update_task_status(
        id1, TaskStatusUpdate(status=TaskStatus.FAILED, error="err")
    )

    deleted = await db.delete_tasks_by_status(TaskStatus.FAILED)
    assert deleted == 1
    assert await db.count_tasks(status=TaskStatus.FAILED) == 0


async def test_delete_tasks_by_status_none_matching(db: Database) -> None:
    """delete_tasks_by_status with no matching tasks should return 0."""
    await db.create_task(_make_task_create())
    deleted = await db.delete_tasks_by_status(TaskStatus.DONE)
    assert deleted == 0


async def test_delete_tasks_by_status_does_not_affect_other_statuses(
    db: Database,
) -> None:
    """delete_tasks_by_status should only remove tasks with the given status."""
    id_done = await db.create_task(_make_task_create())
    id_pending = await db.create_task(_make_task_create())
    id_running = await db.create_task(_make_task_create())

    await db.update_task_status(id_done, TaskStatusUpdate(status=TaskStatus.DONE))
    await db.update_task_status(
        id_running, TaskStatusUpdate(status=TaskStatus.RUNNING)
    )

    deleted = await db.delete_tasks_by_status(TaskStatus.DONE)
    assert deleted == 1

    assert await db.get_task(id_done) is None
    assert await db.get_task(id_pending) is not None
    assert await db.get_task(id_running) is not None


async def test_delete_tasks_by_status_pending(db: Database) -> None:
    """delete_tasks_by_status(PENDING) should remove all pending tasks."""
    for _ in range(3):
        await db.create_task(_make_task_create())

    id_done = await db.create_task(_make_task_create())
    await db.update_task_status(id_done, TaskStatusUpdate(status=TaskStatus.DONE))

    deleted = await db.delete_tasks_by_status(TaskStatus.PENDING)
    assert deleted == 3
    assert await db.count_tasks(status=TaskStatus.PENDING) == 0
    assert await db.count_tasks(status=TaskStatus.DONE) == 1


async def test_delete_tasks_by_status_returns_count(db: Database) -> None:
    """delete_tasks_by_status should return the exact number of deleted rows."""
    for _ in range(5):
        tid = await db.create_task(_make_task_create())
        await db.update_task_status(
            tid, TaskStatusUpdate(status=TaskStatus.FAILED, error="e")
        )

    deleted = await db.delete_tasks_by_status(TaskStatus.FAILED)
    assert deleted == 5
