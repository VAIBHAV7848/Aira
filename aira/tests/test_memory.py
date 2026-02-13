"""Tests for the memory store."""

import pytest
import pytest_asyncio
from aira.memory.memory_store import MemoryStore


@pytest_asyncio.fixture
async def store(tmp_path):
    db = tmp_path / "test.db"
    s = MemoryStore(db)
    await s.init_db()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_save_and_get_task(store):
    await store.save_task("t1", {"description": "Test task", "status": "PLANNING"})
    task = await store.get_task("t1")
    assert task is not None
    assert task["description"] == "Test task"
    assert task["status"] == "PLANNING"


@pytest.mark.asyncio
async def test_get_missing_task(store):
    task = await store.get_task("nonexistent")
    assert task is None


@pytest.mark.asyncio
async def test_update_task_status(store):
    await store.save_task("t2", {"description": "Task 2"})
    await store.update_task_status("t2", "COMPLETED", "All done")
    task = await store.get_task("t2")
    assert task["status"] == "COMPLETED"
    assert task["summary"] == "All done"


@pytest.mark.asyncio
async def test_save_step(store):
    await store.save_task("t3", {"description": "Task 3"})
    await store.save_step("t3", {
        "iteration": 1,
        "tool_name": "file_read",
        "parameters": {"path": "test.txt"},
        "output": "file contents",
        "success": True,
        "duration_ms": 50,
    })
    # No error means success


@pytest.mark.asyncio
async def test_recent_summaries_limit(store):
    for i in range(5):
        await store.save_task(f"s{i}", {
            "description": f"Task {i}",
            "summary": f"Summary {i}",
        })
    summaries = await store.get_recent_summaries(limit=3)
    assert len(summaries) <= 3


@pytest.mark.asyncio
async def test_recent_summaries_token_cap(store):
    # Create a task with a very long summary
    long_summary = "x" * 10000
    await store.save_task("long", {"description": "Long", "summary": long_summary})
    summaries = await store.get_recent_summaries(limit=3)
    total_len = sum(len(s) for s in summaries)
    assert total_len <= 2000 * 4 + 100  # MAX_SUMMARY_TOKENS * CHARS_PER_TOKEN + buffer


@pytest.mark.asyncio
async def test_preferences(store):
    await store.save_preference("theme", "dark")
    val = await store.get_preference("theme")
    assert val == "dark"


@pytest.mark.asyncio
async def test_preference_missing(store):
    val = await store.get_preference("missing")
    assert val is None


@pytest.mark.asyncio
async def test_persona_state(store):
    await store.save_persona_state("mood", "happy")
    val = await store.get_persona_state("mood")
    assert val == "happy"
