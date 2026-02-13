# Phase 5 — Memory System

## Goal
SQLite-backed memory with injection-safe retrieval and token-limited summaries.

## Prerequisites
- Phase 1 config passing

## File to Create

### `aira/memory/memory_store.py`

```python
"""
SQLite-backed memory store for tasks, steps, preferences, and persona state.
Uses aiosqlite for async access.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

# Max tokens allowed when injecting prior summaries into context
MAX_SUMMARY_TOKENS = 2000
# Rough token estimate: 1 token ≈ 4 chars
CHARS_PER_TOKEN = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'INIT',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    summary TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS task_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    tool_name TEXT,
    parameters TEXT,
    output TEXT,
    success INTEGER,
    duration_ms INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS persona_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class MemoryStore:
    """Async SQLite memory store."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def init_db(self) -> None:
        """Initialise the database and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Tasks ──

    async def save_task(self, task_id: str, data: dict) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT OR REPLACE INTO tasks 
               (task_id, description, status, created_at, updated_at, summary, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                data.get("description", ""),
                data.get("status", "INIT"),
                data.get("created_at", now),
                now,
                data.get("summary", ""),
                json.dumps(data.get("metadata", {})),
            ),
        )
        await self._db.commit()

    async def get_task(self, task_id: str) -> Optional[dict]:
        cursor = await self._db.execute(
            "SELECT task_id, description, status, created_at, updated_at, summary, metadata FROM tasks WHERE task_id = ?",
            (task_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "task_id": row[0],
            "description": row[1],
            "status": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "summary": row[5],
            "metadata": json.loads(row[6]),
        }

    async def update_task_status(self, task_id: str, status: str, summary: str = "") -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE tasks SET status = ?, summary = ?, updated_at = ? WHERE task_id = ?",
            (status, summary, now, task_id),
        )
        await self._db.commit()

    # ── Task Steps ──

    async def save_step(self, task_id: str, step_data: dict) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO task_steps 
               (task_id, iteration, tool_name, parameters, output, success, duration_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                step_data.get("iteration", 0),
                step_data.get("tool_name", ""),
                json.dumps(step_data.get("parameters", {})),
                step_data.get("output", "")[:20000],  # truncate
                1 if step_data.get("success") else 0,
                step_data.get("duration_ms", 0),
                now,
            ),
        )
        await self._db.commit()

    # ── Summaries (with token cap) ──

    async def get_recent_summaries(self, limit: int = 3) -> list[str]:
        """
        Get recent task summaries, capped at MAX_SUMMARY_TOKENS total.
        """
        cursor = await self._db.execute(
            "SELECT summary FROM tasks WHERE summary != '' ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        summaries = []
        total_chars = 0
        max_chars = MAX_SUMMARY_TOKENS * CHARS_PER_TOKEN

        for row in rows:
            summary = row[0]
            if total_chars + len(summary) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 50:
                    summaries.append(summary[:remaining] + "...")
                break
            summaries.append(summary)
            total_chars += len(summary)

        return summaries

    # ── Preferences ──

    async def save_preference(self, key: str, value: str) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        await self._db.commit()

    async def get_preference(self, key: str) -> Optional[str]:
        cursor = await self._db.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    # ── Persona State ──

    async def save_persona_state(self, key: str, value: str) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT OR REPLACE INTO persona_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        await self._db.commit()

    async def get_persona_state(self, key: str) -> Optional[str]:
        cursor = await self._db.execute(
            "SELECT value FROM persona_state WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None
```

---

## Test File

### `aira/tests/test_memory.py`

```python
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
```

## Verification

```powershell
pytest aira/tests/test_memory.py -v
```

## Done When
- [ ] `aira/memory/memory_store.py` created
- [ ] `aira/tests/test_memory.py` created
- [ ] All tests pass

## Next Phase
→ `phase_06_local_llm.md`
