"""Tests for the agent loop."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from aira.agent.loop import AgentLoop
from aira.agent.state import TaskState
from aira.agent.state_machine import StateMachine
from aira.agent.completion import is_task_complete
from aira.tools.registry import ToolRegistry, ToolResult
from aira.tools.file_read import FileReadTool
from aira.tools.file_write import FileWriteTool
from aira.security.cost_controller import CostController
from aira.memory.memory_store import MemoryStore


# ── Completion checker tests ──

def test_completion_all_true():
    assert is_task_complete(True, True, [], True) is True


def test_completion_planner_not_done():
    assert is_task_complete(False, True, [], True) is False


def test_completion_tool_failed():
    assert is_task_complete(True, False, [], True) is False


def test_completion_validation_errors():
    assert is_task_complete(True, True, ["error"], True) is False


def test_completion_output_invalid():
    assert is_task_complete(True, True, [], False) is False


# ── Agent loop tests ──

@pytest_asyncio.fixture
async def setup(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create a test file in workspace
    (workspace / "hello.txt").write_text("Hello World", encoding="utf-8")

    # Set up components
    registry = ToolRegistry()
    registry.register("file_read", FileReadTool(workspace))
    registry.register("file_write", FileWriteTool(workspace))

    cost_ctrl = CostController(max_cost_usd=1.00)
    memory = MemoryStore(tmp_path / "test.db")
    await memory.init_db()

    # Mock local LLM that returns a valid plan
    local_llm = AsyncMock()
    local_llm.generate.return_value = (
        '{"steps": [{"tool": "file_read", "params": {"file_path": "hello.txt"}, '
        '"description": "read file"}], "confidence": 0.9, "done": true}'
    )

    planner = AsyncMock()
    sm_factory = lambda tid: StateMachine(tid, log_dir)

    loop = AgentLoop(
        state_machine_factory=sm_factory,
        tool_registry=registry,
        planner=planner,
        local_llm=local_llm,
        memory=memory,
        cost_controller=cost_ctrl,
        max_iterations=15,
        log_dir=log_dir,
    )

    return loop, workspace, memory


@pytest.mark.asyncio
async def test_loop_completes_simple_task(setup):
    loop, workspace, memory = setup
    result = await loop.run("Read hello.txt", "test-001")
    assert result["final_state"] == "COMPLETED"
    assert result["iterations"] >= 1


@pytest.mark.asyncio
async def test_loop_max_iterations(setup):
    loop, workspace, memory = setup
    # Make LLM always return plan that's not done
    loop.local_llm.generate.return_value = (
        '{"steps": [{"tool": "file_read", "params": {"file_path": "hello.txt"}, '
        '"description": "read file"}], "confidence": 0.5, "done": false}'
    )
    loop.max_iterations = 3
    result = await loop.run("Never-ending task", "test-002")
    # It should fail because max iterations exceeded
    assert result["final_state"] == "FAILED"
    assert result["iterations"] <= 3
