"""Tests for file_read, file_write, and python_runner tools."""

import pytest
from aira.tools.file_read import FileReadTool
from aira.tools.file_write import FileWriteTool
from aira.tools.python_runner import PythonRunnerTool
from aira.tools.registry import ToolRegistry, ToolNotFoundError


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


# ── Registry ──

def test_registry_register_and_get():
    reg = ToolRegistry()
    reg.register("test", lambda: None)
    assert reg.has("test")
    assert "test" in reg.list_tools()


def test_registry_not_found():
    reg = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        reg.get("nonexistent")


# ── File Read ──

def test_file_read_success(workspace):
    f = workspace / "hello.txt"
    f.write_text("hello world")
    tool = FileReadTool(workspace)
    result = tool.run("hello.txt")
    assert result.success is True
    assert "hello world" in result.output


def test_file_read_not_found(workspace):
    tool = FileReadTool(workspace)
    result = tool.run("missing.txt")
    assert result.success is False
    assert "not found" in result.error.lower()


def test_file_read_outside_workspace(workspace):
    tool = FileReadTool(workspace)
    result = tool.run("../../etc/passwd")
    assert result.success is False
    assert "security" in result.error.lower()


def test_file_read_env_blocked(workspace):
    tool = FileReadTool(workspace)
    result = tool.run(".env")
    assert result.success is False


def test_file_read_truncation(workspace):
    big = workspace / "big.txt"
    big.write_text("a" * 25000)
    tool = FileReadTool(workspace, max_chars=1000)
    result = tool.run("big.txt")
    assert result.success is True
    assert result.metadata["truncated"] is True
    assert len(result.output) == 1000


# ── File Write ──

def test_file_write_success(workspace):
    tool = FileWriteTool(workspace)
    result = tool.run("output.txt", "test content")
    assert result.success is True
    assert (workspace / "output.txt").read_text() == "test content"


def test_file_write_creates_subdirs(workspace):
    tool = FileWriteTool(workspace)
    result = tool.run("sub/dir/file.txt", "nested")
    assert result.success is True
    assert (workspace / "sub" / "dir" / "file.txt").read_text() == "nested"


def test_file_write_env_blocked(workspace):
    tool = FileWriteTool(workspace)
    result = tool.run(".env", "SECRET=bad")
    assert result.success is False


def test_file_write_outside_workspace(workspace):
    tool = FileWriteTool(workspace)
    result = tool.run("../../escape.txt", "bad")
    assert result.success is False


# ── Python Runner ──

def test_python_runner_success(workspace):
    script = workspace / "hello.py"
    script.write_text("print('hello from script')")
    tool = PythonRunnerTool(workspace, timeout=10)
    result = tool.run("hello.py")
    assert result.success is True
    assert "hello from script" in result.output


def test_python_runner_error(workspace):
    script = workspace / "error.py"
    script.write_text("raise ValueError('test error')")
    tool = PythonRunnerTool(workspace, timeout=10)
    result = tool.run("error.py")
    assert result.success is False
    assert result.metadata["return_code"] != 0


def test_python_runner_not_found(workspace):
    tool = PythonRunnerTool(workspace)
    result = tool.run("missing.py")
    assert result.success is False


def test_python_runner_non_py(workspace):
    f = workspace / "script.sh"
    f.write_text("echo hi")
    tool = PythonRunnerTool(workspace)
    result = tool.run("script.sh")
    assert result.success is False
    assert "not a python file" in result.error.lower()


def test_python_runner_timeout(workspace):
    script = workspace / "slow.py"
    script.write_text("import time; time.sleep(60)")
    tool = PythonRunnerTool(workspace, timeout=2)
    result = tool.run("slow.py")
    assert result.success is False
    assert "timed out" in result.error.lower()


def test_python_runner_outside_workspace(workspace):
    tool = PythonRunnerTool(workspace)
    result = tool.run("../../etc/hack.py")
    assert result.success is False
