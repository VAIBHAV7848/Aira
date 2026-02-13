# Phase 4 — Tools

## Goal
Build three sandboxed tools (file_read, file_write, python_runner) and a registry to manage them.

## Prerequisites
- Phase 2 (security guards passing) + Phase 3 (state machine passing)

## Files to Create

### `aira/tools/registry.py`

```python
"""
Tool registry — maps tool names to callable tool instances.
Each tool returns a structured dict: {"success": bool, "output": str, "error": str|None, "metadata": dict}
"""

from typing import Callable, Any


class ToolNotFoundError(Exception):
    pass


class ToolResult:
    """Structured tool output."""

    def __init__(
        self, success: bool, output: str, error: str | None = None, metadata: dict | None = None
    ):
        self.success = success
        self.output = output
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


class ToolRegistry:
    """Stores and retrieves tools by name."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        self._tools[name] = func

    def get(self, name: str) -> Callable:
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool not found: '{name}'")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        return name in self._tools
```

---

### `aira/tools/file_read.py`

```python
"""
File read tool — reads a file inside the workspace with security validation.
"""

from pathlib import Path
from aira.security.path_guard import validate_path, SecurityError
from aira.tools.registry import ToolResult


class FileReadTool:
    """Reads files inside the workspace boundary."""

    def __init__(self, workspace_root: Path, max_chars: int = 20000):
        self.workspace_root = Path(workspace_root).resolve()
        self.max_chars = max_chars

    def run(self, file_path: str) -> ToolResult:
        """
        Read a file from workspace.

        Args:
            file_path: Relative path inside workspace.

        Returns:
            ToolResult with file contents or error.
        """
        try:
            resolved = validate_path(file_path, self.workspace_root)

            if not resolved.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {file_path}",
                )

            if not resolved.is_file():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a file: {file_path}",
                )

            content = resolved.read_text(encoding="utf-8")
            truncated = False

            if len(content) > self.max_chars:
                content = content[: self.max_chars]
                truncated = True

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "path": str(resolved),
                    "size": len(content),
                    "truncated": truncated,
                },
            )

        except SecurityError as e:
            return ToolResult(success=False, output="", error=f"Security: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Error: {e}")
```

---

### `aira/tools/file_write.py`

```python
"""
File write tool — writes to a file inside the workspace with security validation.
"""

from pathlib import Path
from aira.security.path_guard import validate_path, SecurityError
from aira.tools.registry import ToolResult


class FileWriteTool:
    """Writes files inside the workspace boundary."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).resolve()

    def run(self, file_path: str, content: str) -> ToolResult:
        """
        Write content to a file in workspace.

        Args:
            file_path: Relative path inside workspace.
            content: Text content to write.

        Returns:
            ToolResult with success/failure.
        """
        try:
            resolved = validate_path(file_path, self.workspace_root)

            # Create parent directories if needed
            resolved.parent.mkdir(parents=True, exist_ok=True)

            resolved.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                output=f"Written {len(content)} chars to {file_path}",
                metadata={
                    "path": str(resolved),
                    "size": len(content),
                },
            )

        except SecurityError as e:
            return ToolResult(success=False, output="", error=f"Security: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Error: {e}")
```

---

### `aira/tools/python_runner.py`

```python
"""
Python runner tool — executes Python scripts inside workspace with strict sandboxing.
"""

import subprocess
import sys
from pathlib import Path
from aira.security.path_guard import validate_path, SecurityError
from aira.tools.registry import ToolResult


class PythonRunnerTool:
    """Runs Python scripts with strict sandboxing."""

    def __init__(self, workspace_root: Path, timeout: int = 30):
        self.workspace_root = Path(workspace_root).resolve()
        self.timeout = timeout

    def run(self, script_path: str) -> ToolResult:
        """
        Execute a Python script inside the workspace.

        Args:
            script_path: Relative path to .py file inside workspace.

        Returns:
            ToolResult with stdout, stderr, return code.
        """
        try:
            resolved = validate_path(script_path, self.workspace_root)

            if not resolved.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Script not found: {script_path}",
                )

            if resolved.suffix != ".py":
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a Python file: {script_path}",
                )

            # Execute with strict sandboxing
            result = subprocess.run(
                [sys.executable, str(resolved)],
                shell=False,           # NEVER shell=True
                cwd=str(self.workspace_root),
                env={},                # Empty env — no system secrets
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )

            return ToolResult(
                success=(result.returncode == 0),
                output=result.stdout[:20000] if result.stdout else "",
                error=result.stderr[:5000] if result.stderr else None,
                metadata={
                    "return_code": result.returncode,
                    "script": str(resolved),
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Script timed out after {self.timeout}s",
                metadata={"timeout": self.timeout},
            )
        except SecurityError as e:
            return ToolResult(success=False, output="", error=f"Security: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Error: {e}")
```

---

## Test File

### `aira/tests/test_tools.py`

```python
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
```

## Verification

```powershell
pytest aira/tests/test_tools.py -v
```

## Done When
- [ ] `aira/tools/registry.py` created
- [ ] `aira/tools/file_read.py` created
- [ ] `aira/tools/file_write.py` created
- [ ] `aira/tools/python_runner.py` created
- [ ] `aira/tests/test_tools.py` created
- [ ] All tests pass

## Next Phase
→ `phase_05_memory.md`
