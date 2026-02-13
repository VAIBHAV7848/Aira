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
