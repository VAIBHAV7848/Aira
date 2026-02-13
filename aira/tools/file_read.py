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
