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
