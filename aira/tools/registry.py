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
