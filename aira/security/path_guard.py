"""
Path security guard — blocks workspace escapes, symlinks, and sensitive files.
"""

import re
from pathlib import Path
from typing import List

# Extensions that must NEVER be read or written
BLOCKED_EXTENSIONS: List[str] = [
    ".env", ".pem", ".key", ".pfx", ".p12", ".jks",
    ".keystore", ".crt", ".cer",
]

# Filename patterns that must NEVER be accessed
BLOCKED_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\.env.*", re.IGNORECASE),
    re.compile(r"^credentials.*", re.IGNORECASE),
    re.compile(r"^\.ssh", re.IGNORECASE),
    re.compile(r"^id_rsa", re.IGNORECASE),
    re.compile(r"^id_ed25519", re.IGNORECASE),
]


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


def validate_path(target: str | Path, workspace_root: Path) -> Path:
    """
    Validate that a path is safe to operate on.

    Args:
        target: The path to validate (relative or absolute).
        workspace_root: The resolved workspace root directory.

    Returns:
        The resolved, validated Path.

    Raises:
        SecurityError: If the path violates any security constraint.
    """
    workspace_root = Path(workspace_root).resolve()
    resolved = (workspace_root / Path(target)).resolve()

    # 1. Must be inside workspace
    if not resolved.is_relative_to(workspace_root):
        raise SecurityError(
            f"Path escape blocked: '{target}' resolves to '{resolved}' "
            f"which is outside workspace '{workspace_root}'"
        )

    # 2. Must not be a symlink
    if resolved.exists() and resolved.is_symlink():
        raise SecurityError(
            f"Symlink blocked: '{resolved}' is a symbolic link"
        )

    # 3. Check parents for symlinks too
    for parent in resolved.parents:
        if parent == workspace_root:
            break
        if parent.exists() and parent.is_symlink():
            raise SecurityError(
                f"Symlink in path blocked: '{parent}' is a symbolic link"
            )

    # 4. Check blocked extensions
    suffix = resolved.suffix.lower()
    if suffix in BLOCKED_EXTENSIONS:
        raise SecurityError(
            f"Blocked extension: '{suffix}' is forbidden"
        )

    # 5. Check blocked filename patterns
    name = resolved.name
    for pattern in BLOCKED_PATTERNS:
        if pattern.match(name):
            raise SecurityError(
                f"Blocked filename pattern: '{name}' matches '{pattern.pattern}'"
            )

    # 6. Check for binary content (if file exists and is being read)
    if resolved.exists() and resolved.is_file():
        try:
            with open(resolved, "rb") as f:
                chunk = f.read(8192)
                if b"\x00" in chunk:
                    raise SecurityError(
                        f"Binary file blocked: '{resolved}' contains null bytes"
                    )
        except OSError:
            raise SecurityError(f"Cannot read file: '{resolved}'")

    return resolved
