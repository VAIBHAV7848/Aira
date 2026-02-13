"""Tests for path_guard security module."""

import os
import tempfile
from pathlib import Path
import pytest
from aira.security.path_guard import validate_path, SecurityError


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def test_valid_path_inside_workspace(workspace):
    """A path inside workspace should resolve correctly."""
    test_file = workspace / "test.txt"
    test_file.write_text("hello")
    result = validate_path("test.txt", workspace)
    assert result == test_file.resolve()


def test_traversal_blocked(workspace):
    """Path traversal outside workspace should be blocked."""
    with pytest.raises(SecurityError, match="Path escape"):
        validate_path("../../etc/passwd", workspace)


def test_absolute_path_outside_blocked(workspace):
    """Absolute path outside workspace should be blocked."""
    with pytest.raises(SecurityError, match="Path escape"):
        validate_path("C:\\Windows\\System32\\config", workspace)


def test_blocked_extension_env(workspace):
    """Files with .env extension should be blocked."""
    with pytest.raises(SecurityError, match="Blocked extension"):
        validate_path("secrets.env", workspace)


def test_blocked_extension_pem(workspace):
    """Files with .pem extension should be blocked."""
    with pytest.raises(SecurityError, match="Blocked extension"):
        validate_path("cert.pem", workspace)


def test_blocked_extension_key(workspace):
    """Files with .key extension should be blocked."""
    with pytest.raises(SecurityError, match="Blocked extension"):
        validate_path("private.key", workspace)


def test_blocked_pattern_dotenv(workspace):
    """Files matching .env* pattern should be blocked."""
    with pytest.raises(SecurityError, match="Blocked"):
        validate_path(".env.local", workspace)


def test_blocked_pattern_credentials(workspace):
    """Files matching credentials* should be blocked."""
    with pytest.raises(SecurityError, match="Blocked"):
        validate_path("credentials.json", workspace)


def test_binary_file_blocked(workspace):
    """Binary files should be blocked."""
    binary = workspace / "image.png"
    binary.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00something")
    with pytest.raises(SecurityError, match="Binary file"):
        validate_path("image.png", workspace)


def test_text_file_allowed(workspace):
    """Normal text files should pass."""
    text = workspace / "readme.md"
    text.write_text("# Hello")
    result = validate_path("readme.md", workspace)
    assert result == text.resolve()


def test_nonexistent_file_allowed(workspace):
    """Paths to nonexistent files should be allowed (for writes)."""
    result = validate_path("new_file.txt", workspace)
    assert result.name == "new_file.txt"


def test_subdirectory_allowed(workspace):
    """Paths in subdirectories should work."""
    sub = workspace / "subdir"
    sub.mkdir()
    f = sub / "file.txt"
    f.write_text("test")
    result = validate_path("subdir/file.txt", workspace)
    assert result == f.resolve()
