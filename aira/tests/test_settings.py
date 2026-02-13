"""Tests for Settings configuration."""

import os
from unittest.mock import patch
from aira.config.settings import Settings


def test_settings_defaults():
    """Settings should have sensible defaults even without .env."""
    s = Settings()
    assert s.MAX_COST_USD == 1.00
    assert s.COST_WARNING_THRESHOLD == 0.80
    assert s.MAX_ITERATIONS == 15
    assert s.SUBPROCESS_TIMEOUT == 30
    assert s.MAX_OUTPUT_CHARS == 20000
    assert s.OLLAMA_MODEL == "qwen2.5:7b-instruct"
    assert s.OLLAMA_BASE_URL == "http://localhost:11434"


def test_settings_from_env():
    """Settings should read from environment variables."""
    with patch.dict(os.environ, {"MAX_COST_USD": "5.00", "MAX_ITERATIONS": "10"}):
        s = Settings()
        assert s.MAX_COST_USD == 5.00
        assert s.MAX_ITERATIONS == 10


def test_settings_workspace_is_path():
    """WORKSPACE_ROOT should be a resolved Path."""
    s = Settings()
    assert hasattr(s.WORKSPACE_ROOT, "resolve")
    assert s.WORKSPACE_ROOT.is_absolute()
