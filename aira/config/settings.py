"""
Aira configuration — loads .env and exposes a typed Settings dataclass.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv


def _load_env() -> None:
    """Load .env from project root."""
    # Walk up from this file to find .env
    project_root = Path(__file__).resolve().parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


_load_env()


def _get_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _get_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _get_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _get_path(key: str, default: str = ".") -> Path:
    return Path(os.getenv(key, default)).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""

    TELEGRAM_BOT_TOKEN: str = field(
        default_factory=lambda: _get_str("TELEGRAM_BOT_TOKEN")
    )
    WORKSPACE_ROOT: Path = field(
        default_factory=lambda: _get_path("WORKSPACE_ROOT", "./aira/workspace")
    )
    MAX_COST_USD: float = field(
        default_factory=lambda: _get_float("MAX_COST_USD", 1.00)
    )
    COST_WARNING_THRESHOLD: float = field(
        default_factory=lambda: _get_float("COST_WARNING_THRESHOLD", 0.80)
    )
    MAX_ITERATIONS: int = field(
        default_factory=lambda: _get_int("MAX_ITERATIONS", 15)
    )
    SUBPROCESS_TIMEOUT: int = field(
        default_factory=lambda: _get_int("SUBPROCESS_TIMEOUT", 30)
    )
    MAX_OUTPUT_CHARS: int = field(
        default_factory=lambda: _get_int("MAX_OUTPUT_CHARS", 20000)
    )
    OLLAMA_MODEL: str = field(
        default_factory=lambda: _get_str("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    )
    OLLAMA_BASE_URL: str = field(
        default_factory=lambda: _get_str("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    EXTERNAL_LLM_MODEL: str = field(
        default_factory=lambda: _get_str("EXTERNAL_LLM_MODEL", "")
    )
    EXTERNAL_LLM_API_KEY: str = field(
        default_factory=lambda: _get_str("EXTERNAL_LLM_API_KEY", "")
    )
    LOG_DIR: Path = field(
        default_factory=lambda: _get_path("LOG_DIR", "./aira/logs")
    )
    DB_PATH: Path = field(
        default_factory=lambda: _get_path("DB_PATH", "./aira.db")
    )
