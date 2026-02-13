"""Tests for Telegram adapter (no real Telegram — just routing logic)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aira.adapters.telegram_adapter import TelegramAdapter
from aira.config.settings import Settings
from aira.persona.persona_manager import PersonaManager


@pytest.fixture
def adapter():
    settings = MagicMock(spec=Settings)
    settings.TELEGRAM_BOT_TOKEN = "test-token"
    settings.ALLOWED_USER_IDS = []  # Allow all for basic tests

    agent_loop = MagicMock()
    agent_loop.cost_ctrl = MagicMock()
    agent_loop.cost_ctrl.get_summary.return_value = {
        "total_spent": 0.0,
        "max_budget": 1.00,
        "remaining": 1.00,
        "usage_ratio": 0.0,
        "call_count": 0,
        "warning": False,
    }

    persona = PersonaManager()
    local_llm = AsyncMock()
    memory = AsyncMock()

    return TelegramAdapter(settings, agent_loop, persona, local_llm, memory)


def test_adapter_creation(adapter):
    assert adapter is not None
    assert adapter._current_task_id is None


def test_no_token_raises():
    settings = MagicMock(spec=Settings)
    settings.TELEGRAM_BOT_TOKEN = ""
    settings.ALLOWED_USER_IDS = []
    adapter = TelegramAdapter(settings, MagicMock(), PersonaManager(), AsyncMock(), AsyncMock())
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        adapter.run()

@pytest.mark.asyncio
async def test_auth_check_failure():
    settings = MagicMock(spec=Settings)
    settings.TELEGRAM_BOT_TOKEN = "token"
    settings.ALLOWED_USER_IDS = [12345]
    
    adapter = TelegramAdapter(settings, MagicMock(), PersonaManager(), AsyncMock(), AsyncMock())
    
    update = MagicMock()
    update.effective_user.id = 99999  # Unauthorized
    update.message.reply_text = AsyncMock()
    
    allowed = await adapter._check_auth(update)
    assert allowed is False
    update.message.reply_text.assert_called_with("⛔ Unauthorized access.")

@pytest.mark.asyncio
async def test_auth_check_success():
    settings = MagicMock(spec=Settings)
    settings.TELEGRAM_BOT_TOKEN = "token"
    settings.ALLOWED_USER_IDS = [12345]
    
    adapter = TelegramAdapter(settings, MagicMock(), PersonaManager(), AsyncMock(), AsyncMock())
    
    update = MagicMock()
    update.effective_user.id = 12345  # Authorized
    
    allowed = await adapter._check_auth(update)
    assert allowed is True
