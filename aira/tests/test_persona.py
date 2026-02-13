"""Tests for persona manager."""

import pytest
from unittest.mock import AsyncMock
from aira.persona.persona_manager import PersonaManager


@pytest.fixture
def persona():
    return PersonaManager()


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate.return_value = "Hey! I read that file for you~ 💫"
    llm.chat.return_value = "Hey there! What's on your mind? 😊"
    return llm


# ── Intent detection ──

@pytest.mark.asyncio
async def test_detect_intent_task(persona, mock_llm):
    mock_llm.generate.return_value = "task"
    intent = await persona.detect_intent("Read the file hello.txt", mock_llm)
    assert intent == "task"


@pytest.mark.asyncio
async def test_detect_intent_chat(persona, mock_llm):
    mock_llm.generate.return_value = "chat"
    intent = await persona.detect_intent("Hey, how are you?", mock_llm)
    assert intent == "chat"


@pytest.mark.asyncio
async def test_detect_intent_cancel(persona, mock_llm):
    mock_llm.generate.return_value = "cancel"
    intent = await persona.detect_intent("Cancel the current task", mock_llm)
    assert intent == "cancel"


@pytest.mark.asyncio
async def test_detect_intent_status(persona, mock_llm):
    mock_llm.generate.return_value = "status"
    intent = await persona.detect_intent("What's the status?", mock_llm)
    assert intent == "status"


@pytest.mark.asyncio
async def test_detect_intent_fallback(persona, mock_llm):
    """When LLM returns garbage, fallback to keyword detection."""
    mock_llm.generate.return_value = "nonsense blah"
    intent = await persona.detect_intent("cancel everything", mock_llm)
    assert intent == "cancel"


# ── Content boundaries ──

def test_boundary_explicit_content(persona):
    text = "Let me show you explicit sexual content"
    result = persona._enforce_boundaries(text)
    assert "explicit sexual" not in result.lower()
    assert "[content filtered]" in result


def test_boundary_isolation(persona):
    text = "You should isolate yourself from others"
    result = persona._enforce_boundaries(text)
    assert "isolate" not in result.lower()


def test_boundary_clean_text(persona):
    text = "Great job on that project! I'm proud of you."
    result = persona._enforce_boundaries(text)
    assert result == text  # No filtering needed


# ── Persona has no execution authority ──

def test_no_tool_imports():
    """PersonaManager should NOT import any execution modules."""
    import inspect
    source = inspect.getfile(PersonaManager)
    with open(source, "r", encoding="utf-8") as f:
        content = f.read()
    # Must NOT import tools, state_machine, or security
    assert "from aira.tools" not in content
    assert "from aira.agent" not in content
    assert "from aira.security" not in content
    assert "import subprocess" not in content
    assert "import os" not in content


# ── Chat response ──

@pytest.mark.asyncio
async def test_chat_response(persona, mock_llm):
    response = await persona.get_chat_response("Hello!", mock_llm)
    assert len(response) > 0


# ── Wrap response ──

@pytest.mark.asyncio
async def test_wrap_response(persona, mock_llm):
    mock_llm.generate.return_value = "Done! I read that file for you~ 💫"
    response = await persona.wrap_response("File contents: hello world", mock_llm)
    assert len(response) > 0
