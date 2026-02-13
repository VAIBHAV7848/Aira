"""Tests for LocalQwen (mocked — no real Ollama needed)."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from aira.llm.local_qwen import LocalQwen, OllamaError


@pytest.mark.asyncio
async def test_generate_returns_text():
    """generate() should return the response text."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "Hello! How can I help?",
        "eval_count": 10,
    }

    with patch("httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post.return_value = mock_response
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        llm = LocalQwen()
        result = await llm.generate("say hello")
        assert result == "Hello! How can I help?"


@pytest.mark.asyncio
async def test_generate_connection_error():
    """generate() should raise OllamaError on connection failure."""
    with patch("httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post.side_effect = httpx.ConnectError("refused")
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        llm = LocalQwen()
        with pytest.raises(OllamaError, match="Cannot connect"):
            await llm.generate("hello")


@pytest.mark.asyncio
async def test_generate_timeout():
    """generate() should raise OllamaError on timeout."""
    with patch("httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post.side_effect = httpx.TimeoutException("timeout")
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        llm = LocalQwen()
        with pytest.raises(OllamaError, match="timed out"):
            await llm.generate("hello")


@pytest.mark.asyncio
async def test_unload_sends_zero_keepalive():
    """unload() should send keep_alive='0' to free VRAM."""
    with patch("httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        instance.post.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        llm = LocalQwen()
        await llm.unload()

        call_args = instance.post.call_args
        payload = call_args[1]["json"]
        assert payload["keep_alive"] == "0"
