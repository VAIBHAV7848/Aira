# Phase 6 — Local LLM Integration (Qwen via Ollama)

## Goal
Lazy-loading Ollama wrapper that loads Qwen 2.5 7B on demand and unloads after use.

## Prerequisites
- Phase 1 config passing
- Ollama installed and running (`ollama serve`)
- Qwen model pulled (`ollama pull qwen2.5:7b-instruct`)

## File to Create

### `aira/llm/local_qwen.py`

```python
"""
Local LLM wrapper — async Ollama client for Qwen 2.5 7B.
Loads lazily, unloads after response to free VRAM.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_KEEP_ALIVE = "5m"   # Keep model hot for 5 minutes for conversation bursts
UNLOAD_KEEP_ALIVE = "0"     # Immediately free VRAM


class OllamaError(Exception):
    """Raised when Ollama API errors occur."""
    pass


class LocalQwen:
    """Async wrapper for Qwen 2.5 7B via Ollama's REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system: str = "",
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> str:
        """
        Generate a response from the local model.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            keep_alive: How long to keep model loaded after response.

        Returns:
            The generated text response.

        Raises:
            OllamaError: If the API call fails.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("response", "")
                logger.info(
                    f"Qwen generated {len(text)} chars "
                    f"(eval_count={data.get('eval_count', '?')})"
                )
                return text

        except httpx.TimeoutException:
            raise OllamaError(
                f"Ollama timed out after {self.timeout}s. "
                "Is Ollama running? Try: ollama serve"
            )
        except httpx.ConnectError:
            raise OllamaError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            raise OllamaError(f"Ollama API error: {e.response.status_code}")
        except Exception as e:
            raise OllamaError(f"Unexpected error: {e}")

    async def chat(
        self,
        messages: list[dict],
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> str:
        """
        Chat-style completion using Ollama's /api/chat endpoint.

        Args:
            messages: List of {"role": "user"|"system"|"assistant", "content": "..."} dicts.
            keep_alive: How long to keep model loaded.

        Returns:
            The assistant's response text.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("message", {}).get("content", "")
                return text

        except httpx.TimeoutException:
            raise OllamaError(f"Ollama chat timed out after {self.timeout}s")
        except httpx.ConnectError:
            raise OllamaError(f"Cannot connect to Ollama at {self.base_url}")
        except Exception as e:
            raise OllamaError(f"Chat error: {e}")

    async def unload(self) -> None:
        """Unload the model from VRAM immediately."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "",
                        "keep_alive": UNLOAD_KEEP_ALIVE,
                    },
                )
            logger.info(f"Model {self.model} unloaded from VRAM")
        except Exception as e:
            logger.warning(f"Failed to unload model: {e}")

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is accessible."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    names = [m.get("name", "") for m in models]
                    return any(self.model in n for n in names)
            return False
        except Exception:
            return False
```

---

## Test File

### `aira/tests/test_local_qwen.py`

```python
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
```

## Verification

```powershell
pytest aira/tests/test_local_qwen.py -v
```

## Done When
- [ ] `aira/llm/local_qwen.py` created
- [ ] `aira/tests/test_local_qwen.py` created
- [ ] All tests pass (mocked — no Ollama needed)

## Next Phase
→ `phase_07_external_planner.md`
