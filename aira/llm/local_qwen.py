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
