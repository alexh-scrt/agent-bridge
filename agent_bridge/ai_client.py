"""Async AI client for agent_bridge.

Provides an httpx-based async client that sends prompts to either an
OpenAI-compatible API endpoint or a local Ollama instance, with optional
streaming response support.

Typical usage::

    client = AIClient(settings)
    async with client:
        result = await client.complete("Write a bubble sort in Python")
        print(result)

    # Or with streaming:
    async with client:
        async for chunk in client.stream("Explain async generators"):
            print(chunk, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from agent_bridge.config import Settings

logger = logging.getLogger(__name__)


class AIClientError(Exception):
    """Raised when the AI backend returns an error or an unexpected response."""


class AIClientTimeoutError(AIClientError):
    """Raised when the AI backend request times out."""


class AIClientConnectionError(AIClientError):
    """Raised when the AI backend cannot be reached."""


class AIClient:
    """Async client for interacting with an OpenAI-compatible or Ollama AI backend.

    Supports both streaming and non-streaming completions. Internally uses
    :class:`httpx.AsyncClient` for all HTTP communication.

    Args:
        settings: Application settings containing AI backend configuration.
        http_client: Optional pre-constructed :class:`httpx.AsyncClient` for
            dependency injection (useful in tests).
    """

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._owns_client = http_client is None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AIClient":
        """Enter the async context manager, creating an HTTP client if needed."""
        if self._owns_client:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self._settings.ai_timeout_seconds,
                    write=30.0,
                    pool=10.0,
                )
            )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager, closing the HTTP client if we own it."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_client(self) -> httpx.AsyncClient:
        """Return the active HTTP client or raise if not initialised.

        Returns:
            The active :class:`httpx.AsyncClient`.

        Raises:
            RuntimeError: If the client has not been started via the context
                manager or :meth:`__aenter__`.
        """
        if self._http_client is None:
            raise RuntimeError(
                "AIClient is not initialised. Use it as an async context manager "
                "or call await client.__aenter__() first."
            )
        return self._http_client

    def _build_openai_payload(self, prompt: str, stream: bool = False) -> dict[str, Any]:
        """Build the request payload for an OpenAI-compatible chat completions endpoint.

        Args:
            prompt: The user's prompt text.
            stream: Whether to request a streaming response.

        Returns:
            A dict suitable for JSON-encoding and sending as the request body.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._settings.ai_system_prompt},
            {"role": "user", "content": prompt},
        ]
        payload: dict[str, Any] = {
            "model": self._settings.ai_model,
            "messages": messages,
            "max_tokens": self._settings.ai_max_tokens,
            "temperature": self._settings.ai_temperature,
            "stream": stream,
        }
        return payload

    def _build_ollama_payload(self, prompt: str, stream: bool = False) -> dict[str, Any]:
        """Build the request payload for the Ollama /api/chat endpoint.

        Args:
            prompt: The user's prompt text.
            stream: Whether to request a streaming response.

        Returns:
            A dict suitable for JSON-encoding and sending as the request body.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._settings.ai_system_prompt},
            {"role": "user", "content": prompt},
        ]
        payload: dict[str, Any] = {
            "model": self._settings.ai_model,
            "messages": messages,
            "options": {
                "num_predict": self._settings.ai_max_tokens,
                "temperature": self._settings.ai_temperature,
            },
            "stream": stream,
        }
        return payload

    def _get_endpoint_url(self) -> str:
        """Return the appropriate completion endpoint URL based on backend type.

        Returns:
            The full URL string for the configured AI backend endpoint.
        """
        if self._settings.ai_backend_type == "openai":
            return self._settings.openai_chat_completions_url
        return self._settings.ollama_generate_url

    def _get_headers(self) -> dict[str, str]:
        """Build HTTP headers for the AI backend request.

        Returns:
            A dict of HTTP headers. Includes ``Authorization`` if an API key
            is configured.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.ai_api_key is not None:
            api_key = self._settings.ai_api_key.get_secret_value()
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(self, prompt: str, stream: bool = False) -> dict[str, Any]:
        """Dispatch to the appropriate payload builder based on backend type.

        Args:
            prompt: The user's prompt text.
            stream: Whether to request a streaming response.

        Returns:
            A complete request payload dict.
        """
        if self._settings.ai_backend_type == "openai":
            return self._build_openai_payload(prompt, stream=stream)
        return self._build_ollama_payload(prompt, stream=stream)

    # ------------------------------------------------------------------
    # OpenAI response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_openai_response(data: dict[str, Any]) -> str:
        """Extract the assistant's message text from an OpenAI completion response.

        Args:
            data: Parsed JSON response body from the completions endpoint.

        Returns:
            The content string from the first choice.

        Raises:
            AIClientError: If the response structure is unexpected.
        """
        try:
            choices = data["choices"]
            if not choices:
                raise AIClientError("OpenAI response contained no choices")
            content = choices[0]["message"]["content"]
            if content is None:
                return ""
            return str(content)
        except (KeyError, IndexError, TypeError) as exc:
            raise AIClientError(
                f"Unexpected OpenAI response structure: {exc}"
            ) from exc

    @staticmethod
    def _parse_openai_stream_chunk(line: str) -> str | None:
        """Parse a single SSE line from an OpenAI streaming response.

        Args:
            line: A raw text line from the streaming response.

        Returns:
            The delta content string if present, ``None`` for non-content lines
            (e.g. ``[DONE]`` or empty lines).
        """
        line = line.strip()
        if not line or line == "data: [DONE]":
            return None
        if line.startswith("data: "):
            json_part = line[len("data: "):]
            try:
                data = json.loads(json_part)
                choices = data.get("choices", [])
                if not choices:
                    return None
                delta = choices[0].get("delta", {})
                return delta.get("content")  # may be None
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                return None
        return None

    # ------------------------------------------------------------------
    # Ollama response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ollama_response(data: dict[str, Any]) -> str:
        """Extract the assistant's message text from an Ollama /api/chat response.

        Args:
            data: Parsed JSON response body from the Ollama endpoint.

        Returns:
            The content string from the message field.

        Raises:
            AIClientError: If the response structure is unexpected.
        """
        try:
            message = data["message"]
            content = message.get("content", "")
            return str(content) if content is not None else ""
        except (KeyError, TypeError) as exc:
            raise AIClientError(
                f"Unexpected Ollama response structure: {exc}"
            ) from exc

    @staticmethod
    def _parse_ollama_stream_chunk(line: str) -> str | None:
        """Parse a single newline-delimited JSON line from an Ollama streaming response.

        Args:
            line: A raw text line from the streaming response.

        Returns:
            The delta content string if present, ``None`` for empty or
            non-content lines.
        """
        line = line.strip()
        if not line:
            return None
        try:
            data = json.loads(line)
            message = data.get("message", {})
            return message.get("content")  # may be None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Public API – non-streaming
    # ------------------------------------------------------------------

    async def complete(self, prompt: str) -> str:
        """Send a prompt to the AI backend and return the full response.

        This is a non-streaming call that waits for the complete response
        before returning.

        Args:
            prompt: The user's prompt text to send to the AI.

        Returns:
            The AI-generated completion as a string.

        Raises:
            AIClientTimeoutError: If the request times out.
            AIClientConnectionError: If the backend cannot be reached.
            AIClientError: For any other AI backend error.
            RuntimeError: If the client is not initialised.
        """
        client = self._require_client()
        url = self._get_endpoint_url()
        headers = self._get_headers()
        payload = self._build_payload(prompt, stream=False)

        logger.debug(
            "Sending prompt to %s backend at %s (model=%s, len=%d)",
            self._settings.ai_backend_type,
            url,
            self._settings.ai_model,
            len(prompt),
        )

        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise AIClientTimeoutError(
                f"AI backend request timed out after {self._settings.ai_timeout_seconds}s: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise AIClientConnectionError(
                f"Cannot connect to AI backend at {url}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise AIClientConnectionError(
                f"HTTP request error communicating with AI backend: {exc}"
            ) from exc

        if response.status_code != 200:
            raise AIClientError(
                f"AI backend returned HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            data: dict[str, Any] = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise AIClientError(
                f"AI backend returned non-JSON response: {exc}"
            ) from exc

        if self._settings.ai_backend_type == "openai":
            result = self._parse_openai_response(data)
        else:
            result = self._parse_ollama_response(data)

        logger.debug(
            "Received response from AI backend (len=%d)", len(result)
        )
        return result

    # ------------------------------------------------------------------
    # Public API – streaming
    # ------------------------------------------------------------------

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream a prompt response from the AI backend chunk by chunk.

        Yields decoded text chunks as they arrive from the backend. For
        OpenAI-compatible backends, this uses Server-Sent Events (SSE);
        for Ollama, it uses newline-delimited JSON.

        Args:
            prompt: The user's prompt text to send to the AI.

        Yields:
            Non-empty string chunks of the AI-generated response.

        Raises:
            AIClientTimeoutError: If the initial connection times out.
            AIClientConnectionError: If the backend cannot be reached.
            AIClientError: For any other AI backend error.
            RuntimeError: If the client is not initialised.
        """
        client = self._require_client()
        url = self._get_endpoint_url()
        headers = self._get_headers()
        payload = self._build_payload(prompt, stream=True)

        logger.debug(
            "Streaming prompt to %s backend at %s (model=%s)",
            self._settings.ai_backend_type,
            url,
            self._settings.ai_model,
        )

        try:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise AIClientError(
                        f"AI backend returned HTTP {response.status_code}: "
                        f"{body.decode(errors='replace')[:500]}"
                    )

                async for line in response.aiter_lines():
                    if self._settings.ai_backend_type == "openai":
                        chunk = self._parse_openai_stream_chunk(line)
                    else:
                        chunk = self._parse_ollama_stream_chunk(line)

                    if chunk:
                        yield chunk

        except httpx.TimeoutException as exc:
            raise AIClientTimeoutError(
                f"AI backend streaming request timed out: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise AIClientConnectionError(
                f"Cannot connect to AI backend at {url}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise AIClientConnectionError(
                f"HTTP request error during AI backend streaming: {exc}"
            ) from exc

    async def complete_with_fallback(self, prompt: str) -> str:
        """Attempt a streaming completion and accumulate chunks into a full string.

        Falls back to :meth:`complete` if streaming is not available or
        encounters an error during the stream iteration.

        Args:
            prompt: The user's prompt text.

        Returns:
            The full AI-generated response as a string.

        Raises:
            AIClientError: If both streaming and non-streaming attempts fail.
            RuntimeError: If the client is not initialised.
        """
        chunks: list[str] = []
        try:
            async for chunk in self.stream(prompt):
                chunks.append(chunk)
            return "".join(chunks)
        except AIClientError:
            logger.warning(
                "Streaming failed, falling back to non-streaming completion"
            )
            return await self.complete(prompt)
