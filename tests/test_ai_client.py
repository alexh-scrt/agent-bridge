"""Tests for agent_bridge.ai_client using respx to mock HTTP responses."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx
import pytest
import respx

from agent_bridge.ai_client import (
    AIClient,
    AIClientConnectionError,
    AIClientError,
    AIClientTimeoutError,
)
from agent_bridge.config import Settings, get_settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def ollama_settings() -> Settings:
    """Settings configured for an Ollama backend."""
    with patch.dict(
        os.environ,
        {
            "AGENT_BRIDGE_TESTING": "1",
            "AI_BACKEND_TYPE": "ollama",
            "AI_BACKEND_URL": "http://localhost:11434",
            "AI_MODEL": "llama3",
            "AI_MAX_TOKENS": "100",
            "AI_TEMPERATURE": "0.2",
            "AI_TIMEOUT_SECONDS": "30",
        },
    ):
        yield Settings()


@pytest.fixture
def openai_settings() -> Settings:
    """Settings configured for an OpenAI-compatible backend."""
    with patch.dict(
        os.environ,
        {
            "AGENT_BRIDGE_TESTING": "1",
            "AI_BACKEND_TYPE": "openai",
            "AI_BACKEND_URL": "https://api.openai.com",
            "AI_MODEL": "gpt-4o",
            "AI_MAX_TOKENS": "100",
            "AI_TEMPERATURE": "0.5",
            "AI_TIMEOUT_SECONDS": "30",
            "AI_API_KEY": "sk-test-key",
        },
    ):
        yield Settings()


# ---------------------------------------------------------------------------
# Payload builder tests
# ---------------------------------------------------------------------------


class TestPayloadBuilders:
    def test_ollama_payload_structure(self, ollama_settings: Settings):
        client = AIClient(ollama_settings)
        payload = client._build_ollama_payload("hello", stream=False)
        assert payload["model"] == "llama3"
        assert payload["stream"] is False
        assert any(m["role"] == "system" for m in payload["messages"])
        assert any(m["role"] == "user" and m["content"] == "hello" for m in payload["messages"])
        assert "options" in payload
        assert payload["options"]["temperature"] == 0.2

    def test_openai_payload_structure(self, openai_settings: Settings):
        client = AIClient(openai_settings)
        payload = client._build_openai_payload("hello", stream=True)
        assert payload["model"] == "gpt-4o"
        assert payload["stream"] is True
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0.5
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hello"

    def test_ollama_endpoint_url(self, ollama_settings: Settings):
        client = AIClient(ollama_settings)
        assert client._get_endpoint_url() == "http://localhost:11434/api/chat"

    def test_openai_endpoint_url(self, openai_settings: Settings):
        client = AIClient(openai_settings)
        assert client._get_endpoint_url() == "https://api.openai.com/v1/chat/completions"

    def test_openai_headers_include_auth(self, openai_settings: Settings):
        client = AIClient(openai_settings)
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test-key"

    def test_ollama_headers_no_auth(self, ollama_settings: Settings):
        client = AIClient(ollama_settings)
        headers = client._get_headers()
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Response parser tests
# ---------------------------------------------------------------------------


class TestOpenAIResponseParsing:
    def test_parse_valid_response(self):
        data = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello, world!"}}
            ]
        }
        result = AIClient._parse_openai_response(data)
        assert result == "Hello, world!"

    def test_parse_empty_choices_raises(self):
        with pytest.raises(AIClientError, match="no choices"):
            AIClient._parse_openai_response({"choices": []})

    def test_parse_missing_choices_raises(self):
        with pytest.raises(AIClientError):
            AIClient._parse_openai_response({"model": "gpt-4o"})

    def test_parse_none_content_returns_empty(self):
        data = {"choices": [{"message": {"role": "assistant", "content": None}}]}
        result = AIClient._parse_openai_response(data)
        assert result == ""


class TestOllamaResponseParsing:
    def test_parse_valid_response(self):
        data = {"message": {"role": "assistant", "content": "Here is the code"}}
        result = AIClient._parse_ollama_response(data)
        assert result == "Here is the code"

    def test_parse_missing_message_raises(self):
        with pytest.raises(AIClientError):
            AIClient._parse_ollama_response({"done": True})

    def test_parse_empty_content_returns_empty(self):
        data = {"message": {"role": "assistant", "content": ""}}
        result = AIClient._parse_ollama_response(data)
        assert result == ""

    def test_parse_none_content_returns_empty(self):
        data = {"message": {"role": "assistant", "content": None}}
        result = AIClient._parse_ollama_response(data)
        assert result == ""


class TestStreamChunkParsing:
    def test_openai_parse_data_line(self):
        line = 'data: {"choices": [{"delta": {"content": "hello"}}]}'
        result = AIClient._parse_openai_stream_chunk(line)
        assert result == "hello"

    def test_openai_parse_done_sentinel(self):
        result = AIClient._parse_openai_stream_chunk("data: [DONE]")
        assert result is None

    def test_openai_parse_empty_line(self):
        assert AIClient._parse_openai_stream_chunk("") is None
        assert AIClient._parse_openai_stream_chunk("   ") is None

    def test_openai_parse_non_data_line(self):
        assert AIClient._parse_openai_stream_chunk("event: ping") is None

    def test_openai_parse_no_content_delta(self):
        line = 'data: {"choices": [{"delta": {"role": "assistant"}}]}'
        result = AIClient._parse_openai_stream_chunk(line)
        assert result is None

    def test_ollama_parse_content_line(self):
        line = json.dumps({"message": {"role": "assistant", "content": "world"}})
        result = AIClient._parse_ollama_stream_chunk(line)
        assert result == "world"

    def test_ollama_parse_empty_line(self):
        assert AIClient._parse_ollama_stream_chunk("") is None

    def test_ollama_parse_done_line(self):
        line = json.dumps({"done": True, "message": {"content": ""}})
        result = AIClient._parse_ollama_stream_chunk(line)
        assert result is None  # empty string is falsy, filtered by caller


# ---------------------------------------------------------------------------
# Integration tests using respx
# ---------------------------------------------------------------------------


class TestAIClientComplete:
    @respx.mock
    async def test_ollama_complete_success(self, ollama_settings: Settings):
        response_body = {
            "message": {"role": "assistant", "content": "Here is a sort function."},
            "done": True,
        }
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        async with AIClient(ollama_settings) as client:
            result = await client.complete("Write a sort function")

        assert result == "Here is a sort function."

    @respx.mock
    async def test_openai_complete_success(self, openai_settings: Settings):
        response_body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "def sort(lst): return sorted(lst)",
                    }
                }
            ]
        }
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        async with AIClient(openai_settings) as client:
            result = await client.complete("Write a sort function")

        assert "sort" in result

    @respx.mock
    async def test_http_error_raises_ai_client_error(self, ollama_settings: Settings):
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(AIClientError, match="HTTP 500"):
            async with AIClient(ollama_settings) as client:
                await client.complete("prompt")

    @respx.mock
    async def test_connection_error_raises(self, ollama_settings: Settings):
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with pytest.raises(AIClientConnectionError):
            async with AIClient(ollama_settings) as client:
                await client.complete("prompt")

    @respx.mock
    async def test_timeout_raises(self, ollama_settings: Settings):
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )

        with pytest.raises(AIClientTimeoutError):
            async with AIClient(ollama_settings) as client:
                await client.complete("prompt")

    @respx.mock
    async def test_invalid_json_raises(self, ollama_settings: Settings):
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text="not json")
        )

        with pytest.raises(AIClientError):
            async with AIClient(ollama_settings) as client:
                await client.complete("prompt")

    async def test_requires_context_manager(self, ollama_settings: Settings):
        client = AIClient(ollama_settings)
        with pytest.raises(RuntimeError, match="not initialised"):
            await client.complete("prompt")


class TestAIClientStream:
    @respx.mock
    async def test_ollama_stream_yields_chunks(self, ollama_settings: Settings):
        chunks = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": False}),
            json.dumps({"message": {"content": "!"}, "done": True}),
        ]
        stream_body = "\n".join(chunks)

        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text=stream_body)
        )

        collected: list[str] = []
        async with AIClient(ollama_settings) as client:
            async for chunk in client.stream("prompt"):
                collected.append(chunk)

        assert collected == ["Hello", " world", "!"]

    @respx.mock
    async def test_openai_stream_yields_chunks(self, openai_settings: Settings):
        lines = [
            'data: {"choices": [{"delta": {"content": "def "}}]}',
            'data: {"choices": [{"delta": {"content": "sort"}}]}',
            'data: {"choices": [{"delta": {}}]}',  # role-only delta
            "data: [DONE]",
        ]
        stream_body = "\n".join(lines)

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=stream_body)
        )

        collected: list[str] = []
        async with AIClient(openai_settings) as client:
            async for chunk in client.stream("prompt"):
                collected.append(chunk)

        assert collected == ["def ", "sort"]

    @respx.mock
    async def test_stream_http_error_raises(self, ollama_settings: Settings):
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(503, text="Service Unavailable")
        )

        with pytest.raises(AIClientError, match="HTTP 503"):
            async with AIClient(ollama_settings) as client:
                async for _ in client.stream("prompt"):
                    pass

    @respx.mock
    async def test_stream_connection_error_raises(self, ollama_settings: Settings):
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("refused")
        )

        with pytest.raises(AIClientConnectionError):
            async with AIClient(ollama_settings) as client:
                async for _ in client.stream("prompt"):
                    pass


class TestAIClientInjectedHttpClient:
    """Test using an externally provided httpx.AsyncClient."""

    @respx.mock
    async def test_injected_client_used(self, ollama_settings: Settings):
        response_body = {
            "message": {"role": "assistant", "content": "Injected client worked!"},
            "done": True,
        }
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        # Provide an external client – AIClient must not close it
        external_client = httpx.AsyncClient()
        try:
            ai = AIClient(ollama_settings, http_client=external_client)
            # No context manager needed when client is injected
            await ai.__aenter__()
            result = await ai.complete("test prompt")
            await ai.__aexit__(None, None, None)
        finally:
            await external_client.aclose()

        assert result == "Injected client worked!"
