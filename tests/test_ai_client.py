"""Tests for agent_bridge.ai_client using respx to mock HTTP responses.

Covers:
- Payload builder correctness for Ollama and OpenAI backends
- Response parsing (non-streaming and streaming)
- HTTP error handling (timeouts, connection errors, non-200 responses)
- Context manager lifecycle and injected HTTP client support
"""

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
    """Clear the settings LRU cache before and after each test."""
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
    """Unit tests for internal payload construction methods."""

    def test_ollama_payload_structure(self, ollama_settings: Settings) -> None:
        """Ollama payload should include model, messages, options, and stream flag."""
        client = AIClient(ollama_settings)
        payload = client._build_ollama_payload("hello", stream=False)
        assert payload["model"] == "llama3"
        assert payload["stream"] is False
        assert any(m["role"] == "system" for m in payload["messages"])
        assert any(
            m["role"] == "user" and m["content"] == "hello"
            for m in payload["messages"]
        )
        assert "options" in payload
        assert payload["options"]["temperature"] == 0.2
        assert payload["options"]["num_predict"] == 100

    def test_ollama_payload_stream_true(self, ollama_settings: Settings) -> None:
        """Ollama payload should set stream=True when requested."""
        client = AIClient(ollama_settings)
        payload = client._build_ollama_payload("prompt", stream=True)
        assert payload["stream"] is True

    def test_openai_payload_structure(self, openai_settings: Settings) -> None:
        """OpenAI payload should include model, messages, max_tokens, temperature, stream."""
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

    def test_openai_payload_stream_false(self, openai_settings: Settings) -> None:
        """OpenAI payload should set stream=False when not streaming."""
        client = AIClient(openai_settings)
        payload = client._build_openai_payload("q", stream=False)
        assert payload["stream"] is False

    def test_build_payload_dispatches_to_ollama(self, ollama_settings: Settings) -> None:
        """_build_payload should call the Ollama builder for ollama backend type."""
        client = AIClient(ollama_settings)
        payload = client._build_payload("test", stream=False)
        # Ollama payloads have 'options', OpenAI payloads have 'max_tokens'
        assert "options" in payload
        assert "max_tokens" not in payload

    def test_build_payload_dispatches_to_openai(self, openai_settings: Settings) -> None:
        """_build_payload should call the OpenAI builder for openai backend type."""
        client = AIClient(openai_settings)
        payload = client._build_payload("test", stream=False)
        assert "max_tokens" in payload
        assert "options" not in payload

    def test_ollama_endpoint_url(self, ollama_settings: Settings) -> None:
        """Ollama endpoint should point to /api/chat."""
        client = AIClient(ollama_settings)
        assert client._get_endpoint_url() == "http://localhost:11434/api/chat"

    def test_openai_endpoint_url(self, openai_settings: Settings) -> None:
        """OpenAI endpoint should point to /v1/chat/completions."""
        client = AIClient(openai_settings)
        assert client._get_endpoint_url() == "https://api.openai.com/v1/chat/completions"

    def test_openai_headers_include_auth(self, openai_settings: Settings) -> None:
        """OpenAI headers should include a Bearer Authorization token."""
        client = AIClient(openai_settings)
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test-key"

    def test_ollama_headers_no_auth(self, ollama_settings: Settings) -> None:
        """Ollama headers should not include Authorization when no API key is set."""
        client = AIClient(ollama_settings)
        headers = client._get_headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_system_prompt_included_in_messages(self, ollama_settings: Settings) -> None:
        """The configured system prompt should appear as the first message."""
        client = AIClient(ollama_settings)
        payload = client._build_ollama_payload("user message")
        system_messages = [m for m in payload["messages"] if m["role"] == "system"]
        assert len(system_messages) == 1
        assert ollama_settings.ai_system_prompt in system_messages[0]["content"]


# ---------------------------------------------------------------------------
# Response parser tests – OpenAI
# ---------------------------------------------------------------------------


class TestOpenAIResponseParsing:
    """Unit tests for parsing non-streaming OpenAI responses."""

    def test_parse_valid_response(self) -> None:
        """A well-formed response should return the assistant message content."""
        data = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello, world!"}}
            ]
        }
        result = AIClient._parse_openai_response(data)
        assert result == "Hello, world!"

    def test_parse_empty_choices_raises(self) -> None:
        """An empty choices list should raise AIClientError."""
        with pytest.raises(AIClientError, match="no choices"):
            AIClient._parse_openai_response({"choices": []})

    def test_parse_missing_choices_raises(self) -> None:
        """A response missing the 'choices' key should raise AIClientError."""
        with pytest.raises(AIClientError):
            AIClient._parse_openai_response({"model": "gpt-4o"})

    def test_parse_none_content_returns_empty(self) -> None:
        """None content in the assistant message should return an empty string."""
        data = {"choices": [{"message": {"role": "assistant", "content": None}}]}
        result = AIClient._parse_openai_response(data)
        assert result == ""

    def test_parse_multiple_choices_uses_first(self) -> None:
        """When multiple choices exist, only the first should be returned."""
        data = {
            "choices": [
                {"message": {"role": "assistant", "content": "first"}},
                {"message": {"role": "assistant", "content": "second"}},
            ]
        }
        result = AIClient._parse_openai_response(data)
        assert result == "first"

    def test_parse_malformed_choice_raises(self) -> None:
        """A choice without the 'message' key should raise AIClientError."""
        with pytest.raises(AIClientError):
            AIClient._parse_openai_response({"choices": [{"index": 0}]})


# ---------------------------------------------------------------------------
# Response parser tests – Ollama
# ---------------------------------------------------------------------------


class TestOllamaResponseParsing:
    """Unit tests for parsing non-streaming Ollama responses."""

    def test_parse_valid_response(self) -> None:
        """A well-formed Ollama response should return the content string."""
        data = {"message": {"role": "assistant", "content": "Here is the code"}}
        result = AIClient._parse_ollama_response(data)
        assert result == "Here is the code"

    def test_parse_missing_message_raises(self) -> None:
        """A response without the 'message' key should raise AIClientError."""
        with pytest.raises(AIClientError):
            AIClient._parse_ollama_response({"done": True})

    def test_parse_empty_content_returns_empty(self) -> None:
        """An empty content string should be returned as-is."""
        data = {"message": {"role": "assistant", "content": ""}}
        result = AIClient._parse_ollama_response(data)
        assert result == ""

    def test_parse_none_content_returns_empty(self) -> None:
        """None content should be coerced to an empty string."""
        data = {"message": {"role": "assistant", "content": None}}
        result = AIClient._parse_ollama_response(data)
        assert result == ""

    def test_parse_content_with_code_block(self) -> None:
        """Multi-line content with code fences should be returned intact."""
        content = "Here:\n```python\nprint('hi')\n```"
        data = {"message": {"role": "assistant", "content": content}}
        result = AIClient._parse_ollama_response(data)
        assert result == content


# ---------------------------------------------------------------------------
# Stream chunk parser tests
# ---------------------------------------------------------------------------


class TestStreamChunkParsing:
    """Unit tests for parsing individual streaming response lines."""

    # ---- OpenAI SSE ----

    def test_openai_parse_data_line(self) -> None:
        """A valid SSE data line should yield the delta content."""
        line = 'data: {"choices": [{"delta": {"content": "hello"}}]}'
        result = AIClient._parse_openai_stream_chunk(line)
        assert result == "hello"

    def test_openai_parse_done_sentinel(self) -> None:
        """The [DONE] sentinel should return None."""
        result = AIClient._parse_openai_stream_chunk("data: [DONE]")
        assert result is None

    def test_openai_parse_empty_line(self) -> None:
        """An empty line should return None."""
        assert AIClient._parse_openai_stream_chunk("") is None

    def test_openai_parse_whitespace_only_line(self) -> None:
        """A whitespace-only line should return None."""
        assert AIClient._parse_openai_stream_chunk("   ") is None

    def test_openai_parse_non_data_line(self) -> None:
        """A non-data SSE line (e.g. event:) should return None."""
        assert AIClient._parse_openai_stream_chunk("event: ping") is None

    def test_openai_parse_no_content_delta(self) -> None:
        """A delta without 'content' (e.g. role-only) should return None."""
        line = 'data: {"choices": [{"delta": {"role": "assistant"}}]}'
        result = AIClient._parse_openai_stream_chunk(line)
        assert result is None

    def test_openai_parse_empty_choices(self) -> None:
        """A data line with empty choices list should return None."""
        line = 'data: {"choices": []}'
        result = AIClient._parse_openai_stream_chunk(line)
        assert result is None

    def test_openai_parse_invalid_json_returns_none(self) -> None:
        """Malformed JSON in a data line should return None rather than raising."""
        result = AIClient._parse_openai_stream_chunk("data: not-valid-json{")
        assert result is None

    # ---- Ollama NDJSON ----

    def test_ollama_parse_content_line(self) -> None:
        """A valid NDJSON line with content should return that content."""
        line = json.dumps({"message": {"role": "assistant", "content": "world"}})
        result = AIClient._parse_ollama_stream_chunk(line)
        assert result == "world"

    def test_ollama_parse_empty_line(self) -> None:
        """An empty line should return None."""
        assert AIClient._parse_ollama_stream_chunk("") is None

    def test_ollama_parse_done_line_empty_content(self) -> None:
        """A done=True line with empty content should return None (falsy empty string)."""
        line = json.dumps({"done": True, "message": {"content": ""}})
        result = AIClient._parse_ollama_stream_chunk(line)
        # Empty string is falsy – caller filters it; parser returns it as-is
        assert result == "" or result is None

    def test_ollama_parse_invalid_json_returns_none(self) -> None:
        """Malformed JSON should return None rather than raising."""
        result = AIClient._parse_ollama_stream_chunk("not-json")
        assert result is None

    def test_ollama_parse_missing_message_returns_none(self) -> None:
        """A line without a 'message' key should return None."""
        line = json.dumps({"done": False})
        result = AIClient._parse_ollama_stream_chunk(line)
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests – non-streaming complete()
# ---------------------------------------------------------------------------


class TestAIClientComplete:
    """Integration tests for AIClient.complete() using respx mock HTTP."""

    @respx.mock
    async def test_ollama_complete_success(self, ollama_settings: Settings) -> None:
        """A successful Ollama response should return the assistant content."""
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
    async def test_openai_complete_success(self, openai_settings: Settings) -> None:
        """A successful OpenAI response should return the assistant content."""
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
    async def test_http_500_raises_ai_client_error(self, ollama_settings: Settings) -> None:
        """An HTTP 500 response should raise AIClientError."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(AIClientError, match="HTTP 500"):
            async with AIClient(ollama_settings) as client:
                await client.complete("prompt")

    @respx.mock
    async def test_http_401_raises_ai_client_error(self, openai_settings: Settings) -> None:
        """An HTTP 401 response from OpenAI should raise AIClientError."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        with pytest.raises(AIClientError, match="HTTP 401"):
            async with AIClient(openai_settings) as client:
                await client.complete("prompt")

    @respx.mock
    async def test_connection_error_raises_ai_client_connection_error(
        self, ollama_settings: Settings
    ) -> None:
        """A connection refused error should raise AIClientConnectionError."""
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with pytest.raises(AIClientConnectionError):
            async with AIClient(ollama_settings) as client:
                await client.complete("prompt")

    @respx.mock
    async def test_read_timeout_raises_ai_client_timeout_error(
        self, ollama_settings: Settings
    ) -> None:
        """A read timeout should raise AIClientTimeoutError."""
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )

        with pytest.raises(AIClientTimeoutError):
            async with AIClient(ollama_settings) as client:
                await client.complete("prompt")

    @respx.mock
    async def test_connect_timeout_raises_ai_client_timeout_error(
        self, ollama_settings: Settings
    ) -> None:
        """A connect timeout should raise AIClientTimeoutError."""
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectTimeout("connect timed out")
        )

        with pytest.raises(AIClientTimeoutError):
            async with AIClient(ollama_settings) as client:
                await client.complete("prompt")

    @respx.mock
    async def test_invalid_json_raises_ai_client_error(
        self, ollama_settings: Settings
    ) -> None:
        """A non-JSON response body should raise AIClientError."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text="not json at all")
        )

        with pytest.raises(AIClientError):
            async with AIClient(ollama_settings) as client:
                await client.complete("prompt")

    async def test_requires_context_manager(self, ollama_settings: Settings) -> None:
        """Calling complete() outside of context manager should raise RuntimeError."""
        client = AIClient(ollama_settings)
        with pytest.raises(RuntimeError, match="not initialised"):
            await client.complete("prompt")

    @respx.mock
    async def test_correct_model_sent_in_payload(
        self, ollama_settings: Settings
    ) -> None:
        """The request body must include the configured model name."""
        captured_request: list[httpx.Request] = []

        async def capture(request: httpx.Request) -> httpx.Response:
            captured_request.append(request)
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "ok"}, "done": True},
            )

        respx.post("http://localhost:11434/api/chat").mock(side_effect=capture)

        async with AIClient(ollama_settings) as client:
            await client.complete("test prompt")

        assert len(captured_request) == 1
        body = json.loads(captured_request[0].content)
        assert body["model"] == "llama3"

    @respx.mock
    async def test_user_prompt_included_in_messages(
        self, ollama_settings: Settings
    ) -> None:
        """The user's prompt must appear in the messages array."""
        captured_request: list[httpx.Request] = []

        async def capture(request: httpx.Request) -> httpx.Response:
            captured_request.append(request)
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "ok"}, "done": True},
            )

        respx.post("http://localhost:11434/api/chat").mock(side_effect=capture)

        async with AIClient(ollama_settings) as client:
            await client.complete("Write unit tests for me")

        body = json.loads(captured_request[0].content)
        user_messages = [m for m in body["messages"] if m["role"] == "user"]
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == "Write unit tests for me"

    @respx.mock
    async def test_auth_header_sent_for_openai(
        self, openai_settings: Settings
    ) -> None:
        """OpenAI requests must include the Authorization: Bearer header."""
        captured_request: list[httpx.Request] = []

        async def capture(request: httpx.Request) -> httpx.Response:
            captured_request.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            )

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=capture
        )

        async with AIClient(openai_settings) as client:
            await client.complete("prompt")

        assert "Authorization" in captured_request[0].headers
        assert captured_request[0].headers["Authorization"] == "Bearer sk-test-key"


# ---------------------------------------------------------------------------
# Integration tests – streaming stream()
# ---------------------------------------------------------------------------


class TestAIClientStream:
    """Integration tests for AIClient.stream() using respx mock HTTP."""

    @respx.mock
    async def test_ollama_stream_yields_chunks(
        self, ollama_settings: Settings
    ) -> None:
        """Streaming Ollama response should yield each content chunk in order."""
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
    async def test_openai_stream_yields_chunks(
        self, openai_settings: Settings
    ) -> None:
        """Streaming OpenAI response should yield content delta chunks."""
        lines = [
            'data: {"choices": [{"delta": {"content": "def "}}]}',
            'data: {"choices": [{"delta": {"content": "sort"}}]}',
            'data: {"choices": [{"delta": {}}]}',  # role-only delta – skipped
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
    async def test_stream_http_error_raises(
        self, ollama_settings: Settings
    ) -> None:
        """A non-200 streaming response should raise AIClientError."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(503, text="Service Unavailable")
        )

        with pytest.raises(AIClientError, match="HTTP 503"):
            async with AIClient(ollama_settings) as client:
                async for _ in client.stream("prompt"):
                    pass

    @respx.mock
    async def test_stream_connection_error_raises(
        self, ollama_settings: Settings
    ) -> None:
        """A connection error during streaming should raise AIClientConnectionError."""
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("refused")
        )

        with pytest.raises(AIClientConnectionError):
            async with AIClient(ollama_settings) as client:
                async for _ in client.stream("prompt"):
                    pass

    @respx.mock
    async def test_stream_empty_response_yields_nothing(
        self, ollama_settings: Settings
    ) -> None:
        """An empty body should yield no chunks."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text="")
        )

        collected: list[str] = []
        async with AIClient(ollama_settings) as client:
            async for chunk in client.stream("prompt"):
                collected.append(chunk)

        assert collected == []

    @respx.mock
    async def test_stream_skips_empty_chunks(
        self, ollama_settings: Settings
    ) -> None:
        """Empty content chunks should not be yielded."""
        chunks = [
            json.dumps({"message": {"content": "A"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": False}),  # empty – skipped
            json.dumps({"message": {"content": "B"}, "done": True}),
        ]
        stream_body = "\n".join(chunks)

        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text=stream_body)
        )

        collected: list[str] = []
        async with AIClient(ollama_settings) as client:
            async for chunk in client.stream("prompt"):
                collected.append(chunk)

        # Empty string chunks should be filtered by the generator (falsy check)
        assert "A" in collected
        assert "B" in collected
        assert "" not in collected


# ---------------------------------------------------------------------------
# complete_with_fallback()
# ---------------------------------------------------------------------------


class TestCompleteWithFallback:
    """Tests for the streaming-with-fallback helper."""

    @respx.mock
    async def test_fallback_accumulates_stream_chunks(
        self, ollama_settings: Settings
    ) -> None:
        """complete_with_fallback should join all stream chunks into one string."""
        chunks = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": True}),
        ]
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text="\n".join(chunks))
        )

        async with AIClient(ollama_settings) as client:
            result = await client.complete_with_fallback("prompt")

        assert result == "Hello world"

    @respx.mock
    async def test_fallback_uses_complete_on_stream_error(
        self, ollama_settings: Settings
    ) -> None:
        """If streaming fails, complete_with_fallback should call complete() instead."""
        # First call (stream) returns 503, second call (complete) returns valid JSON
        call_count = 0

        async def flaky_server(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Streaming call – return 503 to trigger fallback
                return httpx.Response(503, text="unavailable")
            # Non-streaming fallback call
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "fallback result"},
                    "done": True,
                },
            )

        respx.post("http://localhost:11434/api/chat").mock(side_effect=flaky_server)

        async with AIClient(ollama_settings) as client:
            result = await client.complete_with_fallback("prompt")

        assert result == "fallback result"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Injected HTTP client
# ---------------------------------------------------------------------------


class TestAIClientInjectedHttpClient:
    """Tests verifying that an externally provided httpx.AsyncClient is used."""

    @respx.mock
    async def test_injected_client_used(
        self, ollama_settings: Settings
    ) -> None:
        """An externally provided HTTP client should be used for requests."""
        response_body = {
            "message": {"role": "assistant", "content": "Injected client worked!"},
            "done": True,
        }
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        external_client = httpx.AsyncClient()
        try:
            ai = AIClient(ollama_settings, http_client=external_client)
            # With injected client, lifecycle is managed externally
            await ai.__aenter__()
            result = await ai.complete("test prompt")
            await ai.__aexit__(None, None, None)
        finally:
            await external_client.aclose()

        assert result == "Injected client worked!"

    @respx.mock
    async def test_injected_client_not_closed_by_ai_client(
        self, ollama_settings: Settings
    ) -> None:
        """AIClient must not close an injected HTTP client on context manager exit."""
        response_body = {
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        }
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        external_client = httpx.AsyncClient()
        try:
            ai = AIClient(ollama_settings, http_client=external_client)
            async with ai:
                await ai.complete("test")

            # Client should still be usable after AIClient exits
            assert not external_client.is_closed
        finally:
            await external_client.aclose()

    @respx.mock
    async def test_owned_client_closed_on_exit(
        self, ollama_settings: Settings
    ) -> None:
        """When AIClient owns its HTTP client, it must close it on exit."""
        response_body = {
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        }
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        ai = AIClient(ollama_settings)
        async with ai:
            await ai.complete("test")
            owned_client = ai._http_client

        # After exit, the owned client should be None (closed and unset)
        assert ai._http_client is None
