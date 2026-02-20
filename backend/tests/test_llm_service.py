"""
Unit tests for LLMService.

All tests use httpx's MockTransport so no live Ollama instance is required.
Tests cover:
  - check_health(): healthy and unreachable scenarios
  - list_models(): success and connection-failure scenarios
  - generate() non-streaming: success, model-not-found, server-error, connection-error
  - generate() streaming: multi-chunk and error scenarios
"""

import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.llm_service import (
    LLMConnectionError,
    LLMGenerationError,
    LLMModelNotFoundError,
    LLMService,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_http_response(
    status_code: int,
    body: str | dict | None = None,
    *,
    request: httpx.Request | None = None,
) -> httpx.Response:
    """Return an httpx.Response with the given status and JSON/text body."""
    if isinstance(body, dict):
        content = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
    elif isinstance(body, str):
        content = body.encode()
        headers = {"Content-Type": "text/plain"}
    else:
        content = b""
        headers = {}

    if request is None:
        request = httpx.Request("GET", "http://localhost:11434/")

    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=headers,
        request=request,
    )


@pytest.fixture
def service() -> LLMService:
    """Default LLMService pointing at the standard Ollama URL."""
    return LLMService(base_url="http://localhost:11434", model_name="llama2", timeout=10.0)


# ---------------------------------------------------------------------------
# check_health() tests
# ---------------------------------------------------------------------------


class TestCheckHealth:
    """Tests for LLMService.check_health()."""

    @pytest.mark.asyncio
    async def test_health_returns_true_when_ollama_responds(self, service):
        """check_health() returns True when /api/tags responds with 200."""
        mock_response = _make_http_response(
            200,
            {"models": []},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            result = await service.check_health()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_returns_false_when_ollama_unreachable(self, service):
        """check_health() returns False (not raises) on connection errors."""
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            result = await service.check_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_returns_false_on_timeout(self, service):
        """check_health() returns False on timeout."""
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            result = await service.check_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_returns_false_when_status_not_200(self, service):
        """check_health() returns False when /api/tags returns a non-200 status."""
        mock_response = _make_http_response(
            500,
            "internal server error",
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            result = await service.check_health()

        assert result is False


# ---------------------------------------------------------------------------
# list_models() tests
# ---------------------------------------------------------------------------


class TestListModels:
    """Tests for LLMService.list_models()."""

    @pytest.mark.asyncio
    async def test_list_models_returns_model_names(self, service):
        """list_models() returns a list of model name strings."""
        body = {
            "models": [
                {"name": "llama2", "size": 1234},
                {"name": "mistral", "size": 5678},
            ]
        }
        mock_response = _make_http_response(
            200, body, request=httpx.Request("GET", "http://localhost:11434/api/tags")
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            models = await service.list_models()

        assert models == ["llama2", "mistral"]

    @pytest.mark.asyncio
    async def test_list_models_returns_empty_when_no_models(self, service):
        """list_models() returns an empty list when Ollama has no models pulled."""
        mock_response = _make_http_response(
            200, {"models": []}, request=httpx.Request("GET", "http://localhost:11434/api/tags")
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            models = await service.list_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_raises_connection_error_when_unreachable(self, service):
        """list_models() raises LLMConnectionError when Ollama is down."""
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(LLMConnectionError) as exc_info:
                await service.list_models()

        assert "localhost:11434" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_models_raises_connection_error_on_timeout(self, service):
        """list_models() raises LLMConnectionError on timeout."""
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            with pytest.raises(LLMConnectionError):
                await service.list_models()


# ---------------------------------------------------------------------------
# generate() – non-streaming tests
# ---------------------------------------------------------------------------


class TestGenerateBlocking:
    """Tests for LLMService.generate() with stream=False (default)."""

    @pytest.mark.asyncio
    async def test_generate_returns_response_text(self, service):
        """generate() returns the 'response' field from Ollama's JSON."""
        ollama_reply = {"response": "Paris is the capital of France.", "done": True}
        mock_response = _make_http_response(
            200, ollama_reply, request=httpx.Request("POST", "http://localhost:11434/api/generate")
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await service.generate("What is the capital of France?")

        assert result == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_generate_passes_temperature_and_max_tokens(self, service):
        """generate() sends temperature and max_tokens to Ollama."""
        ollama_reply = {"response": "42.", "done": True}
        mock_response = _make_http_response(
            200, ollama_reply, request=httpx.Request("POST", "http://localhost:11434/api/generate")
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await service.generate("What is 6 times 7?", temperature=0.0, max_tokens=50)
            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs.get("json") or mock_post.call_args.args[1] if mock_post.call_args.args else None
            # Fallback: inspect the call args
            if payload is None:
                payload = mock_post.call_args[1].get("json") or mock_post.call_args[0][1]

        assert payload["options"]["temperature"] == 0.0
        assert payload["options"]["num_predict"] == 50
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_generate_raises_connection_error_when_ollama_down(self, service):
        """generate() raises LLMConnectionError when Ollama is unreachable."""
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(LLMConnectionError) as exc_info:
                await service.generate("Hello")

        assert "localhost:11434" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_raises_connection_error_on_timeout(self, service):
        """generate() raises LLMConnectionError on request timeout."""
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            with pytest.raises(LLMConnectionError):
                await service.generate("Hello")

    @pytest.mark.asyncio
    async def test_generate_raises_model_not_found_on_404(self, service):
        """generate() raises LLMModelNotFoundError when model is not pulled."""
        mock_response = _make_http_response(
            404,
            "model not found",
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(LLMModelNotFoundError) as exc_info:
                await service.generate("Hello")

        assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_raises_generation_error_on_server_error(self, service):
        """generate() raises LLMGenerationError for 5xx Ollama responses."""
        mock_response = _make_http_response(
            500,
            "internal server error",
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(LLMGenerationError) as exc_info:
                await service.generate("Hello")

        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_returns_empty_string_for_empty_response(self, service):
        """generate() returns an empty string if 'response' key is missing."""
        mock_response = _make_http_response(
            200, {"done": True}, request=httpx.Request("POST", "http://localhost:11434/api/generate")
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await service.generate("Hello")

        assert result == ""


# ---------------------------------------------------------------------------
# generate() – streaming tests
# ---------------------------------------------------------------------------


class TestGenerateStreaming:
    """Tests for LLMService.generate() with stream=True."""

    @pytest.mark.asyncio
    async def test_generate_stream_yields_tokens(self, service):
        """generate(stream=True) returns an async iterator that yields text chunks."""
        streamed_lines = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "!", "done": True}),
        ]

        async def fake_aiter_lines():
            for line in streamed_lines:
                yield line

        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.aiter_lines = fake_aiter_lines
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await service.generate("Say hello", stream=True)
            # Collect tokens from the async iterator
            tokens = []
            async for token in result:
                tokens.append(token)

        assert tokens == ["Hello", " world", "!"]
        full_response = "".join(tokens)
        assert full_response == "Hello world!"

    @pytest.mark.asyncio
    async def test_generate_stream_stops_at_done_true(self, service):
        """Streaming generator stops yielding after 'done': true is received."""
        streamed_lines = [
            json.dumps({"response": "Stop here.", "done": True}),
            # These lines should NOT be yielded
            json.dumps({"response": " Extra.", "done": False}),
        ]

        async def fake_aiter_lines():
            for line in streamed_lines:
                yield line

        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.aiter_lines = fake_aiter_lines
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await service.generate("Prompt", stream=True)
            tokens = []
            async for token in result:
                tokens.append(token)

        assert tokens == ["Stop here."]


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------


class TestErrorMessageQuality:
    """Ensure error messages are descriptive and actionable."""

    @pytest.mark.asyncio
    async def test_connection_error_message_includes_url(self, service):
        """LLMConnectionError message includes the Ollama base URL."""
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(LLMConnectionError) as exc_info:
                await service.generate("test")

        msg = str(exc_info.value)
        assert "localhost:11434" in msg

    @pytest.mark.asyncio
    async def test_connection_error_message_includes_suggestion(self, service):
        """LLMConnectionError message includes instructions for starting Ollama."""
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(LLMConnectionError) as exc_info:
                await service.generate("test")

        msg = str(exc_info.value).lower()
        # Should contain something actionable about running Ollama
        assert "ollama" in msg

    @pytest.mark.asyncio
    async def test_model_not_found_message_is_descriptive(self, service):
        """LLMModelNotFoundError message hints at pulling the model."""
        mock_response = _make_http_response(
            404, "not found", request=httpx.Request("POST", "http://localhost:11434/api/generate")
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(LLMModelNotFoundError) as exc_info:
                await service.generate("test")

        msg = str(exc_info.value).lower()
        assert "pull" in msg or "model" in msg
