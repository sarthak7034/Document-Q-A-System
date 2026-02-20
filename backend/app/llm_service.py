"""
LLM Service for interacting with the Ollama HTTP API.

This module provides the LLMService class that communicates with a locally
running Ollama instance to perform text generation. It supports both
streaming and non-streaming response modes, model discovery, and health
checking — with descriptive error messages for all failure scenarios.
"""

import json
import logging
from typing import AsyncIterator, List, Union

import httpx

logger = logging.getLogger(__name__)


class LLMConnectionError(Exception):
    """Raised when the Ollama service cannot be reached."""


class LLMModelNotFoundError(Exception):
    """Raised when the requested model is not available in Ollama."""


class LLMGenerationError(Exception):
    """Raised when text generation fails for any reason."""


class LLMService:
    """
    Service for generating text using the Ollama HTTP API.

    Supports both streaming and non-streaming generation, model listing,
    and health checks.  All public methods are async so they can be used
    comfortably inside FastAPI route handlers.

    Args:
        base_url:   Base URL of the running Ollama instance.
                    Defaults to ``http://localhost:11434``.
        model_name: Name of the Ollama model to use for generation.
                    Defaults to ``llama2``.
        timeout:    HTTP request timeout in seconds for non-streaming calls.
                    Defaults to 120 seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama2",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate a response from the LLM.

        Args:
            prompt:      The full prompt text to send to the model.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens:  Maximum number of tokens to generate.
            stream:      If *True*, returns an async generator that yields text
                         chunks as they arrive.  If *False*, waits for the full
                         response and returns it as a single string.

        Returns:
            A complete answer string when *stream* is ``False``, or an
            ``AsyncIterator[str]`` when *stream* is ``True``.

        Raises:
            LLMConnectionError:    When Ollama cannot be reached.
            LLMModelNotFoundError: When the requested model is not available.
            LLMGenerationError:    For any other generation failure.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if stream:
            return self._stream_generate(payload)
        else:
            return await self._blocking_generate(payload)

    async def check_health(self) -> bool:
        """
        Check whether the Ollama service is reachable.

        Returns:
            ``True`` if the Ollama API responds, ``False`` otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    async def list_models(self) -> List[str]:
        """
        Retrieve the list of models currently available in Ollama.

        Returns:
            A list of model name strings (e.g. ``["llama2", "mistral"]``).

        Raises:
            LLMConnectionError: When Ollama cannot be reached.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                models = data.get("models", [])
                return [m["name"] for m in models]
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running (e.g. `docker-compose up ollama`)."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMGenerationError(
                f"Ollama returned HTTP {exc.response.status_code} when listing models."
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _blocking_generate(self, payload: dict) -> str:
        """Send a non-streaming request to Ollama and return the full response."""
        url = f"{self.base_url}/api/generate"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                self._raise_for_ollama_errors(response)
                data = response.json()
                return data.get("response", "")
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running (e.g. `docker-compose up ollama`)."
            ) from exc
        except (LLMConnectionError, LLMModelNotFoundError, LLMGenerationError):
            raise
        except Exception as exc:
            raise LLMGenerationError(
                f"Unexpected error during text generation: {exc}"
            ) from exc

    async def _stream_generate(self, payload: dict) -> AsyncIterator[str]:
        """
        Send a streaming request to Ollama and yield text chunks incrementally.

        Ollama's streaming API sends one JSON object per line, each with a
        ``response`` field containing the next token(s) and a ``done`` field
        set to ``True`` on the final message.
        """
        url = f"{self.base_url}/api/generate"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    self._raise_for_ollama_errors(response)
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk_data = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Could not parse streaming chunk: %s", line)
                            continue

                        token = chunk_data.get("response", "")
                        if token:
                            yield token

                        if chunk_data.get("done", False):
                            break
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running (e.g. `docker-compose up ollama`)."
            ) from exc
        except (LLMConnectionError, LLMModelNotFoundError, LLMGenerationError):
            raise
        except Exception as exc:
            raise LLMGenerationError(
                f"Unexpected error during streaming generation: {exc}"
            ) from exc

    @staticmethod
    def _raise_for_ollama_errors(response: httpx.Response) -> None:
        """
        Inspect an Ollama HTTP response and raise a typed exception for errors.

        Ollama returns HTTP 404 when the model is not found and HTTP 5xx for
        server-side failures.  Other 4xx codes signal bad requests.
        """
        if response.status_code == 200:
            return

        if response.status_code == 404:
            # Try to get a meaningful body if possible; some httpx response
            # objects (e.g. streaming) may not have .text available yet.
            try:
                body = response.text
            except Exception:
                body = ""
            raise LLMModelNotFoundError(
                f"Model not found in Ollama (HTTP 404). "
                f"Pull the model first with `ollama pull <model>`. "
                f"Response: {body}"
            )

        try:
            body = response.text
        except Exception:
            body = ""

        raise LLMGenerationError(
            f"Ollama returned HTTP {response.status_code}: {body}"
        )
