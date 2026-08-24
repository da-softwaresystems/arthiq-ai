"""Ollama adapter - the default provider for local development.

Talks to the Ollama HTTP API directly with ``httpx``: no vendor SDK, no
telemetry, no internet. A developer needs a running ``ollama serve`` and a
pulled model, and nothing else - no cloud account and no API key.

The model is read from configuration. There is deliberately no default model
name anywhere in this file: a model baked into source is a model that silently
differs between machines.

``OLLAMA_THINK=false`` disables the reasoning chain on models that have one
(qwen3, deepseek-r1). Worth setting: the chain is discarded by this service, but
it is charged against the output budget, so a thinking model can spend its whole
``num_predict`` allowance before it starts the JSON.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.core.config import Settings
from app.domain.decision import TokenUsage
from app.domain.enums import AnalysisDepth
from app.providers.base import AIProvider, CompletionRequest, CompletionResult, ProviderReadiness
from app.providers.exceptions import (
    ProviderModelUnavailableError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

CHAT_PATH = "/api/chat"
TAGS_PATH = "/api/tags"


class OllamaProvider(AIProvider):
    """A local Ollama server behind the neutral provider interface."""

    name = "ollama"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        if not settings.ollama_model:
            raise ProviderNotConfiguredError(
                "OLLAMA_MODEL is not set; configure a pulled model id", provider=self.name
            )
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        self._deep_model = settings.ollama_deep_model or settings.ollama_model
        self._think = settings.ollama_think
        # Injectable so tests can drive the adapter through a mock transport
        # instead of a server.
        self._client = client
        self._owns_client = client is None

    def model_for(self, depth: AnalysisDepth) -> str:
        return self._deep_model if depth is AnalysisDepth.DEEP else self._model

    def _http(self, timeout: float) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        return self._client

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        model = self.model_for(request.depth)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "stream": False,
            # Ollama's JSON mode. It constrains the shape, not the content, so
            # the output is still validated downstream.
            "format": "json",
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        if self._think is not None:
            # Only sent when configured: a model with no thinking support
            # rejects the field outright.
            payload["think"] = self._think

        started = time.perf_counter()
        client = self._http(request.timeout_seconds)
        try:
            response = await client.post(CHAT_PATH, json=payload, timeout=request.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama did not respond within {request.timeout_seconds:g}s",
                provider=self.name,
                model=model,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Could not reach Ollama at {self._base_url}: {type(exc).__name__}",
                provider=self.name,
                model=model,
            ) from exc

        self._raise_for_status(response, model)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._to_result(response, model, latency_ms)

    def _raise_for_status(self, response: httpx.Response, model: str) -> None:
        if response.is_success:
            return
        status = response.status_code
        if status == httpx.codes.NOT_FOUND:
            raise ProviderModelUnavailableError(
                f"Ollama has no model {model!r}; pull it first",
                provider=self.name,
                model=model,
            )
        if status == httpx.codes.TOO_MANY_REQUESTS:
            raise ProviderRateLimitError(
                "Ollama rejected the call as rate limited", provider=self.name, model=model
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise ProviderUnavailableError(
                f"Ollama returned HTTP {status}", provider=self.name, model=model
            )
        # Body text is deliberately not echoed: it is vendor output, and this
        # message may be logged.
        raise ProviderResponseError(
            f"Ollama rejected the request with HTTP {status}", provider=self.name, model=model
        )

    def _to_result(self, response: httpx.Response, model: str, latency_ms: int) -> CompletionResult:
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                "Ollama returned a body that is not JSON", provider=self.name, model=model
            ) from exc

        if not isinstance(body, dict):
            raise ProviderResponseError(
                "Ollama returned an unexpected payload shape", provider=self.name, model=model
            )
        if body.get("done_reason") == "length":
            raise ProviderResponseError(
                "Ollama stopped at the output limit; the response is truncated",
                provider=self.name,
                model=model,
            )

        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError(
                "Ollama returned no message content", provider=self.name, model=model
            )

        prompt_tokens = body.get("prompt_eval_count")
        completion_tokens = body.get("eval_count")
        usage = TokenUsage(
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=(
                prompt_tokens + completion_tokens
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
                else None
            ),
        )
        # The reported model wins over the configured one: it is what actually ran.
        served_model = body.get("model")
        return CompletionResult(
            text=content.strip(),
            provider=self.name,
            model=served_model if isinstance(served_model, str) and served_model else model,
            latency_ms=latency_ms,
            usage=usage,
        )

    async def check_readiness(self) -> ProviderReadiness:
        """Ask Ollama which models it has. This runs no inference."""
        model = self._model
        try:
            response = await self._http(5.0).get(TAGS_PATH, timeout=5.0)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return ProviderReadiness(
                provider=self.name,
                model=model,
                ready=False,
                detail=f"Ollama not reachable at {self._base_url} ({type(exc).__name__})",
            )

        models = body.get("models", []) if isinstance(body, dict) else []
        names = {entry.get("name") for entry in models if isinstance(entry, dict)}
        # A bare name in configuration is served by Ollama as ``name:latest``.
        available = model in names or f"{model}:latest" in names
        return ProviderReadiness(
            provider=self.name,
            model=model,
            ready=available,
            detail="model available" if available else f"model {model!r} is not pulled",
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        if self._owns_client:
            self._client = None
