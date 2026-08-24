"""Gemini adapter - the cloud provider.

Calls the Generative Language REST API with ``httpx`` rather than the vendor
SDK. Three reasons: one fewer dependency to keep current, no vendor object can
accidentally escape this module, and the whole adapter is testable through
``httpx.MockTransport`` without a key or a network.

Cost control lives here as configuration, not as logic:

* ``GEMINI_MODEL`` is the routine model (a Flash-Lite class model), and
  ``GEMINI_DEEP_MODEL`` the stronger one used only when the caller asks for
  ``depth=DEEP``. Nothing in the domain layer names either.
* both must be pinned ids; the settings validator rejects floating aliases.
* ``maxOutputTokens`` is bounded, and ``responseMimeType`` is JSON so tokens are
  not spent on prose around the answer.

The API key travels in the ``x-goog-api-key`` header, never in the URL, so it
cannot end up in a proxy log or an echoed request line.
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
    ProviderAuthError,
    ProviderModelUnavailableError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

API_KEY_HEADER = "x-goog-api-key"


class GeminiProvider(AIProvider):
    """Google Gemini behind the neutral provider interface."""

    name = "gemini"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        if not settings.gemini_api_key_value:
            raise ProviderNotConfiguredError("GEMINI_API_KEY is not set", provider=self.name)
        if not settings.gemini_model:
            raise ProviderNotConfiguredError(
                "GEMINI_MODEL is not set; configure an explicit model id", provider=self.name
            )
        self._base_url = settings.gemini_base_url
        self._api_key = settings.gemini_api_key_value
        self._model = settings.gemini_model
        self._deep_model = settings.gemini_deep_model or settings.gemini_model
        self._client = client
        self._owns_client = client is None

    def model_for(self, depth: AnalysisDepth) -> str:
        return self._deep_model if depth is AnalysisDepth.DEEP else self._model

    def _http(self, timeout: float) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout,
                headers={API_KEY_HEADER: self._api_key},
            )
        return self._client

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        model = self.model_for(request.depth)
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": [{"text": request.user}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }

        started = time.perf_counter()
        client = self._http(request.timeout_seconds)
        try:
            response = await client.post(
                f"/models/{model}:generateContent",
                json=payload,
                timeout=request.timeout_seconds,
                # Set again here so an injected client without default headers
                # still authenticates.
                headers={API_KEY_HEADER: self._api_key},
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Gemini did not respond within {request.timeout_seconds:g}s",
                provider=self.name,
                model=model,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Could not reach Gemini: {type(exc).__name__}", provider=self.name, model=model
            ) from exc

        self._raise_for_status(response, model)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._to_result(response, model, latency_ms)

    def _raise_for_status(self, response: httpx.Response, model: str) -> None:
        if response.is_success:
            return
        status = response.status_code
        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            # Never echo the body: a rejected-key response can quote the request.
            raise ProviderAuthError(
                "Gemini rejected the configured API key", provider=self.name, model=model
            )
        if status == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("retry-after")
            raise ProviderRateLimitError(
                "Gemini rate limit reached",
                provider=self.name,
                model=model,
                retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            )
        if status == httpx.codes.NOT_FOUND:
            raise ProviderModelUnavailableError(
                f"Gemini has no model {model!r} available to this key",
                provider=self.name,
                model=model,
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise ProviderUnavailableError(
                f"Gemini returned HTTP {status}", provider=self.name, model=model
            )
        raise ProviderResponseError(
            f"Gemini rejected the request with HTTP {status}", provider=self.name, model=model
        )

    def _to_result(self, response: httpx.Response, model: str, latency_ms: int) -> CompletionResult:
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                "Gemini returned a body that is not JSON", provider=self.name, model=model
            ) from exc
        if not isinstance(body, dict):
            raise ProviderResponseError(
                "Gemini returned an unexpected payload shape", provider=self.name, model=model
            )

        feedback = body.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise ProviderResponseError(
                f"Gemini blocked the prompt ({feedback['blockReason']})",
                provider=self.name,
                model=model,
            )

        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderResponseError(
                "Gemini returned no candidates", provider=self.name, model=model
            )
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise ProviderResponseError(
                "Gemini returned an unexpected candidate shape", provider=self.name, model=model
            )

        finish_reason = candidate.get("finishReason")
        if finish_reason == "MAX_TOKENS":
            raise ProviderResponseError(
                "Gemini stopped at the output limit; the response is truncated",
                provider=self.name,
                model=model,
            )
        if finish_reason in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT"):
            raise ProviderResponseError(
                f"Gemini stopped early ({finish_reason})", provider=self.name, model=model
            )

        text = self._extract_text(candidate)
        if not text:
            raise ProviderResponseError(
                "Gemini returned no text content", provider=self.name, model=model
            )

        return CompletionResult(
            text=text,
            provider=self.name,
            model=body.get("modelVersion") or model,
            latency_ms=latency_ms,
            usage=self._extract_usage(body),
        )

    @staticmethod
    def _extract_text(candidate: dict[str, Any]) -> str:
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            return ""
        chunks = [
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "".join(chunks).strip()

    @staticmethod
    def _extract_usage(body: dict[str, Any]) -> TokenUsage | None:
        usage = body.get("usageMetadata")
        if not isinstance(usage, dict):
            return None

        def _count(key: str) -> int | None:
            value = usage.get(key)
            return value if isinstance(value, int) else None

        return TokenUsage(
            prompt_tokens=_count("promptTokenCount"),
            completion_tokens=_count("candidatesTokenCount"),
            total_tokens=_count("totalTokenCount"),
        )

    async def check_readiness(self) -> ProviderReadiness:
        """Configuration-only check.

        Deliberately makes no API call: a readiness probe that reached Gemini
        would bill the account on every poll of the health endpoint.
        """
        return ProviderReadiness(
            provider=self.name,
            model=self._model,
            ready=True,
            detail="api key and model configured (not verified against the API)",
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        if self._owns_client:
            self._client = None
