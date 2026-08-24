"""Ollama and Gemini adapters, driven through a mock transport.

These exercise the real adapter code - request shape, status handling, response
parsing, error translation - with no server, no key and no network. The live
equivalents live in ``tests/integration`` and are opt-in.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from app.core.config import Settings
from app.domain.enums import AnalysisDepth
from app.providers.base import CompletionRequest
from app.providers.exceptions import (
    ProviderAuthError,
    ProviderModelUnavailableError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.gemini import API_KEY_HEADER, GeminiProvider
from app.providers.ollama import OllamaProvider

DECISION_JSON = (
    '{"decision": "HOLD", "confidence": 0.5, "risk_level": "MEDIUM", '
    '"reasoning": "Mixed signals.", "invalidating_conditions": []}'
)
GEMINI_KEY = "AIzaTestKeyValue1234567890"

Handler = Callable[[httpx.Request], httpx.Response]


def _completion(depth: AnalysisDepth = AnalysisDepth.ROUTINE) -> CompletionRequest:
    return CompletionRequest(
        system="system prompt",
        user="user prompt",
        max_output_tokens=256,
        temperature=0.0,
        timeout_seconds=5.0,
        depth=depth,
    )


def _client(handler: Handler, base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url)


def _ollama(handler: Handler, **overrides: object) -> OllamaProvider:
    settings = Settings(
        _env_file=None,
        ai_provider="ollama",
        ollama_base_url="http://ollama.test:11434",
        ollama_model="test-model",
        **overrides,
    )
    return OllamaProvider(settings, client=_client(handler, settings.ollama_base_url))


def _gemini(handler: Handler, **overrides: object) -> GeminiProvider:
    settings = Settings(
        _env_file=None,
        ai_provider="gemini",
        gemini_base_url="https://gemini.test/v1beta",
        gemini_api_key=GEMINI_KEY,
        gemini_model="gemini-2.5-flash-lite",
        **overrides,
    )
    return GeminiProvider(settings, client=_client(handler, settings.gemini_base_url))


def _status(code: int, json_body: object | None = None) -> Handler:
    """A handler that always answers with one status and body."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json=json_body if json_body is not None else {})

    return handler


class TestOllamaAdapter:
    async def test_successful_completion(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = request.read().decode()
            return httpx.Response(
                200,
                json={
                    "model": "test-model",
                    "message": {"role": "assistant", "content": DECISION_JSON},
                    "done_reason": "stop",
                    "prompt_eval_count": 320,
                    "eval_count": 88,
                },
            )

        result = await _ollama(handler).complete(_completion())

        assert result.text == DECISION_JSON
        assert result.provider == "ollama"
        assert result.model == "test-model"
        assert result.usage is not None
        assert result.usage.total_tokens == 408
        assert captured["url"] == "http://ollama.test:11434/api/chat"
        body = json.loads(str(captured["body"]))
        assert body["stream"] is False
        # Bounded output and a fixed temperature are sent on every call.
        assert body["options"] == {"temperature": 0.0, "num_predict": 256}
        assert body["format"] == "json"

    async def test_thinking_is_not_mentioned_unless_configured(self) -> None:
        # A model with no thinking mode rejects the field, so it is only sent
        # when someone has actually configured it.
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.read().decode())
            return httpx.Response(200, json={"message": {"content": DECISION_JSON}})

        await _ollama(handler).complete(_completion())
        assert "think" not in captured["body"]  # type: ignore[operator]

    async def test_thinking_can_be_disabled(self) -> None:
        # A reasoning model spends the output budget on a chain this service
        # discards, so turning it off is the difference between an answer and
        # a truncated one.
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.read().decode())
            return httpx.Response(200, json={"message": {"content": DECISION_JSON}})

        await _ollama(handler, ollama_think=False).complete(_completion())
        assert captured["body"]["think"] is False  # type: ignore[index]

    async def test_deep_depth_uses_the_deep_model(self) -> None:
        provider = _ollama(_status(200), ollama_deep_model="test-model-deep")
        assert provider.model_for(AnalysisDepth.DEEP) == "test-model-deep"
        assert provider.model_for(AnalysisDepth.ROUTINE) == "test-model"

    async def test_deep_model_falls_back_to_the_routine_model(self) -> None:
        assert _ollama(_status(200)).model_for(AnalysisDepth.DEEP) == "test-model"

    async def test_missing_model_is_reported_as_model_unavailable(self) -> None:
        with pytest.raises(ProviderModelUnavailableError, match="test-model"):
            await _ollama(_status(404, {"error": "model not found"})).complete(_completion())

    async def test_server_error_is_unavailable(self) -> None:
        with pytest.raises(ProviderUnavailableError, match="503"):
            await _ollama(_status(503)).complete(_completion())

    async def test_rate_limit_is_translated(self) -> None:
        with pytest.raises(ProviderRateLimitError):
            await _ollama(_status(429)).complete(_completion())

    async def test_connection_failure_is_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(ProviderUnavailableError, match="ConnectError"):
            await _ollama(handler).complete(_completion())

    async def test_read_timeout_is_a_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        with pytest.raises(ProviderTimeoutError, match="5s"):
            await _ollama(handler).complete(_completion())

    async def test_non_json_body_is_a_response_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json at all")

        with pytest.raises(ProviderResponseError, match="not JSON"):
            await _ollama(handler).complete(_completion())

    async def test_missing_content_is_a_response_error(self) -> None:
        with pytest.raises(ProviderResponseError, match="no message content"):
            await _ollama(_status(200, {"model": "test-model"})).complete(_completion())

    async def test_truncated_response_is_rejected(self) -> None:
        handler = _status(
            200, {"message": {"content": '{"deci'}, "done_reason": "length", "model": "test-model"}
        )
        with pytest.raises(ProviderResponseError, match="truncated"):
            await _ollama(handler).complete(_completion())

    async def test_readiness_reports_a_pulled_model(self) -> None:
        handler = _status(200, {"models": [{"name": "test-model:latest"}]})
        readiness = await _ollama(handler).check_readiness()
        assert readiness.ready is True
        assert readiness.provider == "ollama"

    async def test_readiness_reports_a_missing_model(self) -> None:
        provider = _ollama(_status(200, {"models": [{"name": "other-model"}]}))
        assert (await provider.check_readiness()).ready is False

    async def test_readiness_survives_an_unreachable_server(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        readiness = await _ollama(handler).check_readiness()
        assert readiness.ready is False
        assert "not reachable" in readiness.detail


class TestGeminiAdapter:
    async def test_successful_completion(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["header"] = request.headers.get(API_KEY_HEADER)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": DECISION_JSON}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 300,
                        "candidatesTokenCount": 90,
                        "totalTokenCount": 390,
                    },
                    "modelVersion": "gemini-2.5-flash-lite",
                },
            )

        result = await _gemini(handler).complete(_completion())

        assert result.text == DECISION_JSON
        assert result.provider == "gemini"
        assert result.model == "gemini-2.5-flash-lite"
        assert result.usage is not None
        assert result.usage.total_tokens == 390
        assert "models/gemini-2.5-flash-lite:generateContent" in str(captured["url"])
        # The key travels in a header, never in the URL.
        assert captured["header"] == GEMINI_KEY
        assert GEMINI_KEY not in str(captured["url"])

    async def test_deep_depth_uses_the_configured_deep_model(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": DECISION_JSON}]}}]},
            )

        provider = _gemini(handler, gemini_deep_model="gemini-2.5-flash")
        await provider.complete(_completion(AnalysisDepth.DEEP))
        assert "models/gemini-2.5-flash:generateContent" in seen["url"]

    @pytest.mark.parametrize("code", [401, 403])
    async def test_rejected_key_is_an_auth_error_without_the_key(self, code: int) -> None:
        handler = _status(code, {"error": {"message": f"API key {GEMINI_KEY} not valid"}})
        with pytest.raises(ProviderAuthError) as excinfo:
            await _gemini(handler).complete(_completion())
        assert GEMINI_KEY not in str(excinfo.value)

    async def test_rate_limit_carries_retry_after(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"retry-after": "30"})

        with pytest.raises(ProviderRateLimitError) as excinfo:
            await _gemini(handler).complete(_completion())
        assert excinfo.value.retry_after == 30

    async def test_unknown_model_is_model_unavailable(self) -> None:
        with pytest.raises(ProviderModelUnavailableError):
            await _gemini(_status(404)).complete(_completion())

    async def test_server_error_is_unavailable(self) -> None:
        with pytest.raises(ProviderUnavailableError, match="500"):
            await _gemini(_status(500)).complete(_completion())

    async def test_timeout_is_translated(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        with pytest.raises(ProviderTimeoutError):
            await _gemini(handler).complete(_completion())

    async def test_truncated_candidate_is_rejected(self) -> None:
        handler = _status(
            200,
            {"candidates": [{"content": {"parts": [{"text": "{"}]}, "finishReason": "MAX_TOKENS"}]},
        )
        with pytest.raises(ProviderResponseError, match="truncated"):
            await _gemini(handler).complete(_completion())

    async def test_blocked_prompt_is_rejected(self) -> None:
        with pytest.raises(ProviderResponseError, match="blocked"):
            await _gemini(_status(200, {"promptFeedback": {"blockReason": "SAFETY"}})).complete(
                _completion()
            )

    async def test_no_candidates_is_rejected(self) -> None:
        with pytest.raises(ProviderResponseError, match="no candidates"):
            await _gemini(_status(200, {"candidates": []})).complete(_completion())

    async def test_readiness_makes_no_api_call(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
            raise AssertionError("readiness must not call the Gemini API")

        readiness = await _gemini(handler).check_readiness()
        assert readiness.ready is True
        assert readiness.model == "gemini-2.5-flash-lite"
