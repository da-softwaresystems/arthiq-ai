"""The analysis pipeline.

One request travels a fixed path::

    AnalysisRequest -> context builder -> prompt -> provider -> output
    validation -> TradingDecision

Every stage is replaceable and none of them knows about the others' internals.
The service itself holds the two properties that matter most:

* **the provider is injected.** This class never imports Ollama or Gemini, and
  the test suite runs the whole path against
  :class:`~app.providers.fake.FakeAIProvider`.
* **provenance is stamped here.** Provider, model, prompt version, timestamp
  and correlation id come from the objects that did the work, not from the
  model's output.

Exactly one provider call is made per request. There is no retry loop, no
self-critique pass and no second opinion: a failed call is reported to the
backend, which can decide whether another attempt is worth the money.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

from app.analysis.errors import safe_reason, translate_provider_error
from app.analysis.output import parse_decision_draft
from app.context.builder import ContextLimits, build_context
from app.core.config import Settings
from app.core.deadline import ProviderBudget, resolve_budget
from app.core.exceptions import ValidationError
from app.core.redaction import truncate
from app.domain.analysis import AnalysisRequest
from app.domain.decision import DecisionMetadata, TradingDecision
from app.prompts import Prompt, UnknownPromptError, get_prompt
from app.providers.base import AIProvider, CompletionRequest, CompletionResult
from app.providers.exceptions import AIProviderError, ProviderTimeoutError

logger = logging.getLogger(__name__)


class AnalysisService:
    """Turns a validated request into a validated decision."""

    def __init__(self, provider: AIProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings
        self._limits = ContextLimits.from_settings(settings)

    async def analyze(
        self,
        request: AnalysisRequest,
        *,
        request_id: str | None = None,
        deadline_ms: int | None = None,
    ) -> TradingDecision:
        """Produce one decision, or raise an :class:`AppError` describing why not.

        ``deadline_ms`` is how long the caller says it will still wait. The
        provider budget is clamped to it, so this service never keeps a model
        running for an answer that has already been abandoned.

        Raises :class:`~app.core.exceptions.AppError` only - provider failures
        are translated here so that no caller of this method has to know what an
        Ollama connection error is.
        """
        # Monotonic, and started before any work: the time spent resolving a
        # prompt and building the context is time the caller has spent waiting
        # too, so it comes out of the same budget.
        started = time.monotonic()

        prompt = self._resolve_prompt(request)
        context = build_context(request, self._limits)
        rendered = prompt.render(context.render())

        budget = self._budget(deadline_ms, elapsed_seconds=time.monotonic() - started)
        if budget.expired:
            self._raise_expired(request, deadline_ms)

        completion_request = CompletionRequest(
            system=rendered.system,
            user=rendered.user,
            max_output_tokens=self._settings.ai_max_output_tokens,
            temperature=self._settings.ai_temperature,
            timeout_seconds=budget.seconds,
            depth=request.depth,
        )

        if self._settings.log_prompts:
            logger.debug("Prompt rendered", extra={"prompt": truncate(rendered.user, 2000)})

        try:
            result = await self._complete(completion_request)
            if self._settings.log_provider_responses:
                logger.debug("Provider response", extra={"response": truncate(result.text, 2000)})
            draft = parse_decision_draft(result.text, provider=result.provider, model=result.model)
        except AIProviderError as exc:
            logger.warning(
                "Analysis failed",
                extra={
                    "provider": self._provider.name,
                    "model": self._provider.model_for(request.depth),
                    "symbol": request.symbol,
                    "depth": request.depth.value,
                    "error_code": exc.code,
                    "reason": safe_reason(exc, self._settings),
                    "deadline_ms": deadline_ms,
                    "effective_timeout_seconds": round(budget.seconds, 3),
                    "outcome": "failure",
                },
            )
            raise translate_provider_error(exc) from exc

        decision = TradingDecision.from_draft(
            draft,
            DecisionMetadata(
                provider=result.provider,
                model=result.model,
                prompt_name=prompt.name,
                prompt_version=prompt.version_id,
                depth=request.depth,
                generated_at=datetime.now(UTC),
                request_id=request_id,
                latency_ms=result.latency_ms,
                usage=result.usage,
            ),
        )

        logger.info(
            "Analysis complete",
            extra={
                "provider": result.provider,
                "model": result.model,
                "symbol": request.symbol,
                "interval": request.interval.value,
                "depth": request.depth.value,
                "prompt_version": prompt.version_id,
                "latency_ms": result.latency_ms,
                "deadline_ms": deadline_ms,
                "effective_timeout_seconds": round(budget.seconds, 3),
                "decision": decision.decision.value,
                "confidence": decision.confidence,
                "prompt_tokens": result.usage.prompt_tokens if result.usage else None,
                "completion_tokens": result.usage.completion_tokens if result.usage else None,
                "outcome": "success",
            },
        )
        return decision

    def _resolve_prompt(self, request: AnalysisRequest) -> Prompt:
        try:
            return get_prompt(request.prompt_version)
        except UnknownPromptError as exc:
            # A caller error, not a provider error: no money was spent.
            raise ValidationError(
                f"Unknown prompt version: {exc.version_id}",
                details={"field": "prompt_version"},
            ) from exc

    def _budget(self, deadline_ms: int | None, *, elapsed_seconds: float) -> ProviderBudget:
        """The provider budget for this request, clamped to the caller's deadline."""
        return resolve_budget(
            configured_timeout_seconds=self._settings.ai_request_timeout_seconds,
            deadline_ms=deadline_ms,
            elapsed_seconds=elapsed_seconds,
            safety_margin_seconds=self._settings.ai_deadline_safety_margin_seconds,
        )

    def _raise_expired(self, request: AnalysisRequest, deadline_ms: int | None) -> None:
        """Fail fast on an expired deadline, with the normal timeout semantics.

        No provider call is made: the caller stopped waiting before we started,
        so the cheapest correct answer is an immediate one.
        """
        exc = ProviderTimeoutError(
            "Caller deadline expired before the provider was called",
            provider=self._provider.name,
            model=self._provider.model_for(request.depth),
        )
        logger.warning(
            "Analysis abandoned: deadline expired",
            extra={
                "provider": self._provider.name,
                "symbol": request.symbol,
                "depth": request.depth.value,
                "error_code": exc.code,
                "deadline_ms": deadline_ms,
                "effective_timeout_seconds": 0.0,
                "outcome": "failure",
            },
        )
        raise translate_provider_error(exc) from exc

    async def _complete(self, completion_request: CompletionRequest) -> CompletionResult:
        """Run the provider call under a hard timeout.

        The adapters set their own HTTP timeouts, but this backstop applies to
        every provider - including one whose client hangs somewhere other than a
        socket read.
        """
        try:
            async with asyncio.timeout(completion_request.timeout_seconds):
                return await self._provider.complete(completion_request)
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"Provider exceeded the {completion_request.timeout_seconds:g}s budget",
                provider=self._provider.name,
                model=self._provider.model_for(completion_request.depth),
            ) from exc
