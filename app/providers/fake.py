"""A deterministic, offline provider.

This is the provider the test suite runs against. It opens no socket, needs no
key and no model, and returns the same text for the same prompt on every
machine and every run - which is what makes an assertion about a decision a
real assertion rather than a coin toss.

It is also usable as ``AI_PROVIDER=fake`` for a smoke run of the full request
path with no AI installed at all.

The built-in heuristic is a stub, not a strategy. It reads RSI out of the
rendered context so that BUY, SELL and HOLD are all reachable and the output
looks like something a model might have said. Nothing about it is an opinion on
a market. Tests that care about a specific answer should pass ``responses``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from app.domain.decision import TokenUsage
from app.domain.enums import AnalysisDepth, Decision, RiskLevel
from app.providers.base import AIProvider, CompletionRequest, CompletionResult, ProviderReadiness

DEFAULT_FAKE_MODEL = "fake-deterministic-v1"
DEFAULT_FAKE_DEEP_MODEL = "fake-deterministic-deep-v1"

_RSI_PATTERN = re.compile(r"rsi_14:\s*([0-9]+(?:\.[0-9]+)?)")
_SYMBOL_PATTERN = re.compile(r"symbol:\s*(\S+)")

_RISK_LEVELS = (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)


class FakeAIProvider(AIProvider):
    """An :class:`~app.providers.base.AIProvider` that never leaves the process."""

    name = "fake"

    def __init__(
        self,
        *,
        responses: Sequence[str] | None = None,
        error: Exception | None = None,
        delay_seconds: float = 0.0,
        model: str = DEFAULT_FAKE_MODEL,
        deep_model: str = DEFAULT_FAKE_DEEP_MODEL,
        ready: bool = True,
    ) -> None:
        self._responses = list(responses or [])
        self._error = error
        self._delay_seconds = delay_seconds
        self._model = model
        self._deep_model = deep_model
        self._ready = ready
        self._index = 0
        #: Every request this provider was asked to run, for assertions.
        self.calls: list[CompletionRequest] = []

    def model_for(self, depth: AnalysisDepth) -> str:
        return self._deep_model if depth is AnalysisDepth.DEEP else self._model

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls.append(request)
        started = time.perf_counter()

        # Ordered so an injected delay can be used to exercise the caller's
        # timeout even when an error is also configured.
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        if self._error is not None:
            raise self._error

        text = self._next_response(request)
        return CompletionResult(
            text=text,
            provider=self.name,
            model=self.model_for(request.depth),
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=TokenUsage(
                # A stable stand-in for a real tokeniser: four characters a token.
                prompt_tokens=(len(request.system) + len(request.user)) // 4,
                completion_tokens=len(text) // 4,
                total_tokens=(len(request.system) + len(request.user) + len(text)) // 4,
            ),
        )

    async def check_readiness(self) -> ProviderReadiness:
        return ProviderReadiness(
            provider=self.name,
            model=self._model,
            ready=self._ready,
            detail="deterministic in-process provider" if self._ready else "disabled for testing",
        )

    def _next_response(self, request: CompletionRequest) -> str:
        """Canned responses in order; the last one repeats once they run out."""
        if not self._responses:
            return self._synthesise(request)
        index = min(self._index, len(self._responses) - 1)
        self._index += 1
        return self._responses[index]

    def _synthesise(self, request: CompletionRequest) -> str:
        digest = hashlib.sha256(request.user.encode("utf-8")).digest()
        symbol_match = _SYMBOL_PATTERN.search(request.user)
        symbol = symbol_match.group(1) if symbol_match else "the instrument"

        decision, rsi = self._decision_for(request.user, digest[0])
        # 0.50-0.90, stable for a given prompt.
        confidence = round(0.50 + (digest[1] % 41) / 100, 2)
        risk_level = _RISK_LEVELS[digest[2] % len(_RISK_LEVELS)]

        rsi_text = f"RSI-14 at {rsi}" if rsi is not None else "no RSI-14 value in the snapshot"
        payload = {
            "decision": decision.value,
            "confidence": confidence,
            "risk_level": risk_level.value,
            "reasoning": (
                f"Deterministic stub reading for {symbol}: {rsi_text}. "
                "Generated offline by FakeAIProvider for testing; it reflects no market view."
            ),
            "key_factors": [f"rsi_14={rsi}" if rsi is not None else "rsi_14 unavailable"],
            "invalidating_conditions": [
                "Close crosses the opposite side of EMA50",
                "RSI-14 leaves the 30-70 band",
            ],
        }
        return json.dumps(payload)

    @staticmethod
    def _decision_for(prompt: str, seed: int) -> tuple[Decision, Decimal | None]:
        """RSI extremes pick a side; anything else falls back to the digest."""
        match = _RSI_PATTERN.search(prompt)
        if match is not None:
            try:
                rsi = Decimal(match.group(1))
            except InvalidOperation:
                rsi = None
            if rsi is not None:
                if rsi >= Decimal(70):
                    return Decision.SELL, rsi
                if rsi <= Decimal(30):
                    return Decision.BUY, rsi
                return Decision.HOLD, rsi
        return (Decision.BUY, Decision.SELL, Decision.HOLD)[seed % 3], None
