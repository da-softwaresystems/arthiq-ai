"""The provider interface and its neutral request/response types.

An :class:`AIProvider` does one narrow thing: it turns a rendered prompt into
text, and reports what it cost. It does not parse decisions, does not know what
a :class:`~app.domain.decision.TradingDecision` is, and does not decide what a
failure means for an HTTP client. Keeping the interface this thin is what lets
a new vendor be one file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.domain.decision import TokenUsage
from app.domain.enums import AnalysisDepth


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """A rendered prompt plus the limits the call must respect."""

    system: str
    user: str
    max_output_tokens: int
    temperature: float
    timeout_seconds: float
    depth: AnalysisDepth = AnalysisDepth.ROUTINE


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """What a provider returns: text, provenance and cost.

    ``text`` is raw model output. It is not a decision and must not be treated
    as one until it has been through output validation.
    """

    text: str
    provider: str
    model: str
    latency_ms: int
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class ProviderReadiness:
    """Whether the configured provider could serve a call right now.

    Answering this must never run an inference: readiness is free, decisions
    are not.
    """

    provider: str
    model: str | None
    ready: bool
    detail: str


class AIProvider(ABC):
    """What the service requires of any AI vendor."""

    #: Short identifier recorded on every decision.
    name: str = "unknown"

    @abstractmethod
    def model_for(self, depth: AnalysisDepth) -> str:
        """The model id this provider would use for ``depth``.

        Depth-to-model is configuration held by the provider, so no caller and
        no domain rule ever names a model.
        """

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run one completion, or raise an :mod:`app.providers.exceptions` error."""

    async def check_readiness(self) -> ProviderReadiness:
        """Cheap liveness check for the provider. Overridden where useful."""
        return ProviderReadiness(
            provider=self.name,
            model=self.model_for(AnalysisDepth.ROUTINE),
            ready=True,
            detail="configured",
        )

    async def aclose(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release any held resources.

        Not abstract: a provider that holds nothing (the fake one) should not
        have to write an empty override.
        """
