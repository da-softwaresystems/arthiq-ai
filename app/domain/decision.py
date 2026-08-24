"""What this service returns: a validated, provider-neutral trading decision.

The split in this file is the security boundary of the whole service.

:class:`DecisionDraft` is *what the model claimed*. It carries only judgement -
a decision, a confidence, a risk level, reasoning, and the conditions that would
invalidate it. Every field is bounded and every enum is closed, so a model
cannot answer with a fourth decision or a 3000-word essay.

:class:`DecisionMetadata` is *what the service knows*. Provider, model, prompt
version, timestamp and correlation id are stamped server-side from the objects
that actually did the work. A model has no way to influence its own provenance:
if it emits ``"provider": "gemini"`` inside its JSON, that key is rejected as an
unknown field on the draft and never reaches the metadata.

On confidence
-------------
``confidence`` is the model's self-reported conviction on a 0-1 scale. It is
**not** a calibrated probability. Nothing measures it against outcomes, and
0.80 does not mean eight of ten such calls will be profitable. Treat it as an
ordering hint between decisions from the same prompt version and model, and
nothing more.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import AnalysisDepth, Decision, RiskLevel

MAX_REASONING_CHARS = 2000
MAX_CONDITION_CHARS = 200
MAX_CONDITIONS = 6
MAX_FACTORS = 6

_STRICT = ConfigDict(extra="forbid", frozen=True)


class TokenUsage(BaseModel):
    """Token counts, when the provider reports them. All optional."""

    model_config = _STRICT

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class DecisionDraft(BaseModel):
    """The model's answer, after parsing and before it is trusted.

    Constructing one of these *is* the output validation: anything the model
    returns that does not fit becomes a
    :class:`~app.providers.exceptions.ProviderResponseError`, never a decision.
    """

    model_config = _STRICT

    decision: Decision
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported conviction, 0-1. Not a calibrated probability.",
    )
    risk_level: RiskLevel
    reasoning: str = Field(min_length=1, max_length=MAX_REASONING_CHARS)
    key_factors: list[str] = Field(default_factory=list, max_length=MAX_FACTORS)
    invalidating_conditions: list[str] = Field(default_factory=list, max_length=MAX_CONDITIONS)

    @model_validator(mode="after")
    def _check_conditions(self) -> Self:
        """An actionable call must say what would prove it wrong.

        This is a control, not a formality: a BUY or SELL with no invalidating
        condition gives the backend nothing to monitor, so it is rejected.
        """
        for condition in self.invalidating_conditions:
            if not condition.strip():
                raise ValueError("invalidating_conditions must not contain blank entries")
            if len(condition) > MAX_CONDITION_CHARS:
                raise ValueError(
                    f"each invalidating condition must be at most {MAX_CONDITION_CHARS} characters"
                )
        if self.decision is not Decision.HOLD and not self.invalidating_conditions:
            raise ValueError(
                f"a {self.decision} decision requires at least one invalidating condition"
            )
        return self


class DecisionMetadata(BaseModel):
    """How the decision was produced. Filled in by the service, never by a model."""

    model_config = _STRICT

    provider: str
    model: str
    prompt_name: str
    prompt_version: str = Field(description="Versioned prompt id, e.g. technical_analysis_v1")
    depth: AnalysisDepth
    generated_at: datetime = Field(description="UTC instant the decision was produced")
    request_id: str | None = Field(
        default=None, description="Correlation id shared with the caller"
    )
    latency_ms: int | None = Field(default=None, ge=0, description="Provider call duration")
    usage: TokenUsage | None = None


class TradingDecision(BaseModel):
    """A validated analytical opinion, safe to hand back to the backend.

    Analytical only. It authorises nothing: this service cannot place an order,
    and the backend is free to ignore any decision it receives.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    reasoning: str
    key_factors: list[str] = Field(default_factory=list)
    invalidating_conditions: list[str] = Field(default_factory=list)
    metadata: DecisionMetadata

    @classmethod
    def from_draft(cls, draft: DecisionDraft, metadata: DecisionMetadata) -> TradingDecision:
        """Combine a validated draft with service-owned provenance."""
        return cls(
            decision=draft.decision,
            confidence=draft.confidence,
            risk_level=draft.risk_level,
            reasoning=draft.reasoning,
            key_factors=list(draft.key_factors),
            invalidating_conditions=list(draft.invalidating_conditions),
            metadata=metadata,
        )
