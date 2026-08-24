"""API schemas for the internal analysis endpoints.

The request body *is* :class:`~app.domain.analysis.AnalysisRequest` and the
response body *is* :class:`~app.domain.decision.TradingDecision`. There is no
separate wire model, because there is nothing to hide: the domain models are
already provider-neutral, already strict, and already the contract.

Only endpoint-shaped responses that have no domain equivalent live here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.domain.analysis import AnalysisRequest
from app.domain.decision import TradingDecision

#: Aliases that read naturally in the router signature.
AnalyzeRequest = AnalysisRequest
AnalyzeResponse = TradingDecision


class ProviderReadinessResponse(BaseModel):
    """Whether the configured provider could serve a decision right now.

    Answering this never runs an inference, so it costs nothing to poll.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None
    ready: bool
    detail: str
    default_prompt_version: str
    prompt_versions: list[str] = Field(description="Prompt versions this build can run")


__all__ = [
    "AnalyzeRequest",
    "AnalyzeResponse",
    "ProviderReadinessResponse",
]
