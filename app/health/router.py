"""Health endpoints.

``GET /health`` answers one question: is this process able to serve requests?
It is unauthenticated so an orchestrator can call it, it touches no provider,
and it costs nothing. A container must not be restarted because Ollama is
being restarted, and a health probe must never trigger a billable inference.

Provider status has its own authenticated endpoint,
``GET /internal/v1/provider/readiness``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from app.core.config import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness only. Deliberately says nothing about any dependency."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )
