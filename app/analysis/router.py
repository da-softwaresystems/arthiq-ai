"""The internal analysis API.

Every route here is mounted under ``/internal/v1`` and every route requires the
service key. "Internal" is not a naming convention: there is no anonymous
route, no user session, no cookie, and no CORS-friendly browser flow. The only
intended caller is the Arthiq backend.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.analysis.dependencies import get_analysis_service, provider_dependency
from app.analysis.schemas import AnalyzeRequest, AnalyzeResponse, ProviderReadinessResponse
from app.analysis.service import AnalysisService
from app.core.middleware import current_request_id
from app.core.security import verify_service_key
from app.prompts import DEFAULT_PROMPT_ID, available_prompts
from app.providers.base import AIProvider

#: Mounted by the app factory under ``settings.internal_api_prefix``
#: (``/internal/v1`` by default). The service key is required by the router
#: itself, so a route added here cannot forget to authenticate.
router = APIRouter(
    tags=["internal"],
    dependencies=[Depends(verify_service_key)],
)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse one instrument snapshot",
    response_description="A validated, provider-neutral trading decision",
)
async def analyze(
    request: AnalyzeRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """Return one analytical opinion for the supplied snapshot.

    The decision is advisory. This service executes nothing; what the backend
    does with the answer - including ignoring it - is entirely the backend's
    decision.
    """
    return await service.analyze(request, request_id=current_request_id())


@router.get(
    "/provider/readiness",
    response_model=ProviderReadinessResponse,
    summary="Check the configured AI provider",
    response_description="Provider status, without running an inference",
)
async def provider_readiness(
    provider: AIProvider = Depends(provider_dependency),
) -> ProviderReadinessResponse:
    """Report whether the provider is usable, and which prompts this build has.

    Separate from ``GET /health`` on purpose: liveness must not depend on an AI
    provider being up, or a restart loop follows the first Ollama hiccup.
    """
    readiness = await provider.check_readiness()
    return ProviderReadinessResponse(
        provider=readiness.provider,
        model=readiness.model,
        ready=readiness.ready,
        detail=readiness.detail,
        default_prompt_version=DEFAULT_PROMPT_ID,
        prompt_versions=available_prompts(),
    )
