"""FastAPI wiring for the analysis endpoints.

The provider is resolved through the registry and handed to the service, so a
test can swap the whole AI backend with ``set_ai_provider(FakeAIProvider())``
and never touch the router.
"""

from __future__ import annotations

from fastapi import Depends

from app.analysis.service import AnalysisService
from app.core.config import Settings, get_settings
from app.providers.base import AIProvider
from app.providers.registry import get_ai_provider


def provider_dependency() -> AIProvider:
    """The process-wide provider, as a dependency."""
    return get_ai_provider()


def get_analysis_service(
    provider: AIProvider = Depends(provider_dependency),
    settings: Settings = Depends(get_settings),
) -> AnalysisService:
    """Build the service for one request. It holds no state between calls."""
    return AnalysisService(provider, settings)
