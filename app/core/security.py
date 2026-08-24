"""Service-to-service authentication.

The Arthiq backend authenticates the *end user* with Firebase and never passes
that identity here. This service authenticates the *calling service* with a
shared secret presented as ``X-API-Key``.

Two consequences, both deliberate:

* there is no Firebase code in this repository, and no notion of a user; and
* nothing a client sends can grant it authority. A body field claiming to be a
  user id is data for the prompt, never a permission.

The check fails closed: with ``AI_SERVICE_API_KEY`` unset, every internal call
is rejected rather than waved through.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, ConfigurationError

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

#: ``auto_error=False``: an absent header must produce this service's error
#: envelope, not FastAPI's default ``{"detail": ...}`` body.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def verify_service_key(
    presented: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Authenticate the calling service, or raise.

    Comparison is constant-time, and the presented value is never logged or
    echoed - a rejected call learns only that it was rejected.
    """
    accepted = settings.service_api_keys
    if not accepted:
        logger.error("AI_SERVICE_API_KEY is not configured; refusing internal request")
        raise ConfigurationError("Service authentication is not configured")

    if not presented:
        raise AuthenticationError(f"Missing {API_KEY_HEADER} header")

    if not any(secrets.compare_digest(presented, key) for key in accepted):
        logger.warning("Rejected internal request: invalid service key")
        raise AuthenticationError("Invalid service credentials")
