"""Service settings.

Every value comes from the environment (or a local ``.env``). Secrets are
:class:`~pydantic.SecretStr` and have no defaults - a missing secret fails
loudly instead of quietly falling back to something insecure.

Two rules are enforced here rather than left to convention:

* the AI model is configuration, never a literal in the source tree; and
* a floating alias (``gemini-2.5-flash-latest``) is rejected, because a model
  that changes underneath a prompt version makes a recorded decision
  irreproducible.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["development", "test", "staging", "production"]
ProviderName = Literal["fake", "ollama", "gemini"]

#: Vendor endpoints are defaults, not constants: a proxy, a sandbox or a mock
#: server can be pointed at through the environment.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class Settings(BaseSettings):
    """Runtime configuration, loaded once per process."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Application ------------------------------------------------------
    app_env: Environment = "development"
    app_name: str = "arthiq-ai"
    app_version: str = "0.1.0"
    internal_api_prefix: str = "/internal/v1"
    log_level: str = "INFO"
    enable_docs: bool = True
    # NoDecode: the raw env value is a comma-separated string, not JSON.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # -- Service-to-service authentication --------------------------------
    #: Shared secret the backend presents as ``X-API-Key``. Comma-separated
    #: values are accepted so a key can be rotated without downtime. This is
    #: *service* authentication; end users are authenticated by the backend and
    #: are never known to this service.
    ai_service_api_key: SecretStr | None = None

    # -- Provider selection -----------------------------------------------
    ai_provider: ProviderName = "ollama"
    #: Hard ceiling on a single provider call, in seconds.
    ai_request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    #: Response cap. Bounded output is the cheapest cost control there is.
    ai_max_output_tokens: int = Field(default=768, gt=0, le=8192)
    #: Decisions should be reproducible; 0.0 unless deliberately raised.
    ai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # -- Context bounds ---------------------------------------------------
    #: The context builder truncates to these limits. They exist so a prompt
    #: cannot grow with the caller's history.
    context_max_recent_closes: int = Field(default=20, ge=0, le=200)
    context_max_observations: int = Field(default=10, ge=0, le=50)

    # -- Observability ----------------------------------------------------
    #: Prompts and provider responses are business content; they stay out of
    #: the logs unless someone deliberately turns them on for debugging.
    log_prompts: bool = False
    log_provider_responses: bool = False

    # -- Ollama (default local provider) ----------------------------------
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str | None = None
    #: Optional stronger local model for DEEP analysis; falls back to
    #: ``ollama_model`` when unset.
    ollama_deep_model: str | None = None
    #: Reasoning models (qwen3, deepseek-r1) spend the output budget on a
    #: thinking chain this service discards - it never reaches the decision.
    #: Set ``false`` to turn it off. Left unset, the field is not sent at all,
    #: because a model without thinking support rejects it.
    ollama_think: bool | None = None

    # -- Gemini (cloud provider) ------------------------------------------
    gemini_api_key: SecretStr | None = None
    #: Explicit model id, e.g. ``gemini-2.5-flash-lite`` for routine analysis.
    gemini_model: str | None = None
    #: Explicit model id for DEEP analysis, e.g. ``gemini-2.5-flash``.
    gemini_deep_model: str | None = None
    gemini_base_url: str = DEFAULT_GEMINI_BASE_URL

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string as well as a real list."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("ollama_model", "ollama_deep_model", "gemini_model", "gemini_deep_model")
    @classmethod
    def _reject_floating_alias(cls, value: str | None) -> str | None:
        """Pin the model. An alias that moves breaks decision reproducibility."""
        if value is None:
            return None
        model = value.strip()
        if not model:
            return None
        if model.lower().endswith(("latest", "-preview", ":latest")):
            raise ValueError(
                f"Model {model!r} is a floating alias. Configure an explicit, pinned model id."
            )
        return model

    @field_validator("ollama_base_url", "gemini_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def service_api_keys(self) -> frozenset[str]:
        """Every key currently accepted from the backend."""
        if self.ai_service_api_key is None:
            return frozenset()
        raw = self.ai_service_api_key.get_secret_value()
        return frozenset(key.strip() for key in raw.split(",") if key.strip())

    @property
    def service_auth_configured(self) -> bool:
        return bool(self.service_api_keys)

    @property
    def gemini_api_key_value(self) -> str:
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else ""

    @property
    def json_logs(self) -> bool:
        return self.app_env != "development"

    def secret_values(self) -> tuple[str, ...]:
        """Every secret string this process knows, for log/error redaction."""
        secrets: list[str] = [*self.service_api_keys]
        if self.gemini_api_key:
            secrets.append(self.gemini_api_key_value)
        return tuple(value for value in secrets if value)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor, used as a FastAPI dependency and at import."""
    return Settings()
