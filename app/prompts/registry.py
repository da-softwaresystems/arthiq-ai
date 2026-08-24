"""Prompt lookup.

Registering a prompt here is what makes it usable. The service resolves a
version id to a :class:`~app.prompts.base.Prompt` through this module and
nowhere else, so the set of prompts that can ever run is enumerable.
"""

from __future__ import annotations

from app.prompts.base import Prompt
from app.prompts.technical_analysis import TECHNICAL_ANALYSIS_V1

_REGISTERED: tuple[Prompt, ...] = (TECHNICAL_ANALYSIS_V1,)

PROMPTS: dict[str, Prompt] = {prompt.version_id: prompt for prompt in _REGISTERED}

#: Used when a request does not pin a version.
DEFAULT_PROMPT_ID = TECHNICAL_ANALYSIS_V1.version_id


class UnknownPromptError(LookupError):
    """A prompt version was requested that this build does not contain."""

    def __init__(self, version_id: str) -> None:
        self.version_id = version_id
        super().__init__(f"Unknown prompt version: {version_id}")


def get_prompt(version_id: str | None = None) -> Prompt:
    """Resolve a prompt version id, defaulting to :data:`DEFAULT_PROMPT_ID`."""
    key = version_id or DEFAULT_PROMPT_ID
    prompt = PROMPTS.get(key)
    if prompt is None:
        raise UnknownPromptError(key)
    return prompt


def available_prompts() -> list[str]:
    """Every prompt version this build can run, sorted for a stable response."""
    return sorted(PROMPTS)
