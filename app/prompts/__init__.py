"""Versioned prompts.

A prompt is an artefact with a name and a version, not a string literal in the
middle of a service function. Every decision records the version it came from,
so a change in behaviour can always be traced to a change in a prompt.
"""

from app.prompts.base import Prompt, RenderedPrompt
from app.prompts.registry import (
    DEFAULT_PROMPT_ID,
    UnknownPromptError,
    available_prompts,
    get_prompt,
)

__all__ = [
    "DEFAULT_PROMPT_ID",
    "Prompt",
    "RenderedPrompt",
    "UnknownPromptError",
    "available_prompts",
    "get_prompt",
]
