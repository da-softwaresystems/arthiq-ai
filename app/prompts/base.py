"""The prompt type.

Prompts are immutable. A published version is never edited in place: a change
in wording is a new version, because a decision recorded against
``technical_analysis_v1`` must mean the same thing next month as it did today.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A prompt with a context substituted in, ready for a provider."""

    system: str
    user: str


@dataclass(frozen=True, slots=True)
class Prompt:
    """A named, versioned instruction set.

    ``user_template`` contains exactly one placeholder, ``{context}``, filled
    with the rendered output of the context builder. Nothing else is
    interpolated: a template that took arbitrary caller strings would be an
    injection surface.
    """

    name: str
    version: int
    purpose: str
    system: str
    user_template: str

    @property
    def version_id(self) -> str:
        """Stable identifier recorded on every decision, e.g. ``technical_analysis_v1``."""
        return f"{self.name}_v{self.version}"

    def render(self, context: str) -> RenderedPrompt:
        return RenderedPrompt(system=self.system, user=self.user_template.format(context=context))
