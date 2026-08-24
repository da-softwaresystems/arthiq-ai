"""Prompt versioning."""

from __future__ import annotations

import pytest

from app.prompts import DEFAULT_PROMPT_ID, UnknownPromptError, available_prompts, get_prompt
from app.prompts.registry import PROMPTS
from app.prompts.technical_analysis import TECHNICAL_ANALYSIS_V1


class TestRegistry:
    def test_default_prompt_is_technical_analysis_v1(self) -> None:
        assert DEFAULT_PROMPT_ID == "technical_analysis_v1"
        assert get_prompt() is TECHNICAL_ANALYSIS_V1

    def test_prompt_resolves_by_version_id(self) -> None:
        assert get_prompt("technical_analysis_v1") is TECHNICAL_ANALYSIS_V1

    def test_unknown_version_raises(self) -> None:
        with pytest.raises(UnknownPromptError, match="technical_analysis_v9"):
            get_prompt("technical_analysis_v9")

    def test_available_prompts_are_enumerable_and_sorted(self) -> None:
        assert available_prompts() == sorted(PROMPTS)

    def test_every_registered_prompt_is_keyed_by_its_own_version_id(self) -> None:
        assert all(key == prompt.version_id for key, prompt in PROMPTS.items())


class TestPromptShape:
    def test_prompt_carries_name_version_and_purpose(self) -> None:
        prompt = get_prompt()
        assert prompt.name == "technical_analysis"
        assert prompt.version == 1
        assert prompt.purpose
        assert prompt.version_id == "technical_analysis_v1"

    def test_prompts_are_immutable(self) -> None:
        with pytest.raises(AttributeError):
            get_prompt().version = 2  # type: ignore[misc]

    def test_render_substitutes_only_the_context(self) -> None:
        rendered = get_prompt().render("CONTEXT-BLOCK")
        assert "CONTEXT-BLOCK" in rendered.user
        assert rendered.system == TECHNICAL_ANALYSIS_V1.system
        # The JSON example survives rendering, so the braces in the template
        # are escaped correctly rather than eaten by str.format.
        assert '"decision"' in rendered.user

    def test_template_has_no_other_placeholder(self) -> None:
        # Any second placeholder would be an injection surface, since the only
        # value the service substitutes is the built context.
        rendered = get_prompt().render("x")
        assert "{" in rendered.user  # the JSON example
        assert "{context}" not in rendered.user


class TestPromptContent:
    """The prompt has to ask for exactly what the validator will accept."""

    def test_asks_for_the_closed_decision_set(self) -> None:
        user = get_prompt().render("x").user
        for value in ("BUY", "SELL", "HOLD", "LOW", "MEDIUM", "HIGH"):
            assert value in user

    def test_states_that_confidence_is_not_a_probability(self) -> None:
        assert "not a probability" in get_prompt().system.lower()

    def test_requires_invalidating_conditions_for_actionable_calls(self) -> None:
        assert "invalidating_conditions" in get_prompt().system

    def test_forbids_profit_claims_and_execution_language(self) -> None:
        system = get_prompt().system.lower()
        assert "never claim a guaranteed" in system
        assert "nothing you return executes anything" in system
