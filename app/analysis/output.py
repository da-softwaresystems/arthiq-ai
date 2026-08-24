"""Output validation: raw model text in, validated draft out.

Nothing a provider returns is trusted. A model that answers with prose around
its JSON, a code fence, a fourth decision word, a confidence of 78 instead of
0.78, or a BUY with no invalidating condition produces a
:class:`~app.providers.exceptions.ProviderResponseError` - never a decision.

Two accommodations are made, both narrow and both structural rather than
semantic:

* a fenced or prose-wrapped JSON object is extracted, because models wrap
  output even when told not to; and
* ``decision`` and ``risk_level`` are upper-cased, because case is not meaning.

Everything else is rejected. In particular the value set stays closed: no
mapping of "STRONG BUY" onto BUY, and no rescaling of an out-of-range
confidence into one that happens to validate.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.domain.decision import DecisionDraft
from app.providers.exceptions import ProviderResponseError

#: Cap on the text a model may return before parsing is even attempted. The
#: provider already bounds output tokens; this bounds a misbehaving provider.
MAX_RESPONSE_CHARS = 20_000


def parse_decision_draft(text: str, *, provider: str, model: str) -> DecisionDraft:
    """Validate raw provider text into a :class:`DecisionDraft`."""
    if not text or not text.strip():
        raise ProviderResponseError(
            "Provider returned an empty response", provider=provider, model=model
        )
    if len(text) > MAX_RESPONSE_CHARS:
        raise ProviderResponseError(
            f"Provider response exceeds {MAX_RESPONSE_CHARS} characters",
            provider=provider,
            model=model,
        )

    payload = _load_json_object(text, provider=provider, model=model)
    _normalise_case(payload)

    try:
        return DecisionDraft.model_validate(payload)
    except PydanticValidationError as exc:
        # The field names and reasons are ours, not vendor text; the model's own
        # values are left out so nothing unbounded is carried into a log line.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
            for error in exc.errors()[:5]
        )
        raise ProviderResponseError(
            f"Provider output failed decision validation ({problems})",
            provider=provider,
            model=model,
        ) from exc


def _load_json_object(text: str, *, provider: str, model: str) -> dict[str, Any]:
    candidate = _strip_code_fence(text.strip())
    try:
        payload = json.loads(candidate)
    except ValueError:
        extracted = _first_json_object(candidate)
        if extracted is None:
            # An unterminated object is broken JSON; no brace at all is prose.
            message = (
                "Provider response was not valid JSON"
                if "{" in candidate
                else "Provider response contained no JSON object"
            )
            raise ProviderResponseError(message, provider=provider, model=model) from None
        try:
            payload = json.loads(extracted)
        except ValueError as exc:
            raise ProviderResponseError(
                "Provider response was not valid JSON", provider=provider, model=model
            ) from exc

    if not isinstance(payload, dict):
        raise ProviderResponseError(
            "Provider response was not a JSON object", provider=provider, model=model
        )
    return payload


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding ```json ... ``` fence, if present."""
    if not text.startswith("```"):
        return text
    without_open = text[3:]
    if without_open.lower().startswith("json"):
        without_open = without_open[4:]
    closing = without_open.rfind("```")
    return (without_open[:closing] if closing != -1 else without_open).strip()


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` span, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _normalise_case(payload: dict[str, Any]) -> None:
    """Upper-case the two closed-vocabulary fields, in place."""
    for field in ("decision", "risk_level"):
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = value.strip().upper()
