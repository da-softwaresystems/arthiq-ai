"""The caller's deadline, and the provider budget derived from it.

The backend sends ``X-Deadline-Ms``: **how long it will still wait**, in
milliseconds, measured from the moment it sent the request. It is a duration,
not an absolute timestamp, so no clock agreement between the two services is
required.

This exists because of M5.5: the backend gave up at ~30s while Ollama kept
generating for ~250s. Work nobody is waiting for is pure cost. The rule is that
the layer doing the work gives up first, so this service takes the smaller of
its own configured timeout and what the caller says is left:

    effective = min(configured_timeout, remaining - safety_margin)

``remaining`` is the announced deadline minus the time already spent inside
this service, measured on the monotonic clock - the right clock for a duration,
because it cannot be moved by an NTP step mid-request.

The margin exists so this service's own timeout fires before the caller's does.
Being the first to give up is what makes the failure legible: the backend sees a
reported timeout rather than a dead socket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Remaining milliseconds the caller will still wait for this answer.
DEADLINE_HEADER = "X-Deadline-Ms"

#: A deadline beyond this is not believable and is treated as malformed. It is
#: far above any sane budget, so it only ever catches a bug or a bad actor.
MAX_DEADLINE_MS = 3_600_000

#: A signed integer and nothing else. int() alone is too permissive here: it
#: accepts "25_000", which is not a valid header value and must not be quietly
#: reinterpreted as 25000.
_INTEGER = re.compile(r"[+-]?[0-9]+")


@dataclass(frozen=True, slots=True)
class ProviderBudget:
    """How long the provider call may take, and where that number came from."""

    #: Always > 0 when :attr:`expired` is false; never negative.
    seconds: float
    #: True when there is no time left to run a provider call at all.
    expired: bool
    #: Whether the caller's deadline shortened the configured timeout.
    from_deadline: bool


def parse_deadline_ms(raw: str | None) -> int | None:
    """Read the header value, or ``None`` when it is absent or unusable.

    Deliberately lenient: the deadline is an optimisation, and refusing an
    otherwise valid analysis because an optional header was malformed would
    trade a working request for a broken one. A missing or unparseable value
    simply falls back to the configured timeout - the pre-M5.6 behaviour.

    A value of zero or below is *not* malformed. It is an expired deadline, and
    the caller is told so.
    """
    if raw is None:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    if not _INTEGER.fullmatch(candidate):
        return None
    value = int(candidate)
    if abs(value) > MAX_DEADLINE_MS:
        return None
    return value


def resolve_budget(
    *,
    configured_timeout_seconds: float,
    deadline_ms: int | None,
    elapsed_seconds: float = 0.0,
    safety_margin_seconds: float = 0.0,
) -> ProviderBudget:
    """Clamp the configured timeout to what the caller is still waiting for."""
    if deadline_ms is None:
        return ProviderBudget(
            seconds=configured_timeout_seconds, expired=False, from_deadline=False
        )

    remaining = deadline_ms / 1000.0 - max(elapsed_seconds, 0.0) - safety_margin_seconds
    if remaining <= 0.0:
        # Nothing is waiting for this answer any more. Spending a provider call
        # on it is exactly the waste M5.6 exists to stop.
        return ProviderBudget(seconds=0.0, expired=True, from_deadline=True)

    effective = min(configured_timeout_seconds, remaining)
    return ProviderBudget(
        seconds=effective,
        expired=False,
        from_deadline=effective < configured_timeout_seconds,
    )
