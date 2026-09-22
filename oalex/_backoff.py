"""HTTP retry with exponential backoff.

OpenAlex returns 429 (rate-limited) and 5xx (server busy) under load.
Most of these failures are transient; the upstream's Retry-After header
tells us when the burst has cleared. ``with_backoff`` runs the request,
retries on retryable statuses (429 + 5xx) and transport errors with a
1s/2s/4s schedule (three retries, ~7s of waiting in the worst case), and
honors Retry-After when the server suggests a longer wait.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

_log = logging.getLogger(__name__)

_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# 1s, 2s, 4s: three retries plus the initial attempt gives four tries
# and a worst-case ~7s wait. Higher caps don't help in practice: when an
# upstream's 429 cooldown is on the order of minutes, retrying longer
# just delays the user without succeeding.
_DEFAULT_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)

# Retry-After above this means the wait isn't a burst cooldown. On a 429
# it's almost always the daily credit budget, which resets at midnight UTC,
# so we hand the response back instead of sleeping on it.
_MAX_RETRY_AFTER_SECONDS = 30.0


async def with_backoff(
    do_request: Callable[[], Awaitable[httpx.Response]],
    *,
    delays: tuple[float, ...] = _DEFAULT_DELAYS,
) -> httpx.Response:
    """Run ``do_request()`` with exponential backoff on retryable failures.

    Returns the final :class:`httpx.Response`, which may still be a 429 or
    5xx once retries run out; the caller decides how to surface it. A
    transport error (connection refused, timeout) is retried the same way
    and re-raised after the last attempt.

    ``delays`` is the sequence of inter-attempt sleeps; total attempts =
    ``len(delays) + 1``.
    """
    last_attempt = len(delays)
    for attempt in range(last_attempt + 1):
        try:
            response = await do_request()
        except httpx.TransportError as exc:
            if attempt == last_attempt:
                raise
            _log.warning(
                "oalex: network error on attempt %d/%d (%s); retrying",
                attempt + 1, last_attempt + 1, exc,
            )
            sleep_for = delays[attempt]
        else:
            if response.status_code not in _RETRYABLE_STATUS or attempt == last_attempt:
                return response
            if _budget_exhausted(response):
                return response
            sleep_for = delays[attempt]
            hint = _parse_retry_after(response.headers.get("retry-after"))
            if hint is not None:
                if hint > _MAX_RETRY_AFTER_SECONDS:
                    return response
                sleep_for = max(sleep_for, hint)
            _log.warning(
                "oalex: HTTP %d on attempt %d/%d; retrying",
                response.status_code, attempt + 1, last_attempt + 1,
            )
        _log.info("oalex: backing off %.1fs", sleep_for)
        await asyncio.sleep(sleep_for)
    raise AssertionError("unreachable: the final attempt returns or raises")


def _budget_exhausted(response: httpx.Response) -> bool:
    # A 429 also fires for bursts over 100 req/s while credits remain; only
    # a zero balance means waiting seconds is pointless.
    return (
        response.status_code == 429
        and response.headers.get("x-ratelimit-remaining", "").strip() == "0"
    )


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header. Spec allows seconds-as-int OR HTTP-date.

    We honor seconds. HTTP-date is rare and computing the delta is
    error-prone (timezone, clock skew); falling back to the scheduled
    delay is preferable to parsing it wrong.
    """
    if value is None:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None
