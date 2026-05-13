"""HTTP retry with exponential backoff.

OpenAlex returns 429 (rate-limited) and 5xx (server busy) under load.
Most of these failures are transient; the upstream's Retry-After header
tells us when the burst has cleared. ``with_backoff`` runs the request,
retries on retryable statuses (429 + 5xx) and network errors with a
1s/2s/4s schedule (three retries, total wait capped at ~7s), and honors
Retry-After when the server suggests a longer wait.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

_log = logging.getLogger(__name__)

_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# 1s, 2s, 4s — three retries plus the initial attempt gives four tries
# and a worst-case ~7s wait. Higher caps don't help in practice: when an
# upstream's 429 cooldown is on the order of minutes, retrying longer
# just delays the user without succeeding.
_DEFAULT_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)

# Retry-After can suggest very long waits; cap so a misbehaving upstream
# can't pin a single request for minutes.
_MAX_RETRY_AFTER_SECONDS = 30.0


async def with_backoff(
    do_request: Callable[[], Awaitable[httpx.Response]],
    *,
    delays: tuple[float, ...] = _DEFAULT_DELAYS,
) -> httpx.Response:
    """Run ``do_request()`` with exponential backoff on retryable failures.

    Returns the final :class:`httpx.Response` (which the caller should
    pass through :meth:`raise_for_status` for a clean error if non-2xx).
    On a network-level failure (``httpx.HTTPError`` raised before a
    response arrives) all attempts are exhausted before the exception
    propagates.

    ``delays`` is the sequence of inter-attempt sleeps; total attempts =
    ``len(delays) + 1``.
    """
    last_network_error: httpx.HTTPError | None = None
    last_response: httpx.Response | None = None
    for attempt in range(len(delays) + 1):
        if attempt > 0:
            sleep_for = delays[attempt - 1]
            if last_response is not None:
                hint = _parse_retry_after(
                    last_response.headers.get("retry-after")
                )
                if hint is not None:
                    sleep_for = max(sleep_for, min(hint, _MAX_RETRY_AFTER_SECONDS))
            _log.info(
                "oalex: backing off %.1fs before retry %d/%d",
                sleep_for, attempt, len(delays),
            )
            await asyncio.sleep(sleep_for)
        try:
            response = await do_request()
        except httpx.HTTPError as exc:
            last_network_error = exc
            last_response = None
            if attempt == len(delays):
                raise
            _log.warning(
                "oalex: network error on attempt %d/%d (%s); retrying",
                attempt + 1, len(delays) + 1, exc,
            )
            continue
        if response.status_code not in _RETRYABLE_STATUS:
            return response
        last_response = response
        if attempt == len(delays):
            return response
        _log.warning(
            "oalex: HTTP %d on attempt %d/%d; retrying",
            response.status_code, attempt + 1, len(delays) + 1,
        )
    if last_response is not None:
        return last_response
    assert last_network_error is not None
    raise last_network_error


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
