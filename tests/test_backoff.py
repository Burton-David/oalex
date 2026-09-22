"""``with_backoff`` retry behavior."""

from __future__ import annotations

import httpx
import pytest

from oalex._backoff import _parse_retry_after, with_backoff


async def _no_sleep(seconds: float) -> None:
    return None


async def test_returns_immediately_on_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)
    calls = 0

    async def do_request() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"{}")

    response = await with_backoff(do_request)
    assert response.status_code == 200
    assert calls == 1


async def test_retries_on_503_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)
    statuses = iter([503, 503, 200])

    async def do_request() -> httpx.Response:
        return httpx.Response(next(statuses), content=b"{}")

    response = await with_backoff(do_request)
    assert response.status_code == 200


async def test_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)
    statuses = iter([429, 200])

    async def do_request() -> httpx.Response:
        return httpx.Response(next(statuses), content=b"{}")

    response = await with_backoff(do_request)
    assert response.status_code == 200


async def test_returns_last_response_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)

    async def do_request() -> httpx.Response:
        return httpx.Response(503, content=b"busy")

    response = await with_backoff(do_request, delays=(0.0, 0.0, 0.0))
    # Caller will pass through raise_for_status to surface the error
    assert response.status_code == 503


async def test_does_not_retry_4xx_other_than_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)
    calls = 0

    async def do_request() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, content=b"{}")

    response = await with_backoff(do_request)
    assert response.status_code == 404
    assert calls == 1


async def test_retries_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)
    attempts = 0

    async def do_request() -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("simulated")
        return httpx.Response(200, content=b"{}")

    response = await with_backoff(do_request, delays=(0.0, 0.0, 0.0))
    assert response.status_code == 200
    assert attempts == 3


async def test_propagates_network_error_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)

    async def do_request() -> httpx.Response:
        raise httpx.ConnectError("permanent")

    with pytest.raises(httpx.ConnectError):
        await with_backoff(do_request, delays=(0.0,))


async def test_honors_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("oalex._backoff.asyncio.sleep", record_sleep)
    statuses = iter([
        httpx.Response(429, headers={"retry-after": "2"}, content=b""),
        httpx.Response(200, content=b"{}"),
    ])

    async def do_request() -> httpx.Response:
        return next(statuses)

    await with_backoff(do_request, delays=(0.5, 1.0, 2.0))
    # First retry sleep should honor max(scheduled=0.5, retry-after=2) = 2s
    assert sleeps[0] == 2.0


async def test_long_retry_after_returns_429_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Retry-After past 30s means the daily budget is gone; sleeping won't help."""
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("oalex._backoff.asyncio.sleep", record_sleep)
    calls = 0

    async def do_request() -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"retry-after": "9999"}, content=b"")

    response = await with_backoff(do_request, delays=(0.5, 1.0))
    assert response.status_code == 429
    assert calls == 1
    assert sleeps == []


async def test_zero_remaining_budget_returns_429_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X-RateLimit-Remaining is the credits left today (OpenAlex auth docs)."""
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("oalex._backoff.asyncio.sleep", record_sleep)

    async def do_request() -> httpx.Response:
        return httpx.Response(429, headers={"x-ratelimit-remaining": "0"}, content=b"")

    response = await with_backoff(do_request)
    assert response.status_code == 429
    assert sleeps == []


async def test_burst_429_with_credits_left_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)
    statuses = iter([
        httpx.Response(429, headers={"x-ratelimit-remaining": "812"}, content=b""),
        httpx.Response(200, content=b"{}"),
    ])

    async def do_request() -> httpx.Response:
        return next(statuses)

    response = await with_backoff(do_request)
    assert response.status_code == 200


async def test_retry_after_at_cap_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("oalex._backoff.asyncio.sleep", record_sleep)
    statuses = iter([
        httpx.Response(503, headers={"retry-after": "30"}, content=b""),
        httpx.Response(200, content=b"{}"),
    ])

    async def do_request() -> httpx.Response:
        return next(statuses)

    await with_backoff(do_request, delays=(0.5,))
    assert sleeps == [30.0]


async def test_non_transport_http_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redirect loop fails the same way every time; only transport errors retry."""
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)
    calls = 0

    async def do_request() -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.TooManyRedirects("loop")

    with pytest.raises(httpx.TooManyRedirects):
        await with_backoff(do_request)
    assert calls == 1


def test_parse_retry_after_handles_seconds() -> None:
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after("  3.5  ") == 3.5


def test_parse_retry_after_returns_none_for_invalid() -> None:
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("not-a-number") is None
    assert _parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
