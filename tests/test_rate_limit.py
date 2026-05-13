"""Rate limiter behavior."""

from __future__ import annotations

import asyncio
import time

import pytest

from oalex._rate_limit import RateLimiter


def test_rate_limiter_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(0.0)
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(-1.0)


async def test_acquire_no_wait_on_first_call() -> None:
    rl = RateLimiter(0.1)
    start = time.monotonic()
    await rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05


async def test_acquire_enforces_min_interval() -> None:
    """Second call within interval window should sleep until the window passes."""
    rl = RateLimiter(0.1)
    await rl.acquire()
    start = time.monotonic()
    await rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.08


async def test_acquire_no_wait_after_interval_elapses() -> None:
    rl = RateLimiter(0.05)
    await rl.acquire()
    await asyncio.sleep(0.06)
    start = time.monotonic()
    await rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.02


async def test_concurrent_acquires_serialize() -> None:
    """Two concurrent acquires should run back-to-back, not in parallel."""
    rl = RateLimiter(0.05)
    start = time.monotonic()
    await asyncio.gather(rl.acquire(), rl.acquire(), rl.acquire())
    elapsed = time.monotonic() - start
    # 3 calls, with the 2nd and 3rd each waiting ~0.05s
    assert elapsed >= 0.08
