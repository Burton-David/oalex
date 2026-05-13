"""Async rate limiter — at most one call per ``min_interval`` seconds, in order.

OpenAlex's polite pool tolerates ~10 req/sec without complaint, so a
fixed-interval limiter is the right fit (the upstream's documented rate
is what they actually enforce). Process-local; clients in separate
processes maintain their own limiters and can collectively exceed the
documented rate. That's acceptable for a single-process client library;
distributed deployments should layer their own coordination.
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Fixed-interval async rate limiter.

    ``acquire()`` returns immediately if at least ``min_interval`` seconds
    have passed since the last release; otherwise sleeps until that's
    true. Re-entrant from multiple coroutines via an internal lock.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        if min_interval_seconds <= 0:
            raise ValueError("min_interval_seconds must be positive")
        self._interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()
