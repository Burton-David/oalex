"""Errors raised by the oalex client."""

from __future__ import annotations


class OalexError(Exception):
    """Base for all oalex-defined exceptions."""


class OalexUnavailable(OalexError):
    """Raised when a transient failure prevents resolving a request.

    Network errors, HTTP 5xx after exhausted retries, and malformed JSON
    responses surface here. Distinct from "not found": ``fetch_work``
    returns ``None`` for legitimate 404s rather than raising, so callers
    can tell "this id doesn't exist in OpenAlex" apart from "OpenAlex is
    having a bad day."
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class OalexRateLimited(OalexUnavailable):
    """OpenAlex answered 429 and retrying soon won't help.

    Since February 2026 OpenAlex meters usage against a daily credit
    budget that resets at midnight UTC. Keyless callers get a small one,
    so a busy script can run it dry mid-day. ``retry_after`` is the
    server's Retry-After hint in seconds when it sent one.
    """

    def __init__(self, reason: str, *, retry_after: float | None = None) -> None:
        super().__init__(reason)
        self.retry_after = retry_after


class OalexRequestError(OalexError):
    """OpenAlex rejected the request itself (a 4xx other than 404 and 429).

    A bad filter, a malformed id, or an invalid API key lands here.
    Retrying the same request will fail the same way.
    """

    def __init__(self, reason: str, *, status_code: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
