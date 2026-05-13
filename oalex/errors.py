"""Errors raised by the oalex client."""

from __future__ import annotations


class OalexError(Exception):
    """Base for all oalex-defined exceptions."""


class OalexUnavailable(OalexError):
    """Raised when a transient failure prevents resolving a request.

    Network errors, HTTP 429 / 5xx after exhausted retries, and malformed
    JSON responses surface here. Distinct from "not found" — the client's
    ``fetch_work`` returns ``None`` for legitimate 404s rather than
    raising, so callers can distinguish "this id doesn't exist in OpenAlex"
    from "OpenAlex is having a bad day."
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
