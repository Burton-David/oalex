"""Filesystem-backed key/value cache for HTTP responses.

Repeated CLI / REPL invocations on the same record waste network time
and, since OpenAlex started metering list and search calls, daily
credits too. A small disk cache keyed by the request URL sidesteps
that. Entries older than ``ttl_seconds`` are treated as misses.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path

_log = logging.getLogger(__name__)


class DiskCache:
    """Filesystem-backed key/value cache. Values are bytes; keys are arbitrary strings."""

    def __init__(self, directory: str | os.PathLike[str], ttl_seconds: int) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds

    def get(self, key: str) -> bytes | None:
        path = self._path_for(key)
        # Another process can prune the directory between stat and read;
        # either way that's a miss, not an error.
        try:
            if (time.time() - path.stat().st_mtime) > self._ttl:
                return None
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def set(self, key: str, value: bytes) -> None:
        path = self._path_for(key)
        # A per-writer temp file: two processes caching the same key must not
        # interleave writes into one shared .tmp before the rename.
        try:
            fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(value)
                os.replace(tmp, path)
            except BaseException:
                os.unlink(tmp)
                raise
        except OSError as exc:
            # The response is already in hand; a full disk or read-only cache
            # dir shouldn't turn a successful fetch into a failure.
            _log.warning("oalex: could not write cache entry %s: %s", path.name, exc)

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.bin"
