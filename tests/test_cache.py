"""DiskCache TTL + roundtrip behavior."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from oalex._cache import DiskCache


def test_set_then_get_roundtrip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60)
    cache.set("hello", b"world")
    assert cache.get("hello") == b"world"


def test_miss_returns_none(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60)
    assert cache.get("never-written") is None


def test_expired_entry_returns_none(tmp_path: Path) -> None:
    """An entry older than ttl_seconds is a miss."""
    cache = DiskCache(tmp_path, ttl_seconds=1)
    cache.set("temp", b"value")

    # Backdate the file's mtime so it appears older than the TTL.
    path = next(tmp_path.glob("*.bin"))
    old = time.time() - 10
    os.utime(path, (old, old))

    assert cache.get("temp") is None


def test_different_keys_dont_collide(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60)
    cache.set("a", b"one")
    cache.set("b", b"two")
    assert cache.get("a") == b"one"
    assert cache.get("b") == b"two"


def test_overwrite_existing_key(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60)
    cache.set("key", b"first")
    cache.set("key", b"second")
    assert cache.get("key") == b"second"


def test_creates_directory_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "deeper"
    assert not nested.exists()
    cache = DiskCache(nested, ttl_seconds=60)
    cache.set("k", b"v")
    assert nested.exists()
    assert cache.get("k") == b"v"


def test_set_leaves_no_temp_files_behind(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60)
    cache.set("k", b"v")
    cache.set("k", b"v2")
    assert [p.suffix for p in tmp_path.iterdir()] == [".bin"]


def test_write_failure_logs_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A full disk shouldn't turn a successful fetch into an exception."""
    cache = DiskCache(tmp_path, ttl_seconds=60)

    def disk_full(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("oalex._cache.tempfile.mkstemp", disk_full)
    with caplog.at_level(logging.WARNING, logger="oalex._cache"):
        cache.set("k", b"v")
    assert cache.get("k") is None
    assert "No space left" in caplog.text
