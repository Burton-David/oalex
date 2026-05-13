"""DiskCache TTL + roundtrip behavior."""

from __future__ import annotations

import time
from pathlib import Path

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
    import os

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
