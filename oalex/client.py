"""Async client for OpenAlex's ``/works`` endpoint.

The client is reentrant and thread-unsafe (the rate limiter uses
``asyncio.Lock``). Use one instance per asyncio loop. Both
``async with Client(...) as client:`` and manual ``aclose()`` are
supported.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import httpx

from oalex._backoff import with_backoff
from oalex._cache import DiskCache
from oalex._rate_limit import RateLimiter
from oalex.errors import OalexUnavailable
from oalex.types import Work, parse_work

_log = logging.getLogger(__name__)

_API_BASE: Final = "https://api.openalex.org"
_DEFAULT_CACHE_TTL_SECONDS: Final = 24 * 60 * 60
_DEFAULT_TIMEOUT: Final = 30.0
# OpenAlex's polite pool tolerates ~10 req/sec without complaint.
_DEFAULT_MIN_INTERVAL: Final = 0.1
# ``/works`` rejects per-page > 200; cap on our side so a caller passing
# per_page=500 doesn't get a 400 back.
_MAX_PER_PAGE: Final = 200


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "oalex"


class Client:
    """Async client for the OpenAlex Works API.

    OpenAlex is free and unauthenticated; politeness is via a ``mailto``
    query parameter that places requests in the polite pool (faster,
    more reliable than the common pool). ``email`` is required at
    construction — running without one would silently drop into the
    slow pool, which is rarely what callers want.

    Args:
        email: Polite-pool contact address. Required; OpenAlex puts
            requests carrying ``mailto=...`` into a faster, more
            reliable pool. See https://docs.openalex.org/how-to-use-the-api/api-overview#authentication
        cache_dir: Disk cache location. Defaults to
            ``~/.cache/oalex/``. Cached entries older than
            ``ttl_seconds`` count as misses.
        ttl_seconds: Cache TTL in seconds. Defaults to 24 hours.
        min_interval_seconds: Minimum time between requests. Defaults
            to 0.1s (~10 req/sec, the polite-pool tolerance).
        timeout: Per-request HTTP timeout in seconds. Defaults to 30s.
        client: Optional pre-built :class:`httpx.AsyncClient` to reuse.
            When provided, ``aclose()`` does NOT close it — that's the
            caller's responsibility.
    """

    def __init__(
        self,
        *,
        email: str,
        cache_dir: str | os.PathLike[str] | None = None,
        ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
        min_interval_seconds: float = _DEFAULT_MIN_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not email or not email.strip():
            raise ValueError(
                "Client requires a non-empty email — OpenAlex's polite pool "
                "uses ?mailto= for identification, and an empty value silently "
                "degrades to the slower common pool."
            )
        self._email = email.strip()
        cache_path = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self._cache = DiskCache(cache_path, ttl_seconds=ttl_seconds)
        self._rate = RateLimiter(min_interval_seconds)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def search(
        self,
        query: str,
        *,
        per_page: int = 25,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> Sequence[Work]:
        """Search OpenAlex's full-text index.

        Args:
            query: Freeform query string passed to OpenAlex's ``search=`` filter.
            per_page: Maximum number of results to return. Capped at 200
                (OpenAlex's per-page limit); callers needing more should
                paginate themselves.
            year_min: Earliest publication year, inclusive.
            year_max: Latest publication year, inclusive.

        Returns:
            Works in OpenAlex's relevance order. Works that fail to parse
            are silently skipped so a single malformed record doesn't
            poison the batch.

        Raises:
            OalexUnavailable: Transient upstream failure (5xx after
                retries, network error, malformed JSON).
        """
        params: dict[str, str] = {
            "search": query,
            "per-page": str(min(max(per_page, 1), _MAX_PER_PAGE)),
        }
        if year_min is not None or year_max is not None:
            lo = str(year_min) if year_min is not None else ""
            hi = str(year_max) if year_max is not None else ""
            params["filter"] = f"publication_year:{lo}-{hi}"
        body = await self._fetch("/works", params)
        return _parse_search_payload(body)

    async def fetch_work(self, work_id: str) -> Work | None:
        """Fetch a single Work by OpenAlex ID or DOI.

        Args:
            work_id: Either a bare OpenAlex ID (``W2626778328``), an
                ``openalex:`` prefixed ID, or a ``doi:`` prefixed DOI
                (``doi:10.1038/nature12373``). DOIs without the prefix
                are not recognized — use :meth:`fetch_doi` for that.

        Returns:
            A :class:`Work`, or ``None`` if OpenAlex returns 404 (the
            id genuinely doesn't exist in their graph).

        Raises:
            OalexUnavailable: Transient upstream failure.
        """
        prefix, sep, raw = work_id.partition(":")
        if not sep:
            # Bare id like "W2626778328"
            path = f"/works/{work_id}"
        elif prefix == "openalex":
            path = f"/works/{raw}"
        elif prefix == "doi":
            path = f"/works/doi:{raw}"
        else:
            return None
        return await self._fetch_work_at(path)

    async def fetch_doi(self, doi: str) -> Work | None:
        """Fetch a Work by DOI.

        Equivalent to ``fetch_work(f"doi:{doi}")``. Accepts a bare DOI
        (``10.1038/nature12373``) or a DOI URL.
        """
        if doi.startswith(("https://doi.org/", "http://doi.org/")):
            doi = doi.split("doi.org/", 1)[1]
        return await self._fetch_work_at(f"/works/doi:{doi}")

    async def fetch_referenced(
        self,
        work_id: str,
        *,
        limit: int = 10,
    ) -> Sequence[Work]:
        """Resolve a Work's ``referenced_works`` array into full :class:`Work` records.

        OpenAlex stores each Work's outgoing citation list as an array
        of work URLs. This method fetches the parent, takes up to
        ``limit`` items from its ``referenced_works``, and fans out
        individual ``/works/{id}`` calls so callers receive fully
        populated records rather than bare ids. Returns a deterministic
        citation graph — these are the papers the source paper cites.

        Args:
            work_id: Parent paper ID (accepts the same forms as
                :meth:`fetch_work`).
            limit: Maximum number of referenced Works to return.

        Returns:
            Referenced Works in array order. Individual fan-out failures
            (404 on a neighbor, transient 5xx after retries) are dropped
            silently so one bad neighbor doesn't poison the batch.
            Returns ``()`` when the parent doesn't exist or has no
            ``referenced_works``.

        Raises:
            OalexUnavailable: Persistent failure fetching the parent
                (the fan-out failures don't escalate; only the parent
                fetch's transient errors propagate).
        """
        return await self._fetch_graph_neighbors(
            work_id, field="referenced_works", limit=limit
        )

    async def fetch_related(
        self,
        work_id: str,
        *,
        limit: int = 10,
    ) -> Sequence[Work]:
        """Resolve a Work's ``related_works`` array into full :class:`Work` records.

        Same fan-out shape as :meth:`fetch_referenced`, but reads
        ``related_works`` — OpenAlex's similarity-neighborhood signal
        computed from topic-vector overlap. Unlike ``referenced_works``,
        this isn't a citation relationship: a "related" Work need not
        cite the source Work and vice versa. Treat results as "Works
        OpenAlex thinks are adjacent" rather than ground truth.
        """
        return await self._fetch_graph_neighbors(
            work_id, field="related_works", limit=limit
        )

    async def _fetch_work_at(self, path: str) -> Work | None:
        body = await self._fetch_or_none(path)
        if body is None:
            return None
        try:
            return parse_work(json.loads(body))
        except json.JSONDecodeError as exc:
            _log.exception("oalex: fetch response not JSON")
            raise OalexUnavailable("response was not JSON") from exc

    async def _fetch_graph_neighbors(
        self,
        work_id: str,
        *,
        field: str,
        limit: int,
    ) -> Sequence[Work]:
        prefix, sep, raw = work_id.partition(":")
        if not sep:
            path = f"/works/{work_id}"
        elif prefix == "openalex":
            path = f"/works/{raw}"
        elif prefix == "doi":
            path = f"/works/doi:{raw}"
        else:
            return ()
        parent_body = await self._fetch_or_none(path)
        if parent_body is None:
            return ()
        try:
            parent = json.loads(parent_body)
        except json.JSONDecodeError as exc:
            _log.exception("oalex: parent fetch response not JSON")
            raise OalexUnavailable("response was not JSON") from exc
        neighbor_urls = parent.get(field) or []
        if not isinstance(neighbor_urls, list):
            return ()
        return await self._resolve_works(neighbor_urls, limit=limit)

    async def _resolve_works(
        self,
        work_urls: Sequence[Any],
        *,
        limit: int,
    ) -> Sequence[Work]:
        results: list[Work] = []
        for url in work_urls:
            if len(results) >= limit:
                break
            if not isinstance(url, str):
                continue
            ref_id = _bare_id_from_url(url)
            if not ref_id:
                continue
            try:
                body = await self._fetch_or_none(f"/works/{ref_id}")
            except OalexUnavailable as exc:
                _log.warning(
                    "oalex: neighbor fetch failed for %s: %s", ref_id, exc
                )
                continue
            if body is None:
                continue
            try:
                raw = json.loads(body)
            except json.JSONDecodeError:
                _log.warning("oalex: neighbor response not JSON for %s", ref_id)
                continue
            work = parse_work(raw)
            if work is not None:
                results.append(work)
        return tuple(results)

    async def _fetch(self, path: str, params: dict[str, str] | None = None) -> bytes:
        result = await self._fetch_inner(path, params, allow_404=False)
        return result or b""

    async def _fetch_or_none(self, path: str) -> bytes | None:
        return await self._fetch_inner(path, None, allow_404=True)

    async def _fetch_inner(
        self,
        path: str,
        params: dict[str, str] | None,
        *,
        allow_404: bool,
    ) -> bytes | None:
        merged_params = {"mailto": self._email}
        if params:
            merged_params.update(params)
        cache_key = path + "?" + "&".join(
            f"{k}={v}" for k, v in sorted(merged_params.items())
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        await self._rate.acquire()

        async def do_request() -> httpx.Response:
            return await self._client.get(_API_BASE + path, params=merged_params)

        try:
            response = await with_backoff(do_request)
            if allow_404 and response.status_code == 404:
                return None
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _log.warning("oalex: request failed for %s: %s", path, exc)
            raise OalexUnavailable(str(exc)) from exc
        body = response.content
        self._cache.set(cache_key, body)
        return body


def _parse_search_payload(body: bytes) -> list[Work]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        _log.exception("oalex: search response not JSON")
        raise OalexUnavailable("response was not JSON") from exc
    raw_results = payload.get("results") or []
    return [work for raw in raw_results if (work := parse_work(raw))]


def _bare_id_from_url(value: str) -> str | None:
    marker = "openalex.org/"
    if marker not in value:
        return None
    bare = value.split(marker, 1)[1]
    return bare or None
