"""Async client for OpenAlex's ``/works`` endpoint.

The rate limiter uses ``asyncio.Lock``, so one instance belongs to one
event loop and isn't thread-safe. Both ``async with Client() as client:``
and manual ``aclose()`` are supported.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Final, Literal, overload
from urllib.parse import quote, urlencode

import httpx

from oalex._backoff import _parse_retry_after, with_backoff
from oalex._cache import DiskCache
from oalex._rate_limit import RateLimiter
from oalex.errors import OalexRateLimited, OalexRequestError, OalexUnavailable
from oalex.types import Work, _strip_doi_url, _strip_openalex_id_url, parse_work

_log = logging.getLogger(__name__)

_API_BASE: Final = "https://api.openalex.org"
_API_KEY_ENV: Final = "OPENALEX_API_KEY"
_DEFAULT_CACHE_TTL_SECONDS: Final = 24 * 60 * 60
_DEFAULT_TIMEOUT: Final = 30.0
_DEFAULT_MIN_INTERVAL: Final = 0.1
# The API accepted per-page=200 when last checked (Sep 2026) even though the
# current docs say 100. Clamp here so an oversized value never becomes a 400.
_MAX_PER_PAGE: Final = 200


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "oalex"


class Client:
    """Async client for the OpenAlex Works API.

    OpenAlex meters usage against a daily credit budget (since February
    2026). Requests without an API key share a small keyless budget,
    enough for roughly a hundred searches a day. A free key from
    https://openalex.org/settings/api raises it tenfold. Fetching a single
    work by id is free either way; searches and filtered lists are not.

    Args:
        api_key: OpenAlex API key, sent as a Bearer token so it never lands
            in URLs, cache keys, or error messages. Falls back to the
            ``OPENALEX_API_KEY`` environment variable.
        email: Sent as ``mailto=``. OpenAlex has ignored it since the
            polite pool was retired; it stays accepted so older callers
            keep working.
        cache_dir: Disk cache location. Defaults to ``~/.cache/oalex/``.
        ttl_seconds: Cache TTL in seconds. Defaults to 24 hours.
        min_interval_seconds: Minimum time between requests. Defaults
            to 0.1s (10 req/sec; OpenAlex's hard ceiling is 100).
        timeout: Per-request HTTP timeout in seconds. Defaults to 30s.
            Ignored when ``client`` is given; configure that client instead.
        client: Optional pre-built :class:`httpx.AsyncClient` to reuse.
            When provided, ``aclose()`` does NOT close it; that's the
            caller's responsibility.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        email: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
        min_interval_seconds: float = _DEFAULT_MIN_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(_API_KEY_ENV)
        self._headers = {"Authorization": f"Bearer {key.strip()}"} if key and key.strip() else {}
        self._email = email.strip() if email and email.strip() else None
        cache_path = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self._cache = DiskCache(cache_path, ttl_seconds=ttl_seconds)
        self._rate = RateLimiter(min_interval_seconds)
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(timeout=timeout)

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
        """Search OpenAlex's full-text index and return the first page.

        Args:
            query: Freeform query string passed to OpenAlex's ``search=``.
            per_page: Maximum number of results to return, clamped to
                1..200. Use :meth:`iter_search` for more than one page.
            year_min: Earliest publication year, inclusive.
            year_max: Latest publication year, inclusive.

        Returns:
            Works in OpenAlex's relevance order. Records without a
            parseable id are skipped.

        Raises:
            OalexRateLimited: Daily credit budget exhausted, or 429s outlasted retries.
            OalexRequestError: OpenAlex rejected the query (bad filter, bad key).
            OalexUnavailable: Transient upstream failure (5xx after
                retries, network error, malformed JSON).
        """
        params = _search_params(query, per_page=per_page, year_min=year_min, year_max=year_max)
        payload = await self._get_json("/works", params)
        return _parse_results(payload)

    async def iter_search(
        self,
        query: str,
        *,
        year_min: int | None = None,
        year_max: int | None = None,
        per_page: int = _MAX_PER_PAGE,
        max_results: int | None = None,
    ) -> AsyncIterator[Work]:
        """Yield search results across pages using OpenAlex's cursor paging.

        Each page is one metered list call, so on the keyless budget a
        broad query can run the day's credits out long before the result
        set ends. Set ``max_results`` unless you really want everything.
        Offset paging stops at 10,000 results; cursor paging doesn't.
        """
        params = _search_params(query, per_page=per_page, year_min=year_min, year_max=year_max)
        cursor: str | None = "*"
        yielded = 0
        while cursor is not None and (max_results is None or yielded < max_results):
            payload = await self._get_json("/works", {**params, "cursor": cursor})
            works = _parse_results(payload)
            for work in works:
                yield work
                yielded += 1
                # Return mid-page rather than at the top of the loop, which
                # would first pay for a page we'd throw away.
                if max_results is not None and yielded >= max_results:
                    return
            meta = payload.get("meta")
            next_cursor = meta.get("next_cursor") if isinstance(meta, dict) else None
            # Stop on an empty page even if a next_cursor came back; trusting
            # it would let a misbehaving cursor bill us per empty page forever.
            cursor = next_cursor if isinstance(next_cursor, str) and works else None

    async def fetch_work(self, work_id: str) -> Work | None:
        """Fetch a single Work by OpenAlex ID or DOI.

        Args:
            work_id: A bare OpenAlex ID (``W2626778328``), an OpenAlex URL
                (``https://openalex.org/W2626778328``, the form ``Work.url``
                and ``raw["referenced_works"]`` use), an ``openalex:``
                prefixed ID, or a ``doi:`` prefixed DOI
                (``doi:10.1038/nature12373``). Bare DOIs are not
                recognized; use :meth:`fetch_doi` for those.

        Returns:
            A :class:`Work`, or ``None`` if OpenAlex returns 404 or the id
            has a prefix this client doesn't know. Merged-away ids follow
            OpenAlex's redirect to the surviving record.

        Raises:
            OalexRateLimited: Daily credit budget exhausted.
            OalexRequestError: OpenAlex rejected the request.
            OalexUnavailable: Transient upstream failure.
        """
        path = _work_path(work_id)
        if path is None:
            return None
        return await self._fetch_work_at(path)

    async def fetch_doi(self, doi: str) -> Work | None:
        """Fetch a Work by DOI.

        Accepts a bare DOI (``10.1038/nature12373``), a ``doi:`` prefixed
        one, or a ``doi.org`` URL.
        """
        bare = _strip_doi_url(doi.removeprefix("doi:"))
        if bare is None:
            return None
        return await self._fetch_work_at(_doi_path(bare))

    async def fetch_referenced(
        self,
        work_id: str,
        *,
        limit: int = 10,
    ) -> Sequence[Work]:
        """Resolve a Work's ``referenced_works`` array into full :class:`Work` records.

        OpenAlex stores each Work's outgoing citation list as an array of
        work URLs. This fetches the parent, then fetches neighbors in array
        order until ``limit`` of them resolve. Neighbor fetches are
        singleton lookups, which OpenAlex doesn't charge credits for; a
        batched ``ids.openalex`` filter would be one request but is billed
        as a list call.

        Args:
            work_id: Parent paper ID (accepts the same forms as
                :meth:`fetch_work`).
            limit: Maximum number of referenced Works to return.

        Returns:
            Referenced Works in array order. A neighbor that 404s or keeps
            failing after retries is skipped and the next one takes its
            place. Returns ``()`` when the parent doesn't exist or has no
            ``referenced_works``.

        Raises:
            OalexUnavailable: Persistent failure fetching the parent.
                Neighbor failures never escalate.
        """
        return await self._fetch_graph_neighbors(work_id, field="referenced_works", limit=limit)

    async def fetch_related(
        self,
        work_id: str,
        *,
        limit: int = 10,
    ) -> Sequence[Work]:
        """Resolve a Work's ``related_works`` array into full :class:`Work` records.

        Same fan-out shape as :meth:`fetch_referenced`, but reads
        ``related_works``, OpenAlex's similarity neighborhood computed from
        topic overlap. It isn't a citation relationship: a "related" Work
        need not cite the source Work or be cited by it.
        """
        return await self._fetch_graph_neighbors(work_id, field="related_works", limit=limit)

    async def _fetch_work_at(self, path: str) -> Work | None:
        payload = await self._get_json(path, allow_404=True)
        return parse_work(payload) if payload is not None else None

    async def _fetch_graph_neighbors(
        self,
        work_id: str,
        *,
        field: str,
        limit: int,
    ) -> Sequence[Work]:
        path = _work_path(work_id)
        if path is None:
            return ()
        parent = await self._get_json(path, allow_404=True)
        if parent is None:
            return ()
        neighbor_urls = parent.get(field)
        if not isinstance(neighbor_urls, list):
            return ()
        ids = [oa_id for url in neighbor_urls if (oa_id := _strip_openalex_id_url(url))]
        return await self._resolve_works(ids, limit=limit)

    async def _resolve_works(self, ids: Sequence[str], *, limit: int) -> Sequence[Work]:
        results: list[Work] = []
        pending = list(ids)
        # Fetch in waves sized to the shortfall so a 404 or dead neighbor gets
        # replaced by the next id in order, without fetching past the limit
        # when everything resolves.
        while pending and len(results) < limit:
            wave, pending = pending[: limit - len(results)], pending[limit - len(results) :]
            fetched = await asyncio.gather(*(self._fetch_neighbor(ref_id) for ref_id in wave))
            results.extend(work for work in fetched if work is not None)
        return tuple(results)

    async def _fetch_neighbor(self, ref_id: str) -> Work | None:
        try:
            return await self._fetch_work_at(f"/works/{quote(ref_id, safe='')}")
        except (OalexUnavailable, OalexRequestError) as exc:
            _log.warning("oalex: neighbor fetch failed for %s: %s", ref_id, exc)
            return None

    @overload
    async def _get_json(
        self, path: str, params: dict[str, str] | None = ..., *, allow_404: Literal[False] = ...
    ) -> dict[str, Any]: ...

    @overload
    async def _get_json(
        self, path: str, params: dict[str, str] | None = ..., *, allow_404: Literal[True]
    ) -> dict[str, Any] | None: ...

    async def _get_json(
        self,
        path: str,
        params: dict[str, str] | None = None,
        *,
        allow_404: bool = False,
    ) -> dict[str, Any] | None:
        merged: dict[str, str] = {"mailto": self._email} if self._email else {}
        if params:
            merged.update(params)
        cache_key = path + "?" + urlencode(sorted(merged.items()))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return _decode(cached)

        async def do_request() -> httpx.Response:
            # Inside the retried callable so retries also wait their turn.
            await self._rate.acquire()
            # OpenAlex answers merged-away ids with a 301 to the survivor.
            return await self._client.get(
                _API_BASE + path,
                params=merged,
                headers=self._headers,
                follow_redirects=True,
            )

        try:
            response = await with_backoff(do_request)
        except httpx.HTTPError as exc:
            raise OalexUnavailable(f"request to {path} failed: {exc!r}") from exc
        status = response.status_code
        if allow_404 and status == 404:
            return None
        if status == 429:
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            raise OalexRateLimited(
                f"OpenAlex rate limit hit on {path}: {_error_detail(response)}",
                retry_after=retry_after,
            )
        if status >= 500:
            raise OalexUnavailable(f"OpenAlex returned {status} for {path}")
        if status >= 400:
            raise OalexRequestError(
                f"OpenAlex returned {status} for {path}: {_error_detail(response)}",
                status_code=status,
            )
        payload = _decode(response.content)
        # Cache after decoding so an HTML error page served with a 200 by some
        # proxy doesn't get pinned for a whole TTL.
        self._cache.set(cache_key, response.content)
        return payload


def _search_params(
    query: str, *, per_page: int, year_min: int | None, year_max: int | None
) -> dict[str, str]:
    params = {
        "search": query,
        "per-page": str(min(max(per_page, 1), _MAX_PER_PAGE)),
    }
    if year_min is not None or year_max is not None:
        # Open-ended ranges ("2018-") are valid OpenAlex filter syntax.
        lo = str(year_min) if year_min is not None else ""
        hi = str(year_max) if year_max is not None else ""
        params["filter"] = f"publication_year:{lo}-{hi}"
    return params


def _work_path(work_id: str) -> str | None:
    bare = _strip_openalex_id_url(work_id)
    if bare is not None:
        return f"/works/{quote(bare, safe='')}"
    prefix, sep, raw = work_id.partition(":")
    if not sep:
        return f"/works/{quote(work_id, safe='')}"
    if prefix == "openalex":
        return f"/works/{quote(raw, safe='')}"
    if prefix == "doi":
        return _doi_path(raw)
    return None


def _doi_path(doi: str) -> str:
    # DOIs may contain '#', '?' and ';' (old SICI-style Wiley DOIs do);
    # unescaped, httpx would read those as a fragment or query string.
    return f"/works/doi:{quote(doi, safe='/:')}"


def _decode(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OalexUnavailable("response was not JSON") from exc
    if not isinstance(payload, dict):
        raise OalexUnavailable(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def _parse_results(payload: dict[str, Any]) -> list[Work]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []
    return [work for raw in raw_results if isinstance(raw, dict) and (work := parse_work(raw))]


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or body)[:200]
    return str(body)[:200]
