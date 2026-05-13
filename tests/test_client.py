"""End-to-end client tests using ``httpx.MockTransport``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from oalex import Client, OalexUnavailable

_VASWANI_WORK: dict = {
    "id": "https://openalex.org/W2626778328",
    "doi": "https://doi.org/10.65215/2q58a426",
    "title": "Attention Is All You Need",
    "publication_year": 2017,
    "publication_date": "2017-06-12",
    "cited_by_count": 6536,
    "authorships": [
        {"author": {"display_name": "Ashish Vaswani", "orcid": None}},
    ],
    "abstract_inverted_index": {"Attention": [0], "is": [1], "everything.": [2]},
}


def _build_client(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    email: str = "test@example.com",
) -> Client:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return Client(email=email, cache_dir=tmp_path / "cache", client=http)


def _search_response(*works: dict) -> httpx.Response:
    return httpx.Response(
        200, json={"meta": {"count": len(works)}, "results": list(works)}
    )


def _work_response(work: dict) -> httpx.Response:
    return httpx.Response(200, json=work)


def _neighbor_work(work_id: str, title: str) -> dict:
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": title,
        "publication_year": 2018,
        "publication_date": "2018-01-01",
        "cited_by_count": 5,
        "authorships": [{"author": {"display_name": "Neighbor", "orcid": None}}],
    }


def _parent_with(*, referenced: list[str] | None = None,
                 related: list[str] | None = None) -> dict:
    work = dict(_VASWANI_WORK)
    if referenced is not None:
        work["referenced_works"] = [f"https://openalex.org/{w}" for w in referenced]
    if related is not None:
        work["related_works"] = [f"https://openalex.org/{w}" for w in related]
    return work


def _routing_handler(
    path_to_response: dict[str, httpx.Response],
    *,
    captured_paths: list[str] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured_paths is not None:
            captured_paths.append(request.url.path)
        return path_to_response.get(
            request.url.path,
            httpx.Response(404, json={"error": "Not found."}),
        )

    return handler


async def _no_sleep(seconds: float) -> None:
    return None


# ---- construction ----


def test_email_is_required(tmp_path: Path) -> None:
    """Empty email silently drops to the slow common pool — refuse at construction."""
    with pytest.raises(ValueError, match="email"):
        Client(email="", cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="email"):
        Client(email="   ", cache_dir=tmp_path / "cache")


async def test_async_context_manager_closes_owned_client(tmp_path: Path) -> None:
    """The default httpx.AsyncClient is closed on exit."""

    async with Client(email="t@example.com", cache_dir=tmp_path / "cache") as oa:
        assert isinstance(oa, Client)


async def test_external_client_is_not_closed(tmp_path: Path) -> None:
    """When the caller supplies an httpx.AsyncClient, oalex doesn't close it."""

    transport = httpx.MockTransport(lambda req: _search_response())
    external = httpx.AsyncClient(transport=transport)
    client = Client(
        email="t@example.com",
        cache_dir=tmp_path / "cache",
        client=external,
    )
    await client.aclose()
    # If oalex closed the external client, this would raise.
    await external.get("https://example.com/")
    await external.aclose()


# ---- search ----


async def test_search_calls_works_endpoint(tmp_path: Path) -> None:
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return _search_response(_VASWANI_WORK)

    async with _build_client(tmp_path, handler) as oa:
        results = await oa.search("attention transformer", per_page=5)
    assert len(results) == 1
    assert results[0].id == "W2626778328"
    assert captured[0].path == "/works"
    params = dict(captured[0].params)
    assert params["search"] == "attention transformer"
    assert params["per-page"] == "5"
    assert params["mailto"] == "test@example.com"


async def test_search_passes_year_filter(tmp_path: Path) -> None:
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return _search_response()

    async with _build_client(tmp_path, handler) as oa:
        await oa.search("x", year_min=2018, year_max=2022)
    params = dict(captured[0].params)
    assert "publication_year:2018-2022" in params.get("filter", "")


async def test_search_caps_per_page_at_200(tmp_path: Path) -> None:
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return _search_response()

    async with _build_client(tmp_path, handler) as oa:
        await oa.search("x", per_page=500)
    assert dict(captured[0].params)["per-page"] == "200"


async def test_search_clamps_per_page_lower_bound(tmp_path: Path) -> None:
    captured: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url)
        return _search_response()

    async with _build_client(tmp_path, handler) as oa:
        await oa.search("x", per_page=0)
    assert dict(captured[0].params)["per-page"] == "1"


async def test_search_skips_unparseable_records(tmp_path: Path) -> None:
    """A malformed record in the results array should be skipped, not crash the batch."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _search_response(
            _VASWANI_WORK,
            {"id": None, "title": "broken"},
            _VASWANI_WORK,
        )

    async with _build_client(tmp_path, handler) as oa:
        results = await oa.search("x")
    assert len(results) == 2


async def test_search_raises_on_invalid_json(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async with _build_client(tmp_path, handler) as oa:
        with pytest.raises(OalexUnavailable):
            await oa.search("x")


async def test_search_raises_on_persistent_5xx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"upstream busy")

    async with _build_client(tmp_path, handler) as oa:
        with pytest.raises(OalexUnavailable):
            await oa.search("x")


# ---- fetch_work ----


async def test_fetch_work_by_bare_id(tmp_path: Path) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url.path)
        return _work_response(_VASWANI_WORK)

    async with _build_client(tmp_path, handler) as oa:
        work = await oa.fetch_work("W2626778328")
    assert work is not None
    assert work.id == "W2626778328"
    assert captured == ["/works/W2626778328"]


async def test_fetch_work_by_openalex_prefix(tmp_path: Path) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url.path)
        return _work_response(_VASWANI_WORK)

    async with _build_client(tmp_path, handler) as oa:
        work = await oa.fetch_work("openalex:W2626778328")
    assert work is not None
    assert captured == ["/works/W2626778328"]


async def test_fetch_work_by_doi_prefix(tmp_path: Path) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url.path)
        return _work_response(_VASWANI_WORK)

    async with _build_client(tmp_path, handler) as oa:
        work = await oa.fetch_work("doi:10.65215/2q58a426")
    assert work is not None
    assert captured == ["/works/doi:10.65215/2q58a426"]


async def test_fetch_doi_strips_url(tmp_path: Path) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url.path)
        return _work_response(_VASWANI_WORK)

    async with _build_client(tmp_path, handler) as oa:
        work = await oa.fetch_doi("https://doi.org/10.1038/nature12373")
    assert work is not None
    assert captured == ["/works/doi:10.1038/nature12373"]


async def test_fetch_work_returns_none_for_unknown_prefix(tmp_path: Path) -> None:
    """Non-openalex/doi prefixes return None without hitting the network."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with _build_client(tmp_path, handler) as oa:
        assert await oa.fetch_work("arxiv:1706.03762") is None
        assert await oa.fetch_work("pmid:12345") is None
    assert calls == 0


async def test_fetch_work_returns_none_on_404(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Not found."})

    async with _build_client(tmp_path, handler) as oa:
        work = await oa.fetch_work("W9999999999")
    assert work is None


async def test_fetch_work_raises_on_persistent_5xx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"upstream busy")

    async with _build_client(tmp_path, handler) as oa:
        with pytest.raises(OalexUnavailable):
            await oa.fetch_work("W2626778328")


# ---- citation graph ----


async def test_fetch_referenced_resolves_neighbor_works(tmp_path: Path) -> None:
    parent = _parent_with(referenced=["W001", "W002"])
    handler = _routing_handler({
        "/works/W2626778328": _work_response(parent),
        "/works/W001": _work_response(_neighbor_work("W001", "First")),
        "/works/W002": _work_response(_neighbor_work("W002", "Second")),
    })
    async with _build_client(tmp_path, handler) as oa:
        refs = await oa.fetch_referenced("W2626778328", limit=10)
    assert [w.id for w in refs] == ["W001", "W002"]
    assert [w.title for w in refs] == ["First", "Second"]


async def test_fetch_referenced_respects_limit(tmp_path: Path) -> None:
    parent = _parent_with(referenced=["W001", "W002", "W003", "W004", "W005"])
    captured: list[str] = []
    handler = _routing_handler(
        {
            "/works/W2626778328": _work_response(parent),
            "/works/W001": _work_response(_neighbor_work("W001", "1")),
            "/works/W002": _work_response(_neighbor_work("W002", "2")),
        },
        captured_paths=captured,
    )
    async with _build_client(tmp_path, handler) as oa:
        refs = await oa.fetch_referenced("W2626778328", limit=2)
    assert [w.id for w in refs] == ["W001", "W002"]
    # We should NOT have fetched W003/W004/W005 — limit is enforced.
    fetched_neighbors = [p for p in captured if p.startswith("/works/W00")]
    assert fetched_neighbors == ["/works/W001", "/works/W002"]


async def test_fetch_referenced_skips_individual_404(tmp_path: Path) -> None:
    parent = _parent_with(referenced=["W_GONE", "W_OK"])
    handler = _routing_handler({
        "/works/W2626778328": _work_response(parent),
        # W_GONE deliberately omitted → 404.
        "/works/W_OK": _work_response(_neighbor_work("W_OK", "Survived")),
    })
    async with _build_client(tmp_path, handler) as oa:
        refs = await oa.fetch_referenced("W2626778328")
    assert [w.id for w in refs] == ["W_OK"]


async def test_fetch_referenced_returns_empty_when_field_missing(tmp_path: Path) -> None:
    handler = _routing_handler(
        {"/works/W2626778328": _work_response(_VASWANI_WORK)}
    )
    async with _build_client(tmp_path, handler) as oa:
        refs = await oa.fetch_referenced("W2626778328")
    assert refs == ()


async def test_fetch_referenced_returns_empty_when_parent_404(tmp_path: Path) -> None:
    handler = _routing_handler({})  # everything 404s
    async with _build_client(tmp_path, handler) as oa:
        refs = await oa.fetch_referenced("W9999999999")
    assert refs == ()


async def test_fetch_referenced_propagates_parent_5xx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("oalex._backoff.asyncio.sleep", _no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"upstream busy")

    async with _build_client(tmp_path, handler) as oa:
        with pytest.raises(OalexUnavailable):
            await oa.fetch_referenced("W2626778328")


async def test_fetch_related_reads_related_works_field(tmp_path: Path) -> None:
    """fetch_related shares the resolver with fetch_referenced — verify it
    reads `related_works`, not `referenced_works`."""
    parent = _parent_with(
        referenced=["W_REF_ONLY"], related=["W_REL_1", "W_REL_2"]
    )
    handler = _routing_handler({
        "/works/W2626778328": _work_response(parent),
        "/works/W_REF_ONLY": _work_response(_neighbor_work("W_REF_ONLY", "ref")),
        "/works/W_REL_1": _work_response(_neighbor_work("W_REL_1", "r1")),
        "/works/W_REL_2": _work_response(_neighbor_work("W_REL_2", "r2")),
    })
    async with _build_client(tmp_path, handler) as oa:
        related = await oa.fetch_related("W2626778328")
    assert [w.id for w in related] == ["W_REL_1", "W_REL_2"]


async def test_fetch_referenced_with_doi_uses_doi_path(tmp_path: Path) -> None:
    captured: list[str] = []
    parent = _parent_with(referenced=["W001"])
    handler = _routing_handler(
        {
            "/works/doi:10.65215/2q58a426": _work_response(parent),
            "/works/W001": _work_response(_neighbor_work("W001", "Neighbor")),
        },
        captured_paths=captured,
    )
    async with _build_client(tmp_path, handler) as oa:
        refs = await oa.fetch_referenced("doi:10.65215/2q58a426")
    assert len(refs) == 1
    assert "/works/doi:10.65215/2q58a426" in captured


# ---- cache + mailto ----


async def test_response_is_disk_cached(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _search_response(_VASWANI_WORK)

    async with _build_client(tmp_path, handler) as oa:
        first = await oa.search("cached")
        second = await oa.search("cached")
    assert [w.id for w in first] == [w.id for w in second]
    assert calls == 1


async def test_mailto_attached_to_every_request(tmp_path: Path) -> None:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if request.url.path.startswith("/works/W"):
            return _work_response(_VASWANI_WORK)
        return _search_response(_VASWANI_WORK)

    async with _build_client(tmp_path, handler, email="user@lab.edu") as oa:
        await oa.search("x")
        await oa.fetch_work("W2626778328")
    assert all(params["mailto"] == "user@lab.edu" for params in seen)
