# oalex

[![CI](https://github.com/Burton-David/oalex/actions/workflows/ci.yml/badge.svg)](https://github.com/Burton-David/oalex/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%20to%203.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Async Python client for the [OpenAlex](https://openalex.org/) scholarly works API. Typed, rate-limited, disk-cached, and careful with your daily credit budget.

```python
import asyncio
from oalex import Client

async def main() -> None:
    async with Client(api_key="your-openalex-key") as oa:
        works = await oa.search("attention is all you need", per_page=5)
        for w in works:
            print(w.id, w.title, w.citation_count)

        vaswani = await oa.fetch_work("W2626778328")
        if vaswani is not None:
            print(vaswani.title, "cites", len(vaswani.referenced_works), "works")

asyncio.run(main())
```

## Why this exists

The OpenAlex HTTP API is clean, but every new client ends up re-solving the same things: rebuilding abstracts from the inverted index, retrying 429s and 5xx without burning the daily budget, following redirects for merged records, and caching so repeat lookups cost nothing. `oalex` does those once.

## Features

- **Typed.** `Work` and `Author` are frozen dataclasses. `Work.raw` exposes the full payload for fields the typed surface doesn't cover.
- **Async.** `httpx` under the hood; `async with Client(...)` for clean teardown.
- **API-key aware.** Pass `api_key=` or set `OPENALEX_API_KEY`. The key travels as a Bearer header, so it never shows up in URLs, cache files, or exception messages.
- **Rate-limited.** One request per 0.1s by default, well under OpenAlex's 100 requests/second ceiling. Retries wait their turn too.
- **Disk-cached.** `~/.cache/oalex/` with a 24-hour TTL. Only valid JSON is cached.
- **Retried.** 1s/2s/4s backoff on 429 and 5xx, honoring `Retry-After` up to 30s. When a 429 says the daily budget is spent (`X-RateLimit-Remaining: 0`, or a `Retry-After` longer than 30s), the client raises `OalexRateLimited` at once instead of sleeping.
- **Cursor paging.** `iter_search` walks past the first page (and past the 10,000-result offset limit).
- **Citation graph.** `fetch_referenced` and `fetch_related` resolve OpenAlex's `referenced_works` / `related_works` arrays into full `Work` records.

## Install

Not on PyPI yet. Install from GitHub:

```bash
pip install git+https://github.com/Burton-David/oalex
```

Python 3.10+. The only runtime dependency is [httpx](https://www.python-httpx.org/).

## OpenAlex credits

Since February 2026 OpenAlex meters usage against a daily budget that resets at midnight UTC. These are the numbers the live API reported in September 2026:

| Call | oalex method | Credits |
|------|--------------|---------|
| Single work by id or DOI | `fetch_work`, `fetch_doi`, each neighbor in `fetch_referenced` / `fetch_related` | 0 |
| Search page | `search`, each page of `iter_search` | 10 |

A keyless client gets 1,000 credits per day, which is about 100 searches. A [free API key](https://openalex.org/settings/api) raises the budget tenfold. OpenAlex has changed these numbers before; its [pricing page](https://help.openalex.org/access/pricing/) is the source of truth. Cache hits cost nothing.

## Usage

### Search

```python
works = await client.search(
    "graph neural networks",
    per_page=25,
    year_min=2020,
    year_max=2024,
)
```

`search` returns one page. `per_page` is clamped to 1..200. Either year bound can be left off for an open-ended range.

For more than one page, iterate:

```python
async for work in client.iter_search("graph neural networks", max_results=1000):
    print(work.id, work.title)
```

Each page is a metered call, so set `max_results` on broad queries.

### Fetch a single work

```python
# By OpenAlex ID: three equivalent forms
work = await client.fetch_work("W2626778328")
work = await client.fetch_work("openalex:W2626778328")
work = await client.fetch_work("https://openalex.org/W2626778328")

# By DOI
work = await client.fetch_doi("10.1038/nature12373")
work = await client.fetch_doi("https://doi.org/10.1038/nature12373")
work = await client.fetch_work("doi:10.1038/nature12373")
```

`fetch_work` returns `None` for a 404 and for an id prefix it doesn't recognize (`arxiv:`, `pmid:`). A merged-away id follows OpenAlex's redirect and returns the surviving record.

### Citation graph

```python
# Papers this paper cites
refs = await client.fetch_referenced("W2626778328", limit=10)

# Papers OpenAlex considers similar (topic overlap, not citations)
related = await client.fetch_related("W2626778328", limit=10)
```

Neighbors are fetched concurrently, in array order. A neighbor that 404s or keeps failing is skipped, and the next id in the array takes its place. Errors fetching the parent propagate.

### Raw payload access

`Work.raw` is a read-only mapping of the full OpenAlex response. Reach into it for fields the typed surface doesn't expose:

```python
topics = [t["display_name"] for t in work.raw.get("topics", [])]
source = (work.raw.get("primary_location") or {}).get("source") or {}
source_id = source.get("id")
```

## Configuration

```python
Client(
    api_key="...",                    # default: $OPENALEX_API_KEY, else keyless
    cache_dir="/var/cache/oalex",     # default: ~/.cache/oalex
    ttl_seconds=24 * 60 * 60,         # default: 24h
    min_interval_seconds=0.1,         # default: 0.1s between requests
    timeout=30.0,                     # default: 30s per request
    client=my_httpx_client,           # optional: bring your own AsyncClient
)
```

When you pass your own `httpx.AsyncClient`, `oalex` won't close it on exit, and `timeout` is ignored in favor of that client's settings.

`email=` is still accepted and sent as `mailto=`. OpenAlex has ignored it since retiring the polite pool, so new code can leave it off.

## Errors

- `OalexError` is the base class for everything below.
- `OalexUnavailable` covers transient failures: network errors, 5xx after retries, and a body that isn't a JSON object. Retrying later is reasonable.
- `OalexRateLimited` is a subclass of `OalexUnavailable` for 429s. `retry_after` holds the server's `Retry-After` hint in seconds when it sent one. A spent daily budget refills at midnight UTC.
- `OalexRequestError` covers any other 4xx except 404: a bad filter or an invalid API key, for example. `status_code` holds the HTTP status. Retrying the same request won't help.

A 404 is not an error: `fetch_work` and `fetch_doi` return `None`, so "OpenAlex doesn't know this id" stays distinct from "OpenAlex is having a bad day."

## Limits

- Only the `/works` endpoint is wrapped. Authors, sources, and institutions are reachable only through `Work.raw`.
- The rate limiter is per process. Several processes sharing one API key also share one daily budget, and nothing here coordinates them.
- The cache stores responses as individual files and never prunes them. Clear `~/.cache/oalex/` yourself if it grows.

## Development

```bash
git clone https://github.com/Burton-David/oalex
cd oalex
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check oalex tests
mypy oalex
pytest
```

Tests use `httpx.MockTransport` and never touch the live API.

## Credits

Extracted from [research-mcp](https://github.com/Burton-David/ResearchAssistantMCP)'s OpenAlex source adapter, where the retry policy and disk-cache design were settled first.

## License

MIT. See `LICENSE`.
