# oalex

[![CI](https://github.com/Burton-David/oalex/actions/workflows/ci.yml/badge.svg)](https://github.com/Burton-David/oalex/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Async Python client for the [OpenAlex](https://openalex.org/) scholarly works API. Typed, rate-limited, disk-cached, polite-pool-first.

```python
import asyncio
from oalex import Client

async def main() -> None:
    async with Client(email="you@example.com") as oa:
        works = await oa.search("attention is all you need", per_page=5)
        for w in works:
            print(w.id, w.title, w.citation_count)

        vaswani = await oa.fetch_work("W2626778328")
        if vaswani is not None:
            print(vaswani.title, "→", len(vaswani.referenced_works), "references")

asyncio.run(main())
```

## Why this exists

OpenAlex is a comprehensive open scholarly graph (>240M works, >1B citation edges). The HTTP API is clean — but the [polite-pool rules](https://docs.openalex.org/how-to-use-the-api/api-overview#authentication), inverted-index abstract reconstruction, and per-page caps add friction to every greenfield client. `oalex` packages those concerns once.

## Features

- **Typed** — `Work` and `Author` are frozen dataclasses; `Work.raw` exposes the full payload for fields the typed surface doesn't cover.
- **Async** — `httpx` under the hood; `async with Client(...)` for clean teardown.
- **Polite-pool aware** — `email` is required at construction; every request carries `mailto=` so you land in the fast pool.
- **Rate-limited** — defaults to 0.1s minimum interval (~10 req/sec, the documented polite-pool tolerance).
- **Disk-cached** — `~/.cache/oalex/` with a 24-hour TTL by default; configurable via constructor kwargs.
- **Retried** — 1s/2s/4s exponential backoff on 429 and 5xx, honoring `Retry-After`.
- **Citation graph** — `fetch_referenced` and `fetch_related` resolve OpenAlex's `referenced_works` / `related_works` arrays into full `Work` records, one fan-out per neighbor.

## Install

```bash
pip install oalex
```

Python 3.10+. Only runtime dependency is [httpx](https://www.python-httpx.org/).

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

`per_page` is capped at 200 (OpenAlex's per-page limit); for larger result sets, paginate yourself using OpenAlex's `cursor` parameter against `Client._fetch` or roll your own wrapper.

### Fetch a single work

```python
# By OpenAlex ID — three equivalent forms
work = await client.fetch_work("W2626778328")
work = await client.fetch_work("openalex:W2626778328")

# By DOI
work = await client.fetch_doi("10.1038/nature12373")
work = await client.fetch_work("doi:10.1038/nature12373")
```

`fetch_work` returns `None` for 404 (the ID doesn't exist in OpenAlex's graph) and raises `OalexUnavailable` for transient failures so you can distinguish "not found" from "outage."

### Citation graph

```python
# Papers this paper cites
refs = await client.fetch_referenced("W2626778328", limit=10)

# Papers OpenAlex considers similar (topic-vector overlap, not citations)
related = await client.fetch_related("W2626778328", limit=10)
```

Individual neighbor failures are skipped silently so one bad reference doesn't poison the batch; the parent fetch's transient errors propagate as `OalexUnavailable`.

### Raw payload access

`Work.raw` is a read-only mapping of the full OpenAlex response. Reach into it for fields the typed surface doesn't expose:

```python
concepts = work.raw.get("concepts", [])
host_venue_id = work.raw.get("host_venue", {}).get("id")
```

## Configuration

```python
Client(
    email="you@example.com",          # required — polite-pool contact
    cache_dir="/var/cache/oalex",     # default: ~/.cache/oalex
    ttl_seconds=24 * 60 * 60,         # default: 24h
    min_interval_seconds=0.1,         # default: 0.1s (polite-pool tolerance)
    timeout=30.0,                     # default: 30s per request
    client=my_httpx_client,           # optional: bring your own AsyncClient
)
```

When you pass your own `httpx.AsyncClient`, `oalex` won't close it on exit — that's your responsibility.

## Errors

- `OalexError` — base class for all oalex-defined exceptions.
- `OalexUnavailable` — transient failure (network error, 429/5xx after retries, malformed JSON). Always safe to retry.

404s from `fetch_work` / `fetch_doi` are NOT raised — they return `None` so you can distinguish "OpenAlex doesn't know this ID" from "OpenAlex is having a bad day."

## Development

```bash
git clone https://github.com/Burton-David/oalex
cd oalex
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check oalex tests
mypy oalex
```

## Credits

Extracted from [research-mcp](https://github.com/Burton-David/ResearchAssistantMCP)'s OpenAlex source adapter. The polite-pool design, retry policy, and disk-cache invariants were settled there first.

## License

MIT — see `LICENSE`.
