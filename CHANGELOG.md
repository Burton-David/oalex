# Changelog

## 0.2.0 (2026-09-22)

OpenAlex retired the mailto polite pool in February 2026 and now meters a daily credit budget. This release catches the client up.

### Breaking

- A 4xx other than 404 and 429 now raises `OalexRequestError`, which is not a subclass of `OalexUnavailable`. Code that retried on `OalexUnavailable` was retrying bad filters and bad keys forever.
- `email` is optional. A blank one no longer raises `ValueError`; OpenAlex ignores `mailto` now anyway.

### Added

- `api_key` argument, falling back to `OPENALEX_API_KEY`. Sent as a Bearer header.
- `OalexRateLimited` for 429s, with the server's `retry_after`. Subclasses `OalexUnavailable`.
- `iter_search` for cursor paging past the first page.
- `fetch_work` accepts `https://openalex.org/W...` URLs; `fetch_doi` accepts a `doi:` prefix.

### Fixed

- A 429 with no credits left (or a `Retry-After` over 30s) is no longer retried.
- Merged-away ids follow OpenAlex's 301 instead of raising.
- DOIs containing `#`, `?` or `;` are percent-encoded instead of being truncated.
- A 200 response that isn't a JSON object is no longer cached for the whole TTL.
- Retries go through the rate limiter.
- Concurrent cache writers no longer share a temp file, and a failed cache write logs instead of failing the request.

### Build

- hatchling instead of setuptools. CI now actually runs Python 3.10 through 3.14 (0.1.0 only ran 3.12).

## 0.1.0 (2026-09-22)

First release, extracted from research-mcp's OpenAlex adapter.
