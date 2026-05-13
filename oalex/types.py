"""Typed wrappers for OpenAlex Work and Author payloads.

OpenAlex's Work JSON is large (50+ fields) and changes shape over time;
``Work`` exposes the commonly-used fields as typed attributes while
preserving the raw payload at ``.raw`` for anything the typed surface
doesn't cover. This keeps the package's surface small and lets callers
reach into the full response without subclassing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class Author:
    """An author on an OpenAlex Work.

    ``orcid`` is the full ORCID URL when OpenAlex has one — many records
    don't, in which case it's ``None``.
    """

    name: str
    orcid: str | None = None


@dataclass(frozen=True, slots=True)
class Work:
    """A scholarly work parsed from OpenAlex's ``/works`` endpoint.

    ``id`` is the bare OpenAlex ID (``W2626778328``), not the full URL
    that OpenAlex returns. Use ``.url`` for the canonical URL form.

    ``raw`` holds the full deserialized JSON payload for fields not
    surfaced as typed attributes (institutions, concepts, host_venue,
    abstract_inverted_index, etc.).
    """

    id: str
    title: str
    abstract: str
    authors: tuple[Author, ...]
    published: date | None = None
    url: str | None = None
    doi: str | None = None
    venue: str | None = None
    pdf_url: str | None = None
    citation_count: int | None = None
    referenced_works: tuple[str, ...] = ()
    """Bare OpenAlex IDs of works this work cites. Empty when OpenAlex
    has no citation graph for this record (older items, non-citable
    types) or when the field was stripped by a ``select=`` parameter."""
    related_works: tuple[str, ...] = ()
    """Bare OpenAlex IDs of works OpenAlex considers similar. Computed
    from topic-vector overlap, not a citation relationship."""
    field_name: str | None = None
    """``primary_topic.field.display_name`` when present — coarse
    discipline label ("Computer Science", "Medicine", "Mathematics")."""
    raw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


def parse_work(raw: Mapping[str, Any] | None) -> Work | None:
    """Parse an OpenAlex Work JSON payload into a :class:`Work`.

    Returns ``None`` when the payload is empty or lacks a parseable
    ``id`` — OpenAlex sometimes returns placeholder records (merged
    duplicates, withdrawn entries) that we don't want to surface to
    callers.
    """
    if not raw:
        return None
    openalex_id = _strip_openalex_id_url(raw.get("id"))
    if not openalex_id:
        return None
    title = (raw.get("title") or raw.get("display_name") or "").strip()
    abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))
    published = _parse_publication_date(
        raw.get("publication_date"), raw.get("publication_year")
    )
    doi = _strip_doi_url(raw.get("doi"))
    venue = _extract_venue(raw)
    pdf_url = _extract_pdf_url(raw)
    authors = _extract_authors(raw)
    raw_count = raw.get("cited_by_count")
    citation_count = raw_count if isinstance(raw_count, int) else None
    referenced = _extract_id_list(raw.get("referenced_works"))
    related = _extract_id_list(raw.get("related_works"))
    field_name = _extract_field_name(raw)
    return Work(
        id=openalex_id,
        title=title,
        abstract=abstract,
        authors=authors,
        published=published,
        url=raw.get("id") if isinstance(raw.get("id"), str) else None,
        doi=doi,
        venue=venue,
        pdf_url=pdf_url,
        citation_count=citation_count,
        referenced_works=referenced,
        related_works=related,
        field_name=field_name,
        raw=MappingProxyType(dict(raw)),
    )


def _strip_openalex_id_url(value: Any) -> str | None:
    """``https://openalex.org/W123`` → ``W123``. Anything else → ``None``.

    OpenAlex always returns its own ids as full URLs; this canonicalizes
    on the bare form so callers don't have to slice URLs themselves.
    """
    if not isinstance(value, str) or not value:
        return None
    marker = "openalex.org/"
    if marker not in value:
        return None
    bare = value.split(marker, 1)[1]
    return bare or None


def _strip_doi_url(value: Any) -> str | None:
    """``https://doi.org/10.x/y`` → ``10.x/y``. Lower-cases per spec.

    DOIs are case-insensitive per the spec; lower-casing here means
    downstream dedup logic that hashes the DOI doesn't miss matches
    when OpenAlex returns a mixed-case form.
    """
    if not isinstance(value, str) or not value:
        return None
    stripped = value
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if stripped.startswith(prefix):
            stripped = stripped.removeprefix(prefix)
            break
    return stripped.lower() or None


def _reconstruct_abstract(inverted: Any) -> str:
    """Flatten OpenAlex's ``{word: [positions]}`` index back to a string.

    Each word can appear at multiple positions; we sort by position and
    join with spaces. Quirky but preserves word order without storing
    the abstract verbatim (OpenAlex's copyright workaround).
    """
    if not inverted or not isinstance(inverted, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, posns in inverted.items():
        if not isinstance(posns, list):
            continue
        for pos in posns:
            if isinstance(pos, int):
                positions.append((pos, str(word)))
    positions.sort()
    return " ".join(word for _, word in positions)


def _parse_publication_date(date_str: Any, year: Any) -> date | None:
    if isinstance(date_str, str) and date_str:
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            pass
    if isinstance(year, int):
        try:
            return date(year, 1, 1)
        except ValueError:
            return None
    return None


def _extract_venue(raw: Mapping[str, Any]) -> str | None:
    primary = raw.get("primary_location") or {}
    source = primary.get("source") if isinstance(primary, dict) else None
    if isinstance(source, dict):
        name = (source.get("display_name") or "").strip()
        if name:
            return name
    return None


def _extract_pdf_url(raw: Mapping[str, Any]) -> str | None:
    """Prefer ``best_oa_location.pdf_url``; fall back to ``open_access.oa_url``.

    Both fields can be present; best_oa_location is a curated open-access
    landing, oa_url is whatever URL OpenAlex has cached. Either is more
    likely to actually serve a PDF than primary_location.pdf_url, which
    is often null even when an OA copy exists elsewhere.
    """
    best = raw.get("best_oa_location") or {}
    if isinstance(best, dict):
        url = best.get("pdf_url")
        if isinstance(url, str) and url:
            return url
    oa = raw.get("open_access") or {}
    if isinstance(oa, dict):
        url = oa.get("oa_url")
        if isinstance(url, str) and url:
            return url
    return None


def _extract_authors(raw: Mapping[str, Any]) -> tuple[Author, ...]:
    authors: list[Author] = []
    for entry in raw.get("authorships") or []:
        if not isinstance(entry, dict):
            continue
        author = entry.get("author")
        if not isinstance(author, dict):
            continue
        name = (author.get("display_name") or "").strip()
        if not name:
            continue
        orcid = author.get("orcid")
        authors.append(
            Author(name=name, orcid=orcid if isinstance(orcid, str) else None)
        )
    return tuple(authors)


def _extract_id_list(value: Any) -> tuple[str, ...]:
    """Pull bare OpenAlex IDs out of a list of work URLs.

    OpenAlex returns ``referenced_works`` / ``related_works`` as lists
    of full work URLs (``https://openalex.org/W123``). This strips each
    to its bare form and drops anything that doesn't parse.
    """
    if not isinstance(value, list):
        return ()
    return tuple(
        oa_id
        for item in value
        if (oa_id := _strip_openalex_id_url(item)) is not None
    )


def _extract_field_name(raw: Mapping[str, Any]) -> str | None:
    topic = raw.get("primary_topic")
    if not isinstance(topic, dict):
        return None
    field_block = topic.get("field")
    if not isinstance(field_block, dict):
        return None
    name = field_block.get("display_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None
