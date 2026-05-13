"""Parsing tests: ``parse_work``, abstract reconstruction, URL strippers."""

from __future__ import annotations

from datetime import date

import pytest

from oalex.types import (
    Work,
    _reconstruct_abstract,
    _strip_doi_url,
    _strip_openalex_id_url,
    parse_work,
)

# Trimmed-down real OpenAlex response captured from a live API call.
_VASWANI_WORK: dict = {
    "id": "https://openalex.org/W2626778328",
    "doi": "https://doi.org/10.65215/2q58a426",
    "title": "Attention Is All You Need",
    "publication_year": 2017,
    "publication_date": "2017-06-12",
    "cited_by_count": 6536,
    "is_retracted": False,
    "primary_location": {
        "id": "doi:10.65215/2q58a426",
        "pdf_url": None,
        "source": {"display_name": "NeurIPS", "type": "conference"},
    },
    "best_oa_location": {"pdf_url": "https://example.org/vaswani.pdf"},
    "open_access": {"is_oa": True, "oa_url": "https://example.org/vaswani.pdf"},
    "authorships": [
        {
            "author_position": "first",
            "author": {
                "display_name": "Ashish Vaswani",
                "orcid": "https://orcid.org/0000-0002-7794-2085",
            },
        },
        {
            "author_position": "middle",
            "author": {"display_name": "Noam Shazeer", "orcid": None},
        },
    ],
    "abstract_inverted_index": {
        "The": [0],
        "dominant": [1],
        "sequence": [2],
        "transduction": [3],
        "models": [4],
        "are": [5],
        "RNNs.": [6],
    },
}


# ---- URL strippers ----


def test_strip_openalex_id_url_extracts_w_id() -> None:
    assert _strip_openalex_id_url("https://openalex.org/W2626778328") == "W2626778328"


def test_strip_openalex_id_url_returns_none_for_garbage() -> None:
    assert _strip_openalex_id_url("not-an-openalex-url") is None
    assert _strip_openalex_id_url("") is None
    assert _strip_openalex_id_url(None) is None


def test_strip_doi_url_extracts_bare_doi() -> None:
    assert _strip_doi_url("https://doi.org/10.1038/nature12373") == "10.1038/nature12373"
    assert _strip_doi_url("http://doi.org/10.x/y") == "10.x/y"


def test_strip_doi_url_lowercases() -> None:
    """DOIs are case-insensitive per spec; canonical form is lower."""
    assert _strip_doi_url("https://doi.org/10.1038/NATURE12373") == "10.1038/nature12373"


def test_strip_doi_url_returns_none_for_empty() -> None:
    assert _strip_doi_url(None) is None
    assert _strip_doi_url("") is None


# ---- abstract reconstruction ----


def test_reconstruct_abstract_orders_words_by_position() -> None:
    inverted = {"The": [0, 4], "quick": [1], "brown": [2], "fox": [3], "rest": [5]}
    assert _reconstruct_abstract(inverted) == "The quick brown fox The rest"


def test_reconstruct_abstract_handles_none() -> None:
    assert _reconstruct_abstract(None) == ""


def test_reconstruct_abstract_handles_empty_dict() -> None:
    assert _reconstruct_abstract({}) == ""


def test_reconstruct_abstract_skips_non_int_positions() -> None:
    """Defensive: malformed values shouldn't crash the parser."""
    inverted = {"hello": [0], "broken": ["not-an-int", None]}
    assert _reconstruct_abstract(inverted) == "hello"


# ---- parse_work ----


def test_parse_work_extracts_canonical_fields() -> None:
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    assert w.id == "W2626778328"
    assert w.title == "Attention Is All You Need"
    assert w.doi == "10.65215/2q58a426"
    assert w.published == date(2017, 6, 12)
    assert w.citation_count == 6536


def test_parse_work_reconstructs_abstract() -> None:
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    assert w.abstract == "The dominant sequence transduction models are RNNs."


def test_parse_work_extracts_authors_with_orcids() -> None:
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    assert len(w.authors) == 2
    assert w.authors[0].name == "Ashish Vaswani"
    assert w.authors[0].orcid == "https://orcid.org/0000-0002-7794-2085"
    assert w.authors[1].orcid is None


def test_parse_work_extracts_venue_from_primary_location_source() -> None:
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    assert w.venue == "NeurIPS"


def test_parse_work_uses_best_oa_pdf_url() -> None:
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    assert w.pdf_url == "https://example.org/vaswani.pdf"


def test_parse_work_falls_back_to_year_when_date_missing() -> None:
    raw = dict(_VASWANI_WORK)
    raw["publication_date"] = None
    w = parse_work(raw)
    assert w is not None
    assert w.published == date(2017, 1, 1)


def test_parse_work_returns_none_when_id_missing() -> None:
    raw = dict(_VASWANI_WORK)
    raw["id"] = None
    assert parse_work(raw) is None


def test_parse_work_returns_none_when_id_unparseable() -> None:
    raw = dict(_VASWANI_WORK)
    raw["id"] = "not-an-openalex-url"
    assert parse_work(raw) is None


def test_parse_work_returns_none_for_empty_payload() -> None:
    assert parse_work({}) is None
    assert parse_work(None) is None


def test_parse_work_handles_missing_doi() -> None:
    raw = dict(_VASWANI_WORK)
    raw["doi"] = None
    w = parse_work(raw)
    assert w is not None
    assert w.doi is None


def test_parse_work_handles_null_primary_location_source() -> None:
    raw = dict(_VASWANI_WORK)
    raw["primary_location"] = {"source": None, "pdf_url": None}
    w = parse_work(raw)
    assert w is not None
    assert w.venue is None


def test_parse_work_handles_null_open_access_block() -> None:
    raw = dict(_VASWANI_WORK)
    raw["best_oa_location"] = None
    raw["open_access"] = None
    w = parse_work(raw)
    assert w is not None
    assert w.pdf_url is None


def test_parse_work_handles_no_abstract() -> None:
    raw = dict(_VASWANI_WORK)
    raw["abstract_inverted_index"] = None
    w = parse_work(raw)
    assert w is not None
    assert w.abstract == ""


def test_parse_work_handles_missing_citation_count() -> None:
    raw = dict(_VASWANI_WORK)
    raw["cited_by_count"] = None
    w = parse_work(raw)
    assert w is not None
    assert w.citation_count is None


def test_parse_work_extracts_referenced_works_as_bare_ids() -> None:
    raw = dict(_VASWANI_WORK)
    raw["referenced_works"] = [
        "https://openalex.org/W001",
        "https://openalex.org/W002",
    ]
    w = parse_work(raw)
    assert w is not None
    assert w.referenced_works == ("W001", "W002")


def test_parse_work_extracts_related_works_as_bare_ids() -> None:
    raw = dict(_VASWANI_WORK)
    raw["related_works"] = ["https://openalex.org/W777"]
    w = parse_work(raw)
    assert w is not None
    assert w.related_works == ("W777",)


def test_parse_work_drops_malformed_referenced_entries() -> None:
    raw = dict(_VASWANI_WORK)
    raw["referenced_works"] = [
        "https://openalex.org/W_GOOD",
        None,
        42,
        "not-an-openalex-url",
    ]
    w = parse_work(raw)
    assert w is not None
    assert w.referenced_works == ("W_GOOD",)


def test_parse_work_referenced_works_empty_when_field_absent() -> None:
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    assert w.referenced_works == ()
    assert w.related_works == ()


def test_parse_work_extracts_field_name_from_primary_topic() -> None:
    raw = dict(_VASWANI_WORK)
    raw["primary_topic"] = {
        "id": "https://openalex.org/T11636",
        "display_name": "Sequence-to-sequence translation",
        "field": {"display_name": "Computer Science"},
    }
    w = parse_work(raw)
    assert w is not None
    assert w.field_name == "Computer Science"


def test_parse_work_field_name_none_when_topic_block_missing() -> None:
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    assert w.field_name is None


def test_parse_work_raw_payload_is_accessible() -> None:
    """``Work.raw`` exposes fields the typed surface doesn't cover."""
    raw = dict(_VASWANI_WORK)
    raw["concepts"] = [{"display_name": "Transformer"}]
    w = parse_work(raw)
    assert w is not None
    assert w.raw.get("concepts") == [{"display_name": "Transformer"}]


def test_parse_work_raw_payload_is_read_only() -> None:
    """``raw`` is a MappingProxyType so callers can't accidentally mutate it."""
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    with pytest.raises(TypeError):
        w.raw["foo"] = "bar"  # type: ignore[index]


def test_work_is_frozen() -> None:
    """Work is immutable — frozen dataclass."""
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    with pytest.raises(AttributeError):
        w.title = "Different"  # type: ignore[misc]


def test_work_is_not_hashable_because_raw_contains_dicts() -> None:
    """Work is frozen but not hashable: ``raw`` carries nested dicts/lists
    from the OpenAlex payload. Callers needing a hashable identity should
    key on ``Work.id`` (a string) rather than the Work object itself."""
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    with pytest.raises(TypeError, match="unhashable"):
        hash(w)


def test_work_field_name_is_optional() -> None:
    """field_name is None when primary_topic.field is missing."""
    raw = dict(_VASWANI_WORK)
    raw["primary_topic"] = {"id": "T1", "field": None}
    w = parse_work(raw)
    assert w is not None
    assert w.field_name is None


def test_authors_are_a_tuple_not_a_list() -> None:
    """Tuple so Work stays hashable (lists aren't hashable)."""
    w = parse_work(_VASWANI_WORK)
    assert w is not None
    assert isinstance(w.authors, tuple)


def test_parse_work_skips_authorship_entries_without_display_name() -> None:
    raw = dict(_VASWANI_WORK)
    raw["authorships"] = [
        {"author": {"display_name": "Valid Author", "orcid": None}},
        {"author": {"display_name": "", "orcid": None}},
        {"author": None},
        {"not_author_key": "garbage"},
        "not_a_dict",
    ]
    w = parse_work(raw)
    assert w is not None
    assert len(w.authors) == 1
    assert w.authors[0].name == "Valid Author"


def test_work_construction_directly_is_supported() -> None:
    """Smoke test that the dataclass works without going through parse_work."""
    w = Work(
        id="W1",
        title="Test",
        abstract="abstract",
        authors=(),
    )
    assert w.id == "W1"
    assert w.referenced_works == ()
