"""Phase 0 gate: the project imports, the config loads, the corpus is 5 pages."""

from __future__ import annotations

from app.config import SETTINGS
from app.sources import (
    ALLOWLIST,
    DENIED_HOST_SUFFIXES,
    EXAMPLE_QUESTIONS,
    is_allowed_url,
    is_denied_host,
    spec_for,
)


def test_allowlist_has_exactly_five_sources():
    assert len(ALLOWLIST) == 5


def test_allowlist_urls_are_unique_and_on_groww():
    urls = [s.url for s in ALLOWLIST]
    assert len(set(urls)) == 5
    assert all(u.startswith("https://groww.in/mutual-funds/") for u in urls)


def test_allowlist_covers_the_three_required_categories():
    categories = {s.category for s in ALLOWLIST}
    assert "Large Cap" in categories
    assert "Flexi Cap" in categories
    assert "ELSS" in categories


def test_all_plans_are_direct_growth():
    assert all(s.plan == "Direct-Growth" for s in ALLOWLIST)


def test_is_allowed_url_only_matches_exact_allowlist_entries():
    assert is_allowed_url(ALLOWLIST[0].url)
    assert is_allowed_url(ALLOWLIST[0].url + "/")          # trailing slash tolerated
    assert not is_allowed_url("https://groww.in/mutual-funds/hdfc-parag-parag-flexi-cap")
    assert not is_allowed_url("https://example.com/hdfc-large-cap")
    assert not is_allowed_url("https://groww.in/blog/hdfc-large-cap")


def test_denied_hosts_are_rejected():
    assert is_denied_host("https://medium.com/some-mf-blog")
    assert is_denied_host("https://www.reddit.com/r/IndianStockMarket")
    assert is_denied_host("https://moneycontrol.com/news/fund")
    assert not is_denied_host("https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth")


def test_spec_for_round_trips_and_rejects_unknown():
    assert spec_for("hdfc-elss").category == "ELSS"
    try:
        spec_for("parag-parag")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("spec_for should reject an unknown source_id")


def test_example_questions_present():
    assert len(EXAMPLE_QUESTIONS) == 3
    assert all(q.endswith("?") for q in EXAMPLE_QUESTIONS)


def test_config_loads_key_values():
    assert SETTINGS.get("retrieval.score_floor") == 0.35
    assert SETTINGS.get("embedding.model_id") == "sentence-transformers/all-MiniLM-L6-v2"
    assert SETTINGS.get("corpus.collection_name") == "hdfc_faq"
    assert SETTINGS.get("generation.temperature") == 0.0
    assert SETTINGS.get("chunking.max_wordpieces") == 256


def test_guardrails_are_enabled_by_default():
    assert SETTINGS.get("guards.pii_enabled") is True
    assert SETTINGS.get("guards.performance_guard") is True
    assert SETTINGS.get("guards.intent_enabled") is True


def test_models_import():
    from app.models import Answer, Chunk, RawDoc, RetrievedChunk, SourceSpec

    assert SourceSpec("a", "b", "c", "d", "e").source_id == "a"
    assert RawDoc.__dataclass_fields__.keys() >= {"text", "url", "fetched_at"}
    assert "sources" in Answer.__dataclass_fields__
    assert RetrievedChunk.__dataclass_fields__.keys() >= {"chunk", "score", "rank"}


def test_denied_suffixes_are_lowercase():
    assert all(s == s.lower() for s in DENIED_HOST_SUFFIXES)
