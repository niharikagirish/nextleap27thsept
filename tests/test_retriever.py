"""Retrieval tests (Phase 4) — no index, no model, no network.

``search`` accepts an injected ``collection`` and its query embedder is a single
module-level function, so the whole Stage 4 can be exercised with a fake. That
injection is the point of the design: C10 asks for each stage to be testable in
isolation, and a retriever that can only be tested against a real ChromaDB on
disk fails that.

The properties asserted are the ones that actually change answers:

* **score = 1.0 - distance.** A sign error here inverts the ranking and still
  produces plausible-looking output, so it is asserted directly.
* **the floor is a floor.** Nothing under 0.35 is returned, ever, and an
  all-below-floor result is ``[]`` rather than a padded list.
* **fetch wide, trim narrow.** 12 in, 6 out; fetching 6 and filtering would
  starve multi-part questions.
* **dedupe keeps the better-scoring copy.**
"""

from __future__ import annotations

import pytest

from app.pipeline import retriever
from app.pipeline.store import HEADING_DELIM, chunk_id_for

URL_A = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
URL_B = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"

FAKE_VECTOR = [0.0] * 384


class FakeCollection:
    """Minimal stand-in for a ChromaDB collection."""

    def __init__(self, rows: list[tuple[str, dict, float]]) -> None:
        # rows: (document, metadata, distance)
        self.rows = rows
        self.calls: list[dict] = []

    def query(self, query_embeddings, n_results=10, where=None, include=None):
        self.calls.append({"n_results": n_results, "where": where,
                           "include": include, "dim": len(query_embeddings[0])})
        rows = self.rows[:n_results]
        if where and "scheme_slug" in where:
            rows = [r for r in self.rows
                    if r[1].get("scheme_slug") == where["scheme_slug"]]
        return {
            "documents": [[r[0] for r in rows]],
            "metadatas": [[r[1] for r in rows]],
            "distances": [[r[2] for r in rows]],
        }


def _meta(source_id: str = "hdfc-large-cap", index: int = 0,
          trail: tuple[str, ...] = ("HDFC Large Cap Fund", "Exit load"),
          url: str = URL_A) -> dict:
    return {
        "source_id": source_id, "scheme_name": "HDFC Large Cap Fund",
        "scheme_slug": source_id, "category": "Large Cap", "plan": "Direct-Growth",
        "url": url, "page_title": "Large Cap",
        "heading_trail": HEADING_DELIM.join(trail), "chunk_index": index,
        "fetched_at": "2026-09-27T10:00:00Z", "char_len": 42, "word_count": 8,
        "content_hash": "abc123",
    }


def _row(text: str, distance: float, source_id: str = "hdfc-large-cap",
         index: int = 0, trail: tuple[str, ...] = ("HDFC Large Cap Fund", "Exit load"),
         url: str = URL_A) -> tuple[str, dict, float]:
    return (text, _meta(source_id, index, trail, url), distance)


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    """Never load SentenceTransformer in this file."""
    monkeypatch.setattr(retriever, "embed_query", lambda text: list(FAKE_VECTOR))


# --------------------------------------------------------------------------- #
# Score conversion
# --------------------------------------------------------------------------- #


def test_score_is_one_minus_distance() -> None:
    collection = FakeCollection([
        _row("Exit load is 1%.", 0.05),      # score 0.95
        _row("Expense ratio is 0.87%.", 0.20),  # score 0.80
    ])
    results = retriever.search("exit load?", collection=collection)
    assert [rc.score for rc in results] == [0.95, 0.8]


def test_results_are_sorted_by_descending_score() -> None:
    collection = FakeCollection([
        _row("c", 0.60),   # 0.40
        _row("a", 0.10),   # 0.90
        _row("b", 0.30),   # 0.70
    ])
    results = retriever.search("q?", collection=collection)
    assert [rc.score for rc in results] == [0.90, 0.70, 0.40]


def test_rank_is_one_based_and_dense() -> None:
    collection = FakeCollection([_row(f"chunk {i}", 0.1 + i * 0.05) for i in range(4)])
    results = retriever.search("q?", collection=collection)
    assert [rc.rank for rc in results] == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# The score floor
# --------------------------------------------------------------------------- #


def test_below_floor_chunks_are_dropped() -> None:
    collection = FakeCollection([
        _row("strong", 0.10),   # 0.90 kept
        _row("weak", 0.70),     # 0.30 dropped
    ])
    results = retriever.search("q?", collection=collection)
    assert [rc.chunk.text for rc in results] == ["strong"]


def test_nothing_above_floor_returns_empty_list() -> None:
    collection = FakeCollection([_row("a", 0.80), _row("b", 0.90)])
    assert retriever.search("q?", collection=collection) == []


def test_floor_is_configured_at_035() -> None:
    collection = FakeCollection([_row("exactly at floor", 0.65)])  # score 0.35
    results = retriever.search("q?", collection=collection)
    assert len(results) == 1, "0.35 is inclusive: score >= floor"


def test_just_below_floor_is_excluded() -> None:
    collection = FakeCollection([_row("just below", 0.6501)])  # score 0.3499
    assert retriever.search("q?", collection=collection) == []


# --------------------------------------------------------------------------- #
# Wide fetch, narrow trim
# --------------------------------------------------------------------------- #


def test_fetches_wide_by_default() -> None:
    collection = FakeCollection([])
    retriever.search("q?", collection=collection)
    assert collection.calls[0]["n_results"] == 12


def test_trims_to_top_k() -> None:
    rows = [_row(f"chunk {i}", 0.05 + i * 0.01) for i in range(10)]
    results = retriever.search("q?", collection=FakeCollection(rows))
    assert len(results) == 6
    assert results[0].chunk.text == "chunk 0", "the best chunk must survive the trim"


def test_n_results_and_top_k_are_overridable() -> None:
    collection = FakeCollection([_row(f"c{i}", 0.1 + i * 0.01) for i in range(8)])
    results = retriever.search("q?", n_results=8, top_k=2, collection=collection)
    assert collection.calls[0]["n_results"] == 8
    assert len(results) == 2


def test_fewer_candidates_than_top_k_is_fine() -> None:
    collection = FakeCollection([_row("only one", 0.1)])
    assert len(retriever.search("q?", collection=collection)) == 1


# --------------------------------------------------------------------------- #
# Dedupe
# --------------------------------------------------------------------------- #


def test_near_duplicate_chunks_are_deduped() -> None:
    collection = FakeCollection([
        _row("Exit load is 1% within 1 year.", 0.10, index=0),
        _row("Exit load is 1% within 1 year.", 0.12, index=1),
    ])
    results = retriever.search("q?", collection=collection)
    assert len(results) == 1


def test_dedupe_keeps_the_higher_scoring_copy() -> None:
    collection = FakeCollection([
        _row("Exit load is 1% within 1 year.", 0.20, index=0),  # 0.80
        _row("Exit load is 1% within 1 year.", 0.10, index=1),  # 0.90
    ])
    results = retriever.search("q?", collection=collection)
    assert len(results) == 1
    assert results[0].score == 0.90


def test_distinct_chunks_are_not_deduped() -> None:
    collection = FakeCollection([
        _row("Exit load is 1% within 1 year.", 0.10, index=0),
        _row("The expense ratio is 0.87% per annum.", 0.15, index=1),
    ])
    assert len(retriever.search("q?", collection=FakeCollection(rows=collection.rows))) == 2


def test_jaccard_basics() -> None:
    assert retriever.jaccard("a b c", "a b c") == 1.0
    assert retriever.jaccard("a b c", "x y z") == 0.0
    assert 0.0 < retriever.jaccard("a b c", "a b d") < 1.0


def test_jaccard_ignores_case_and_punctuation() -> None:
    assert retriever.jaccard("Exit load, 1%.", "exit load 1%") == 1.0


def test_jaccard_of_two_empty_strings_is_one() -> None:
    assert retriever.jaccard("", "") == 1.0


# --------------------------------------------------------------------------- #
# Scheme filter
# --------------------------------------------------------------------------- #


def test_scheme_slug_is_passed_through_as_a_metadata_filter() -> None:
    collection = FakeCollection([_row("x", 0.1)])
    retriever.search("q?", scheme_slug="hdfc-small-cap", collection=collection)
    assert collection.calls[0]["where"] == {"scheme_slug": "hdfc-small-cap"}


def test_no_scheme_means_no_filter() -> None:
    collection = FakeCollection([_row("x", 0.1)])
    retriever.search("q?", collection=collection)
    assert collection.calls[0]["where"] is None


def test_detect_scheme_slug_from_an_explicit_mention() -> None:
    assert retriever.detect_scheme_slug(
        "What is the expense ratio of HDFC Small Cap Fund?") == "hdfc-small-cap"
    assert retriever.detect_scheme_slug(
        "Tell me about HDFC ELSS Tax Saver Fund") == "hdfc-elss"


def test_bare_fund_word_is_not_a_scheme_signal() -> None:
    """'fund' is in every scheme name and must not select one arbitrarily."""
    assert retriever.detect_scheme_slug("What is the expense ratio?") is None


def test_unknown_scheme_mention_yields_none() -> None:
    assert retriever.detect_scheme_slug("What about the HDFC Mid Cap fund?") is None


# --------------------------------------------------------------------------- #
# Chunk reconstruction
# --------------------------------------------------------------------------- #


def test_heading_trail_round_trips_through_the_delimiter() -> None:
    collection = FakeCollection([
        _row("x", 0.1, index=0, trail=("HDFC Small Cap Fund", "Fees", "Expense ratio")),
    ])
    result = retriever.search("q?", collection=collection)[0]
    assert result.chunk.heading_trail == [
        "HDFC Small Cap Fund", "Fees", "Expense ratio"
    ]


def test_text_is_the_raw_document_not_the_embed_text() -> None:
    collection = FakeCollection([_row("Exit load is 1%.", 0.1)])
    result = retriever.search("q?", collection=collection)[0]
    assert result.chunk.text == "Exit load is 1%."
    assert result.chunk.heading_trail  # rebuilt into embed_text, not text


def test_chunk_id_is_the_deterministic_formula() -> None:
    collection = FakeCollection([_row("x", 0.1, source_id="hdfc-elss", index=7)])
    result = retriever.search("q?", collection=collection)[0]
    assert result.chunk.chunk_id == chunk_id_for("hdfc-elss", 7)


def test_provenance_fields_survive() -> None:
    collection = FakeCollection([_row("x", 0.1, source_id="hdfc-elss", url=URL_B)])
    result = retriever.search("q?", collection=collection)[0]
    assert result.chunk.source_id == "hdfc-elss"
    assert result.chunk.url == URL_B
    assert result.chunk.fetched_at == "2026-09-27T10:00:00Z"
    assert result.chunk.scheme_slug == "hdfc-elss"


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


def test_empty_collection_yields_empty_results() -> None:
    assert retriever.search("q?", collection=FakeCollection([])) == []


def test_row_with_no_metadata_does_not_crash() -> None:
    class Ragged(FakeCollection):
        def query(self, query_embeddings, n_results=10, where=None, include=None):
            return {"documents": [["orphan text"]], "metadatas": [[]],
                    "distances": [[0.10]]}

    results = retriever.search("q?", collection=Ragged([]))
    assert len(results) == 1
    assert results[0].chunk.text == "orphan text"
    assert results[0].chunk.url == ""
