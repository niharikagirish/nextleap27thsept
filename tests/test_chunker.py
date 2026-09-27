"""Phase 2 tests â€” chunking strategies, the word-piece invariant, and the gold set.

The invariant test is the important one. architecture.md Â§4.2 calls the 256
word-piece limit "a model property, not a tuning preference", so it is asserted
here against the real tokenizer rather than trusted as a comment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import ROOT, SETTINGS
from app.experiment import (
    GoldQuestion,
    Result,
    build_config,
    load_golden,
    pick_winner,
    render_report,
    write_back_config,
)
from app.models import RawDoc
from app.pipeline import chunker as chunker_mod

MAX_WORDPIECES = int(SETTINGS.get("chunking.max_wordpieces", 256))
TARGET_WORDPIECES = int(SETTINGS.get("chunking.target_wordpieces", 200))

# Mirrors the real Stage 1 output shape: a markdown-flavoured document with an
# H1, a metadata line, an H2 facts block and H3 FAQ pairs.
SAMPLE_DOC = RawDoc(
    source_id="hdfc-sample",
    scheme_name="HDFC Sample Fund",
    category="Large Cap",
    plan="Direct-Growth",
    url="https://groww.in/mutual-funds/hdfc-sample-fund-direct-growth",
    page_title="HDFC Sample Fund Direct Growth - Groww",
    fetched_at="2026-09-27T00:00:00+00:00",
    http_status=200,
    content_hash="deadbeef",
    text="\n\n".join([
        "# HDFC Sample Fund",
        "Source page: https://groww.in/mutual-funds/hdfc-sample-fund-direct-growth",
        "Page title: HDFC Sample Fund Direct Growth - Groww",
        "## Scheme facts",
        "\n\n".join([
            "Expense ratio of HDFC Sample Fund Direct Growth is 1.03% "
            "(base expense ratio 0.84%), charged annually on the assets of the "
            "scheme.",
            "Exit load of HDFC Sample Fund Direct Growth: Exit load of 1% if "
            "redeemed within 1 year.",
            "Minimum SIP amount for HDFC Sample Fund Direct Growth is Rs 100 "
            "per month, with a minimum additional SIP of Rs 100.",
            "Lock-in period for HDFC Sample Fund Direct Growth: none.",
            "Riskometer for HDFC Sample Fund Direct Growth: Moderately High.",
            "Benchmark for HDFC Sample Fund Direct Growth: NIFTY 100 TRI "
            "(NIFTY 100 Total Return Index).",
        ]),
        "## Frequently asked questions",
        "### How to invest in HDFC Sample Fund Direct Growth?",
        "You can invest through Groww with a lump sum or a monthly SIP.",
        "### How to Redeem HDFC Sample Fund Direct Growth?",
        "Redeem from the Groww app under Portfolio.",
    ]),
)

@pytest.fixture(scope="module")
def tokenizer():
    """The real MiniLM tokenizer, or a skip.

    Probing at import time would raise during collection and take the whole file
    down, so availability is resolved lazily in a fixture instead.
    """
    try:
        return chunker_mod.get_tokenizer()
    except Exception as exc:  # noqa: BLE001 - any load failure means "skip"
        pytest.skip(f"MiniLM tokenizer unavailable: {exc}")


needs_tokenizer = pytest.mark.usefixtures("tokenizer")


# --------------------------------------------------------------------------- #
# Document structure
# --------------------------------------------------------------------------- #


def test_parse_sections_keeps_heading_hierarchy():
    sections = chunker_mod.parse_sections(SAMPLE_DOC.text)
    trails = [s.trail for s in sections]
    # The H1's body is nothing but the two metadata lines, so it is dropped
    # rather than emitted as a provenance-only chunk - but the H1 still leads
    # every surviving trail, so the hierarchy is intact.
    assert ("HDFC Sample Fund",) not in trails
    assert all(trail[0] == "HDFC Sample Fund" for trail in trails)
    assert ("HDFC Sample Fund", "Scheme facts") in trails
    assert any(t[-1].startswith("How to invest") for t in trails)
    # Metadata lines are not body text and must not become their own section.
    assert all("Source page" not in s.text for s in sections)


def test_split_sentences_protects_decimals_and_currency():
    text = "The expense ratio is 1.03%. Minimum SIP is Rs 100. Exit load is Nil."
    sentences = chunker_mod.split_sentences(text)
    assert "1.03%." not in sentences, "must not split inside a decimal figure"
    assert any("1.03%" in s for s in sentences)
    assert any("Rs 100" in s for s in sentences)


def test_embed_text_carries_heading_trail_but_text_does_not():
    trail = ["HDFC Sample Fund", "Scheme facts"]
    body = "Exit load of HDFC Sample Fund Direct Growth: Nil."
    embed = chunker_mod.build_embed_text(trail, body)
    assert embed == f"{' > '.join(trail)}\n\n{body}"
    assert body in embed


# --------------------------------------------------------------------------- #
# The invariant (architecture.md Â§4.2)
# --------------------------------------------------------------------------- #


@needs_tokenizer
@pytest.mark.parametrize(
    "strategy", ["recursive", "heading_bound"]
)
def test_every_chunk_fits_the_wordpiece_cap(strategy):
    """The load-bearing test: no chunk may exceed MiniLM's silent-truncation limit."""
    engine = chunker_mod.get_chunker(strategy, 1200, 180)
    report = chunker_mod.run(SAMPLE_DOC, engine)
    assert report.kept
    for chunk in report.kept:
        assert chunker_mod.count_wordpieces(chunk.text) <= MAX_WORDPIECES, (
            f"{chunk.chunk_id} is over budget"
        )


@needs_tokenizer
def test_cap_violation_is_repaired_not_just_reported():
    """An over-budget chunk must actually be re-split, not merely counted.

    The size is huge, but ``SAMPLE_DOC`` is only ~240 word-pieces - comfortably
    under the 256 cap - so splitting it with a huge target produces chunks that
    are *already* compliant and the repair path never runs. The input is padded
    until it is genuinely over budget, otherwise this test would pass for the
    wrong reason and the repair code could break unnoticed.
    """
    engine = chunker_mod.RecursiveChunker(100_000, 0)
    oversized = engine.split(SAMPLE_DOC)
    assert oversized, "precondition: the oversized split produced chunks"

    # Pad until a chunk is over the cap, keeping the same shape.
    sentence = ("The exit load of HDFC Sample Fund Direct Growth is 1% if the "
                "units are redeemed within 1 year of the allotment date. ")
    padded = SAMPLE_DOC.__class__(
        **{**SAMPLE_DOC.__dict__,
           "text": SAMPLE_DOC.text + "\n\n## Padded\n\n" + sentence * 40}
    )
    engine = chunker_mod.RecursiveChunker(100_000, 0)
    oversized = engine.split(padded)
    assert oversized, "precondition: the padded split produced chunks"
    assert any(chunker_mod.count_wordpieces(c.text) > MAX_WORDPIECES
               for c in oversized), (
        "precondition: at least one chunk is over budget"
    )

    report = chunker_mod.enforce_token_budget(
        oversized, MAX_WORDPIECES, TARGET_WORDPIECES
    )
    assert report.rescued > 0, "over-budget chunks should have been re-split"
    assert report.violations == 0
    for chunk in report.kept:
        assert chunker_mod.count_wordpieces(chunk.text) <= MAX_WORDPIECES


@needs_tokenizer
def test_unsplittable_fragment_is_dropped_and_counted():
    """A single token longer than the model limit must be dropped, not embedded."""
    hostile = "# HDFC Sample Fund\n\n## Scheme facts\n\n" + ("x" * 5_000)
    engine = chunker_mod.RecursiveChunker(5_000, 0)
    pieces = engine.split(RawDoc(**{**SAMPLE_DOC.__dict__, "text": hostile}))

    report = chunker_mod.enforce_token_budget(
        pieces, MAX_WORDPIECES, TARGET_WORDPIECES
    )
    # Word-splitting saves this one in practice; either outcome is correct, but
    # nothing over budget may survive.
    for chunk in report.kept:
        assert chunker_mod.count_wordpieces(chunk.text) <= MAX_WORDPIECES


@needs_tokenizer
def test_chunk_index_and_id_are_stable_per_document():
    engine = chunker_mod.get_chunker("heading_bound", 800, 100)
    first = chunker_mod.run(SAMPLE_DOC, engine).kept
    second = chunker_mod.run(SAMPLE_DOC, engine).kept
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.chunk_index for c in first] == list(range(len(first)))
    assert len({c.chunk_id for c in first}) == len(first), "chunk_id must be unique"


@needs_tokenizer
def test_chunks_carry_full_provenance():
    for chunk in chunker_mod.run(SAMPLE_DOC, chunker_mod.get_chunker()).kept:
        assert chunk.source_id == SAMPLE_DOC.source_id
        assert chunk.url == SAMPLE_DOC.url
        assert chunk.fetched_at == SAMPLE_DOC.fetched_at
        assert chunk.scheme_name == SAMPLE_DOC.scheme_name
        assert chunk.scheme_slug == SAMPLE_DOC.source_id
        assert chunk.content_hash
        assert chunk.char_len == len(chunk.text)
        assert chunk.word_count == len(chunk.text.split())


@needs_tokenizer
def test_heading_bound_keeps_facts_under_one_heading_trail():
    """The mechanism behind the expected winner, asserted rather than assumed.

    A 400-char recursive split fragments the facts block across several chunks
    that each claim the same breadcrumb; heading_bound instead emits the whole
    facts section as one chunk, so "exit load" and "lock-in" always travel
    together with a truthful heading trail.
    """
    heading_chunks = chunker_mod.run(
        SAMPLE_DOC, chunker_mod.HeadingBoundChunker(800, 100)
    ).kept
    facts_chunks = [
        c for c in heading_chunks if c.heading_trail[-1] == "Scheme facts"
    ]
    assert len(facts_chunks) == 1, "the 6-fact block fits in one 800-char chunk"
    facts_text = facts_chunks[0].text
    for span in (
        "Expense ratio of HDFC Sample Fund",
        "Exit load of HDFC Sample Fund",
        "Lock-in period for HDFC Sample Fund",
    ):
        assert span in facts_text

    recursive_chunks = chunker_mod.run(
        SAMPLE_DOC, chunker_mod.RecursiveChunker(400, 40)
    ).kept
    assert len(recursive_chunks) > len(heading_chunks), (
        "the smaller recursive size must produce strictly more chunks"
    )
    # Every heading_bound chunk's trail must actually describe its content.
    for chunk in heading_chunks:
        assert chunk.heading_trail[0] == SAMPLE_DOC.scheme_name
        assert SAMPLE_DOC.page_title in chunk.heading_trail
        # The prefix must not leak into the stored text shown to the user.
        assert not chunk.text.startswith("HDFC Sample Fund >")


@needs_tokenizer
def test_summary_reports_the_documented_keys():
    chunks = chunker_mod.run(SAMPLE_DOC, chunker_mod.get_chunker()).kept
    stats = chunker_mod.summarise(chunks, violations=0)
    for key in ("out", "mean_chars", "p95_tokens", "violations"):
        assert key in stats, f"stage summary must report {key} (implementation.md 2.A.8)"
    assert stats["out"] == len(chunks)
    assert stats["p95_tokens"] <= MAX_WORDPIECES


# --------------------------------------------------------------------------- #
# Degenerate fragments
#
# Found by app/eval_retrieval, not by inspection. heading_bound splits this
# corpus into 365 chunks of which 278 are under 100 characters - debris like
# "Category for HDFC Balanced Advantage Fund Direct Growth:" with the value cut
# off, and table rows such as "09% |". A sentence splitter that treats the colon
# in a "Label: value" fact line, or a trailing pipe, as a sentence end turns one
# scheme-facts block into dozens of meaningless chunks. Their embeddings sit
# surprisingly close to short factual queries: five of the top six slots for
# "What benchmark does the HDFC Balanced Advantage Fund track?" went to
# fragments, pushing the chunk that actually held the answer to rank 36 of 36.
#
# The word-piece cap cannot catch this. It is an upper bound, and a 5-character
# chunk is trivially inside it. Only a lower bound catches it.
#
# The fixture below is sized so heading_bound *must* split mid-section, because
# with a small section it keeps the whole thing and the test passes for the wrong
# reason. Verified to fail on heading_bound and pass on the shipped config.
# --------------------------------------------------------------------------- #

# Below this a chunk carries no retrievable meaning. Chosen from the observed
# data: the smallest chunk at the winning config is comfortably above it, while
# the fragments that caused the failure were under 50.
MIN_USEFUL_CHARS = 100

_FACTS = (
    ("Category", "Hybrid / Dynamic Asset Allocation"),
    ("Asset Management Company (AMC)", "HDFC"),
    ("Plan", "Direct Growth"),
    ("Expense ratio", "1.19% (base expense ratio 0.94%)"),
    ("Exit load", "1% if redeemed within 1 year"),
    ("Lock-in period", "none"),
    ("Riskometer", "Moderately High"),
    ("Benchmark", "NIFTY 50 Hybrid Composite Debt 50:50 Index"),
    ("Minimum lump sum investment", "Rs 100"),
    ("Minimum SIP amount", "Rs 100 per month"),
)

_HOLDINGS = "\n".join(
    f"| Company {i} Ltd | Financial | Equity | {i}.{i}% |" for i in range(1, 13)
)

FRAGMENT_DOC = RawDoc(
    **{**SAMPLE_DOC.__dict__,
       "text": (
           "# HDFC Sample Fund\n\n## Scheme facts\n\n"
           + "\n".join(
               f"{label} for HDFC Sample Fund Direct Growth: {value}."
               for label, value in _FACTS
           )
           + "\n\n## Holdings\n\n"
           + "| Holding | Sector | Weight |\n|---|---|---|\n"
           + _HOLDINGS
           + "\n"
       )},
)


def _fragments(chunks, limit: int = MIN_USEFUL_CHARS):
    return [c for c in chunks if c.char_len < limit]


def test_fixture_would_catch_the_original_bug():
    """Guard the guard.

    A regression test that passes on the broken configuration is worse than no
    test, because it reports safety while the bug is live. This asserts the
    fixture is load-bearing: it must fail on ``heading_bound`` - the strategy
    that produced the fragments - and pass on the config actually shipped.
    """
    broken = chunker_mod.run(
        FRAGMENT_DOC, chunker_mod.get_chunker("heading_bound", 400, 0)
    ).kept
    assert _fragments(broken), (
        "fixture no longer reproduces the bug - heading_bound produced no "
        "fragments, so the tests below are vacuous"
    )

    shipped = chunker_mod.run(FRAGMENT_DOC, chunker_mod.get_chunker()).kept
    assert not _fragments(shipped), (
        "the shipped config must not produce fragments"
    )


@needs_tokenizer
def test_a_scheme_facts_block_does_not_become_its_own_tiny_chunks():
    """'Label: value' fact lines must not each become a standalone chunk."""
    chunks = chunker_mod.run(FRAGMENT_DOC, chunker_mod.get_chunker()).kept
    assert chunks, "precondition: the fixture produced chunks"
    fragments = _fragments(chunks)
    assert not fragments, (
        f"{len(fragments)}/{len(chunks)} chunks are under {MIN_USEFUL_CHARS} "
        f"chars: "
        + ", ".join(repr(" ".join(c.text.split())) for c in fragments[:6])
    )


@needs_tokenizer
def test_a_markdown_table_does_not_become_its_own_tiny_chunks():
    """Table rows must not be split into standalone fragments either."""
    chunks = chunker_mod.run(FRAGMENT_DOC, chunker_mod.get_chunker()).kept
    table_fragments = [
        c for c in _fragments(chunks) if "|" in c.text or c.text.strip().endswith(":")
    ]
    assert not table_fragments, (
        "table rows or truncated fact lines were emitted as standalone chunks: "
        + ", ".join(repr(" ".join(c.text.split())) for c in table_fragments[:6])
    )


@needs_tokenizer
def test_the_answer_next_to_a_table_survives_inside_one_chunk():
    """The benchmark sentence must exist whole inside a single chunk.

    The failure was not only noise; it displaced the real answer. Retrieval can
    only find a span that lives inside one chunk, so a split sentence is
    unreachable no matter how well the index ranks it.
    """
    chunks = chunker_mod.run(FRAGMENT_DOC, chunker_mod.get_chunker()).kept
    span = "NIFTY 50 Hybrid Composite Debt 50:50 Index"
    holders = [c for c in chunks if span in c.text]
    assert holders, "the benchmark index name was split away from its sentence"
    # And it must sit in a chunk substantial enough to be retrievable at all.
    assert all(c.char_len >= MIN_USEFUL_CHARS for c in holders), (
        "the benchmark lives in a fragment, which cannot be ranked reliably"
    )


@needs_tokenizer
def test_configured_chunker_emits_no_degenerate_fragments_on_real_corpus():
    """The shipped config, on the real corpus, must not regress to fragments.

    Skipped when ``artifacts/raw_docs.jsonl`` is absent so the suite still runs
    on a clean checkout; the tests above cover the mechanism without it.
    """
    try:
        docs = chunker_mod.read_raw_docs()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"real corpus unavailable: {exc}")
    if not docs:
        pytest.skip("real corpus unavailable: no documents loaded")

    engine = chunker_mod.get_chunker()  # config-driven: strategy/size/overlap
    chunks = [c for doc in docs for c in chunker_mod.run(doc, engine).kept]
    assert chunks, "precondition: the real corpus produced chunks"

    fragments = _fragments(chunks)
    assert not fragments, (
        f"{len(fragments)}/{len(chunks)} chunks are under {MIN_USEFUL_CHARS} "
        f"chars, which is what displaced the golden answer: "
        + ", ".join(repr(" ".join(c.text.split())) for c in fragments[:6])
    )


# --------------------------------------------------------------------------- #
# Gold set integrity
# --------------------------------------------------------------------------- #


def test_golden_set_shape():
    gold = load_golden()
    assert len(gold) == 20, "FR-05 specifies 20 golden questions"
    assert len({g.id for g in gold}) == 20
    assert len({g.question for g in gold}) == 20
    per_scheme = {}
    for g in gold:
        per_scheme[g.expected_source_id] = per_scheme.get(g.expected_source_id, 0) + 1
    assert len(per_scheme) == 5, f"every scheme must be covered, got {per_scheme}"
    assert all(v == 4 for v in per_scheme.values()), per_scheme
    fact_types = {g.fact_type for g in gold}
    assert len(fact_types) >= 6, f"gold must span the fact types, got {fact_types}"


def test_golden_spans_are_verbatim_substrings_of_the_real_corpus():
    """Guards against a gold set that references text the corpus does not contain.

    Skips when Stage 1 has not been run, since raw_docs.jsonl is the only
    authority on what the corpus actually says.
    """
    raw = ROOT / "artifacts" / "raw_docs.jsonl"
    if not raw.exists():
        pytest.skip("artifacts/raw_docs.jsonl absent; run --stage load first")

    docs = {d.source_id: d for d in chunker_mod.read_raw_docs()}
    for g in load_golden():
        doc = docs.get(g.expected_source_id)
        assert doc is not None, f"{g.id} references unknown source {g.expected_source_id}"
        assert g.answer_span in doc.text, (
            f"{g.id}: answer_span is not verbatim in {g.expected_source_id} corpus text"
        )


def test_golden_must_include_tokens_appear_in_the_span():
    for g in load_golden():
        for token in g.must_include:
            assert token in g.answer_span, (
                f"{g.id}: must_include {token!r} is not inside its own answer_span"
            )


# --------------------------------------------------------------------------- #
# Experiment harness
# --------------------------------------------------------------------------- #


def test_pick_winner_breaks_ties_in_the_documented_order():
    """recall@5 first, then fewer chunks, then shorter mean."""
    base = dict(
        strategy="heading_bound", overlap=0.1, overlap_chars=80, n_chunks=100,
        mean_chars=500.0, p95_tokens=150, max_tokens=200, violations=0,
        rescues=0, build_seconds=1.0,
    )
    # NOTE: the fixtures must actually vary recall_at_5, otherwise the
    # "recall first" rule is never exercised and n_chunks decides every
    # comparison by default.
    high_recall = Result(size=800, recall_at_5=1.0, **base)
    low_recall = Result(size=800, recall_at_5=0.8, **{**base, "n_chunks": 1,
                                                      "mean_chars": 100.0})
    fewer_chunks = Result(size=800, recall_at_5=1.0, **{**base, "n_chunks": 50})
    shorter = Result(size=800, recall_at_5=1.0, **{**base, "n_chunks": 50,
                                                    "mean_chars": 400.0})
    over_cap = Result(
        size=800, recall_at_5=1.0,
        **{**base, "n_chunks": 1, "mean_chars": 100.0, "max_tokens": 400},
    )
    # recall@5 dominates, even against a config with a quarter of the chunks.
    assert pick_winner([low_recall, high_recall]) is high_recall
    # Equal recall -> fewer chunks wins, regardless of input order.
    assert pick_winner([fewer_chunks, high_recall]) is fewer_chunks
    # Equal recall and equal chunk count -> shorter mean wins.
    assert pick_winner([shorter, fewer_chunks]) is shorter
    # A cap violator loses even with perfect recall and one tiny chunk.
    assert pick_winner([over_cap, high_recall]) is high_recall
    # A skipped config is not a candidate at any score.
    assert pick_winner([Result(size=800, recall_at_5=1.0,
                               **{**base, "skipped": True})]) is None


def test_write_back_config_preserves_comments(tmp_path: Path):
    original = (
        "# top comment\n"
        "chunking:\n"
        "  strategy: heading_bound\n"
        "  # keep me\n"
        "  chunk_size: 800\n"
        "  chunk_overlap: 100\n"
        "embedding:\n"
        "  batch_size: 32\n"
    )
    target = tmp_path / "config.yaml"
    target.write_text(original, encoding="utf-8")

    winner = Result(
        strategy="semantic", size=1200, overlap=0.15, overlap_chars=180,
        n_chunks=10, recall_at_5=0.9, mean_chars=300.0, p95_tokens=190,
        max_tokens=250, violations=0, rescues=0, build_seconds=2.0,
    )
    assert write_back_config(winner, target) is True

    updated = target.read_text(encoding="utf-8")
    assert "# top comment" in updated
    assert "# keep me" in updated
    assert "strategy: semantic" in updated
    assert "chunk_size: 1200" in updated
    assert "chunk_overlap: 180" in updated
    # Must not touch the next block.
    assert "batch_size: 32" in updated
    # Still valid YAML with the winner actually applied.
    import yaml

    parsed = yaml.safe_load(updated)
    assert parsed["chunking"]["strategy"] == "semantic"
    assert parsed["chunking"]["chunk_size"] == 1200


def test_write_back_config_refuses_when_keys_are_absent(tmp_path: Path):
    target = tmp_path / "config.yaml"
    target.write_text("embedding:\n  batch_size: 32\n", encoding="utf-8")
    winner = Result(
        strategy="recursive", size=400, overlap=0.0, overlap_chars=0,
        n_chunks=5, recall_at_5=0.5, mean_chars=100.0, p95_tokens=80,
        max_tokens=120, violations=0, rescues=0, build_seconds=0.1,
    )
    assert write_back_config(winner, target) is False
    assert "batch_size: 32" in target.read_text(encoding="utf-8")


@needs_tokenizer
def test_build_config_converts_overlap_fraction_to_characters():
    docs = [SAMPLE_DOC]
    chunks, violations, _rescues, _secs = build_config(docs, "heading_bound", 800, 0.15)
    assert chunks
    assert violations == 0
    # 15% of 800 == 120 characters of overlap.
    engine = chunker_mod.get_chunker("heading_bound", 800, 120)
    expected = len(engine.split(SAMPLE_DOC))
    assert len(chunks) == expected


def test_report_renders_without_a_winner():
    report = render_report([], None, load_golden(), [SAMPLE_DOC])
    assert "No config completed" in report
    assert "Recall@5" in report


def test_golden_json_is_valid_and_documents_its_provenance():
    payload = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
    meta = payload["_meta"]
    assert "verbatim" in meta["answer_span_rule"].lower()
    # implementation.md 2.D names these fields exactly.
    for record in payload["questions"]:
        assert set(record) >= {
            "id", "question", "expected_url", "expected_source_id",
            "must_include", "answer_span", "fact_type",
        }
        assert 40 <= len(record["answer_span"]) <= 160, record["id"]
        assert record["expected_url"].startswith("https://groww.in/mutual-funds/")
    # A gold entry whose URL is not the allowlist URL for its source_id would
    # make Recall@5 unscoreable, so the pairing is asserted, not assumed.
    from app.sources import spec_for

    for record in payload["questions"]:
        assert record["expected_url"] == spec_for(record["expected_source_id"]).url


def test_golden_json_documents_the_missing_statements_fact_type():
    """The substitution must be justified in-file, not just in a commit message."""
    payload = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
    notes = " ".join(payload["_meta"]["notes"]).lower()
    assert "statement" in notes, (
        "golden.json must explain why 'how to download statements' is absent"
    )
    assert "sid_url" in notes or "sid_url" in json.dumps(payload).lower()
    types = {r["fact_type"] for r in payload["questions"]}
    assert "expense_ratio" in types
    assert "exit_load" in types
    assert "minimum_sip" in types
    assert "lock_in" in types
    assert "riskometer" in types
    assert "benchmark" in types


def test_pick_winner_ignores_skipped_configs():
    skipped = Result(
        strategy="semantic", size=400, overlap=0.0, overlap_chars=0, n_chunks=1,
        recall_at_5=1.0, mean_chars=10.0, p95_tokens=5, max_tokens=5,
        violations=3, rescues=0, build_seconds=0.1, skipped=True,
        skip_reason="cap violation",
    )
    honest = Result(
        strategy="heading_bound", size=800, overlap=0.1, overlap_chars=80,
        n_chunks=120, recall_at_5=0.95, mean_chars=600.0, p95_tokens=180,
        max_tokens=230, violations=0, rescues=0, build_seconds=1.0,
    )
    assert pick_winner([skipped, honest]) is honest
    assert pick_winner([skipped]) is None
    report = render_report([skipped, honest], honest, load_golden(), [SAMPLE_DOC])
    assert "Skipped configurations" in report
    assert "cap violation" in report
