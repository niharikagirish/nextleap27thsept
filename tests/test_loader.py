"""Phase 1 gate: extraction, cleaning, and the corpus-integrity invariants.

These tests run against the committed ``artifacts/raw_docs.jsonl`` and never
touch the network, so they are green on a fresh clone before ingestion has ever
been run.
"""

from __future__ import annotations

import json

import pytest

from app.pipeline import loader
from app.sources import ALLOWLIST


# --------------------------------------------------------------------------- #
# Unit tests: no corpus required
# --------------------------------------------------------------------------- #

def test_clean_strips_zero_width_and_collapses_whitespace():
    raw = "Expense\u200b ratio: 0.78%\r\n\r\n\r\n   Exit   load: 1%\r\n"
    cleaned = loader.clean(raw)
    assert "\u200b" not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "  " not in cleaned
    assert cleaned.startswith("Expense ratio: 0.78%")


def test_clean_drops_boilerplate_lines():
    raw = "Expense ratio: 0.78%\nDownload the Groww App today\nAll rights reserved\nExit load: 1%"
    cleaned = loader.clean(raw)
    assert "Expense ratio: 0.78%" in cleaned
    assert "Exit load: 1%" in cleaned
    assert "Download the Groww App" not in cleaned
    assert "All rights reserved" not in cleaned


def test_clean_keeps_long_lines_that_merely_mention_boilerplate():
    long_line = "The exit load of 1% applies " + ("and the terms are governed by the "
                 "scheme information document " * 6) + "which is available online."
    assert loader.clean(long_line) == long_line.strip()


def test_lock_in_renders_none_explicitly():
    assert loader._lock_in_phrase({"years": None, "months": None, "days": None}) == "none"
    assert loader._lock_in_phrase({"years": 3, "months": 0, "days": 0}) == "3 years"
    assert loader._lock_in_phrase({"years": 1, "months": 6, "days": 0}) == "1 year 6 months"
    assert loader._lock_in_phrase({"years": 0, "months": 0, "days": 1}) == "1 day"
    assert loader._lock_in_phrase(None) == "not stated"


def test_money_and_yes_no_formatters():
    assert loader._fmt_money(100) == "Rs 100"
    assert loader._fmt_money(500.5) == "Rs 500.50"
    assert loader._fmt_money(None) is None
    assert loader._fmt_money(0) is None
    assert loader._fmt_yes_no(True) == "Yes"
    assert loader._fmt_yes_no(False) == "No"
    assert loader._fmt_yes_no("true") == "Yes"
    assert loader._fmt_yes_no(None) is None


def test_performance_faq_patterns_match_expected_questions():
    blocked = [
        "What kind of returns does HDFC Small Cap Fund provide?",
        "What is the NAV of HDFC Small Cap Fund?",
        "What is the AUM of HDFC Small Cap Fund?",
        "Is it a good performing fund?",
        "How much has HDFC ELSS grown since inception?",
        "What is the money multiplier?",
        "best ranking among large caps",
    ]
    kept = [
        "How much expense ratio is charged?",
        "How to Invest in HDFC Small Cap Fund Direct Growth?",
        "How to Redeem HDFC Small Cap Fund Direct Growth?",
        "Can I invest in SIP and Lump Sum?",
        "What is the PE and PB ratio of HDFC Small Cap Fund?",
    ]
    for q in blocked:
        assert any(p.search(q) for p in loader.EXCLUDED_FAQ_PATTERNS), q
    for q in kept:
        assert not any(p.search(q) for p in loader.EXCLUDED_FAQ_PATTERNS), q


def test_every_real_faq_question_classifies_as_expected():
    """Regression guard for a real bug: a loose ``how (?:much|has).*grow`` pattern
    matched "How much **expense ratio** is charged by HDFC Small Cap Fund Direct
    **Growth**?" because "Growth" contains "grow", silently dropping the single
    most important FAQ on every page.

    These are the 8 question templates the schema.org FAQPage block actually uses,
    checked against a scheme name that ends in "Growth" — the hardest case.
    """
    scheme = "HDFC Small Cap Fund Direct Growth"
    cases = {
        "How to Invest in {s}?": "keep",
        "What kind of returns does {s} provide?": "drop",
        "How much expense ratio is charged by {s}?": "keep",
        "What is the AUM of {s}?": "drop",
        "How to Redeem {s}?": "keep",
        "Can I invest in SIP and Lump Sum of {s}?": "keep",
        "What is the NAV of {s}?": "drop",
        "What is the PE and PB ratio of {s}?": "keep",
    }
    for template, expected in cases.items():
        question = template.format(s=scheme)
        dropped = any(p.search(question) for p in loader.EXCLUDED_FAQ_PATTERNS)
        actual = "drop" if dropped else "keep"
        assert actual == expected, f"{question!r}: expected {expected}, got {actual}"

    # Every allowlisted scheme name ends in a growth variant, so confirm none of
    # them trips a pattern when the templates are substituted.
    for spec in ALLOWLIST:
        for template, expected in cases.items():
            question = template.format(s=spec.scheme_name)
            dropped = any(p.search(question) for p in loader.EXCLUDED_FAQ_PATTERNS)
            assert ("drop" if dropped else "keep") == expected, question


def test_facts_renderer_produces_readable_sentences():
    mf = {
        "scheme_name": "HDFC Test Fund Direct Growth",
        "category": "Equity", "sub_category": "Small Cap", "amc": "HDFC",
        "plan_type": "Direct", "scheme_type": "Growth",
        "expense_ratio": "0.78", "base_expense_ratio": "0.63",
        "exit_load": "Exit load of 1% if redeemed within 1 year",
        "min_sip_investment": 100, "min_investment_amount": 100,
        "mini_additional_investment": 100,
        "sip_allowed": True, "lumpsum_allowed": True,
        "lock_in": {"years": None, "months": None, "days": None},
        "nfo_risk": "Moderately High Riskometer",
        "benchmark": "BSE 250 SmallCap TRI",
        "benchmark_name": "BSE 250 SmallCap Total Return Index",
        "launch_date": "01-Jan-2013", "fund_manager": "Test Manager",
    }
    lines = loader._render_facts(mf)
    joined = "\n".join(lines)

    assert "Expense ratio of HDFC Test Fund Direct Growth is 0.78%" in joined
    assert "base expense ratio 0.63%" in joined
    assert "Exit load of HDFC Test Fund Direct Growth: Exit load of 1% if redeemed within 1 year" in joined
    assert "Minimum SIP amount for HDFC Test Fund Direct Growth is Rs 100 per month" in joined
    assert "Lock-in period for HDFC Test Fund Direct Growth: none." in joined
    # The label already says "Riskometer", so the redundant suffix is stripped.
    assert "Riskometer for HDFC Test Fund Direct Growth: Moderately High." in joined
    assert "Riskometer Riskometer" not in joined
    assert "Benchmark for HDFC Test Fund Direct Growth: BSE 250 SmallCap TRI " \
           "(BSE 250 SmallCap Total Return Index)." in joined


def test_nil_exit_load_is_stated_explicitly():
    lines = loader._render_facts({"scheme_name": "X", "exit_load": None, "lock_in": None})
    assert "Exit load of X: Nil." in "\n".join(lines)


# --------------------------------------------------------------------------- #
# Fund manager roster — regression for the stale-scalar bug
# --------------------------------------------------------------------------- #
#
# `mfServerSideData.fund_manager` is a single name Groww stopped maintaining. On
# four of the five corpus pages it names a manager who has left the fund, and on
# the fifth it omits a co-manager. The authoritative roster is
# `fund_manager_details`. These tests pin the fix so the scalar cannot creep back.

# Roster observed on the live pages 2026-09-27, including the stale scalar the
# page *also* carries, so a regression to the scalar is visible in the diff.
OBSERVED_ROSTERS: dict[str, tuple[str, list[str]]] = {
    "hdfc-large-cap": (
        "Prashant Jain",
        ["Rahul Baijal", "Dhruv Muchhal"],
    ),
    "hdfc-equity": (
        "Prashant Jain",
        ["Dhruv Muchhal", "Amit Ganatra"],
    ),
    "hdfc-elss": (
        "Vinay Kulkarni",
        ["Amar Kalkundrikar", "Dhruv Muchhal"],
    ),
    "hdfc-small-cap": (
        "Chirag Setalvad",
        ["Dhruv Muchhal", "Chirag Setalvad"],
    ),
    "hdfc-balanced-adv": (
        "Srinivas Rao Ravuri",
        [
            "Anil Bamboli", "Arun Agarwal", "Dhruv Muchhal",
            "Nandita Menezes", "Gopal Agrawal", "Ihab Dalwai",
        ],
    ),
}


def _mf_with_details(scheme_name: str, stale_scalar: str, details: list[dict]) -> dict:
    return {
        "scheme_name": scheme_name,
        "exit_load": None,
        "lock_in": None,
        "fund_manager": stale_scalar,
        "fund_manager_details": details,
    }


def test_fund_manager_comes_from_details_not_the_stale_scalar():
    mf = _mf_with_details(
        "HDFC Large Cap Fund Direct Growth",
        "Prashant Jain",
        [{"person_name": "Rahul Baijal", "funds_managed": [
            {"scheme_name": "HDFC Large Cap Fund Direct Growth"}]}],
    )
    lines = loader._render_facts(mf)
    joined = "\n".join(lines)

    assert "Fund manager of HDFC Large Cap Fund Direct Growth: Rahul Baijal." in joined
    assert "Prashant Jain" not in joined


def test_fund_manager_roster_is_joined_and_pluralised():
    mf = _mf_with_details(
        "HDFC Small Cap Fund Direct Growth",
        "Chirag Setalvad",
        [
            {"person_name": "Dhruv Muchhal", "funds_managed": [
                {"scheme_name": "HDFC Small Cap Fund Direct Growth"}]},
            {"person_name": "Chirag Setalvad", "funds_managed": [
                {"scheme_name": "HDFC Small Cap Fund Direct Growth"}]},
        ],
    )
    joined = "\n".join(loader._render_facts(mf))

    # Both managers, in page order — the scalar listed only one of the two.
    assert (
        "Fund managers of HDFC Small Cap Fund Direct Growth: "
        "Dhruv Muchhal, Chirag Setalvad." in joined
    )


def test_fund_manager_excludes_people_managing_only_a_sibling_scheme():
    """Dhruv Muchhal appears on every HDFC page; a page's detail list also
    contains people who manage a *different* scheme. Those must not be listed."""
    mf = _mf_with_details(
        "HDFC Large Cap Fund Direct Growth",
        "Prashant Jain",
        [
            {"person_name": "Rahul Baijal", "funds_managed": [
                {"scheme_name": "HDFC Large Cap Fund Direct Growth"}]},
            # Manages an ELSS scheme, not this one:
            {"person_name": "Amar Kalkundrikar", "funds_managed": [
                {"scheme_name": "HDFC ELSS Tax Saver Fund Direct Growth"}]},
        ],
    )
    joined = "\n".join(loader._render_facts(mf))

    assert "Rahul Baijal" in joined
    assert "Amar Kalkundrikar" not in joined


def test_fund_manager_entry_without_scheme_list_is_kept():
    """An entry that carries no `funds_managed` cannot be disproved, so it is
    kept rather than silently dropping a real manager."""
    mf = _mf_with_details(
        "HDFC Large Cap Fund Direct Growth", "Prashant Jain",
        [{"person_name": "Rahul Baijal"}],
    )
    assert "Rahul Baijal" in "\n".join(loader._render_facts(mf))


def test_fund_manager_falls_back_to_scalar_when_details_absent():
    """A future page that drops the list degrades to the old behaviour rather
    than to no answer at all."""
    joined = "\n".join(loader._render_facts({
        "scheme_name": "HDFC Test Fund Direct Growth",
        "exit_load": None, "lock_in": None,
        "fund_manager": "Test Manager",
    }))
    assert "Fund manager of HDFC Test Fund Direct Growth: Test Manager." in joined


def test_fund_manager_line_is_omitted_when_nothing_is_known():
    joined = "\n".join(loader._render_facts({
        "scheme_name": "HDFC Test Fund Direct Growth", "exit_load": None, "lock_in": None,
    }))
    assert "Fund manager" not in joined


def test_fund_manager_roster_matches_the_committed_corpus():
    """The committed corpus must carry the current roster, not the stale scalar.

    This is the test that fails if the artifacts are rebuilt from unfixed code,
    and it needs no network because it reads artifacts/raw_docs.jsonl.
    """
    expected = {
        slug: names for slug, (_stale, names) in OBSERVED_ROSTERS.items()
    }
    for record in _read_raw_docs():
        slug = record["source_id"]
        if slug not in expected:
            continue
        text = record["text"]
        for name in expected[slug]:
            assert name in text, f"{slug}: corpus is missing manager {name!r}"
        stale = OBSERVED_ROSTERS[slug][0]
        if stale not in {n for n in expected[slug]}:
            assert stale not in text, (
                f"{slug}: corpus still carries the stale scalar {stale!r}"
            )


def _read_raw_docs() -> list[dict]:
    from app.config import SETTINGS

    path = SETTINGS.path("loading.raw_docs_path", "artifacts/raw_docs.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_extract_rejects_empty_html():
    with pytest.raises(loader.ExtractionError):
        loader.extract("<html><body></body></html>")


def test_extract_raises_on_shell_with_no_facts():
    """A client-rendered shell with no embedded JSON must fail loudly, not
    silently produce an empty corpus entry (architecture.md §10)."""
    shell = "<html><head><title>Loading</title></head><body><div id=\"__next\"></div></body></html>"
    with pytest.raises(loader.ExtractionError):
        loader.extract(shell)


def test_fetcher_refuses_non_allowlisted_url():
    from app.models import SourceSpec

    rogue = SourceSpec("rogue", "Rogue", "Equity", "Direct-Growth",
                       "https://example.com/mutual-funds/anything")
    with pytest.raises(loader.AllowlistViolation):
        loader.fetch(rogue)


def test_fetcher_refuses_denied_host():
    from app.models import SourceSpec

    blog = SourceSpec("blog", "Blog", "Equity", "Direct-Growth",
                      "https://medium.com/some-mf-post")
    with pytest.raises(loader.AllowlistViolation):
        loader.fetch(blog)


# --------------------------------------------------------------------------- #
# Corpus tests: require artifacts/raw_docs.jsonl
# --------------------------------------------------------------------------- #

def _load_corpus() -> list[dict]:
    from app.config import SETTINGS

    path = SETTINGS.path("loading.raw_docs_path", "artifacts/raw_docs.jsonl")
    if not path.exists():
        pytest.skip(
            "artifacts/raw_docs.jsonl not present — run `python -m app.ingest "
            "--stage load` first"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_corpus_has_five_documents_matching_allowlist():
    docs = _load_corpus()
    assert len(docs) == 5
    by_id = {d["source_id"]: d for d in docs}
    for spec in ALLOWLIST:
        assert spec.source_id in by_id
        assert by_id[spec.source_id]["url"] == spec.url


def test_corpus_urls_are_all_allowlisted():
    for doc in _load_corpus():
        assert loader.is_allowed_url(doc["url"])


def test_corpus_has_meaningful_length():
    for doc in _load_corpus():
        assert doc["http_status"] == 200
        assert len(doc["text"]) > 500, doc["source_id"]
        assert len(doc["text"].split()) > 100, doc["source_id"]


def test_corpus_covers_all_seven_required_fact_types():
    """The brief requires expense ratio, exit load, min SIP, lock-in, riskometer,
    benchmark and document/statement guidance to be answerable."""
    # Markers must line up 1:1 with loader.REQUIRED_FACT_FIELDS, so the brief's
    # required fact types and the extractor's field map cannot drift apart.
    markers = {
        "expense_ratio": "Expense ratio",
        "exit_load": "Exit load",
        "minimum_sip": "Minimum SIP",
        "lock_in": "Lock-in period",
        "riskometer": "Riskometer",
        "benchmark": "Benchmark",
        "documents": "Official documents",
    }
    assert set(markers) == set(loader.REQUIRED_FACT_FIELDS)
    assert len(markers) == 7

    for doc in _load_corpus():
        for fact_type, marker in markers.items():
            assert marker in doc["text"], f"{doc['source_id']} missing {fact_type}"


def test_elss_corpus_states_three_year_lock_in():
    """The 3-year lock-in is the ELSS scheme's distinguishing fact.

    Asserted name-agnostically on purpose: facts are rendered using the page's
    own ``scheme_name`` ("HDFC ELSS Tax Saver Fund Direct Plan Growth"), which
    differs from the allowlist display name ("HDFC ELSS Tax Saver Fund").
    """
    docs = {d["source_id"]: d for d in _load_corpus()}
    lines = [
        line for line in docs["hdfc-elss"]["text"].split("\n")
        if line.startswith("Lock-in period for")
    ]
    assert len(lines) == 1, lines
    assert lines[0].endswith(": 3 years."), lines[0]


def test_non_elss_schemes_state_no_lock_in():
    for doc in _load_corpus():
        if doc["source_id"] == "hdfc-elss":
            continue
        assert "Lock-in period for" in doc["text"]
        assert ": none." in doc["text"], doc["source_id"]


def test_corpus_excludes_performance_content():
    """C4 defence in depth: return/NAV/AUM figures must not enter the corpus.

    Percentage signs are still expected (expense ratio, exit load), so this
    asserts on the return/NAV/AUM *wording* rather than on numbers.
    """
    forbidden = [
        "average annual returns", "SIP Returns", "CAGR", "XIRR",
        "1 year return", "annualised returns", "absolute return",
    ]
    for doc in _load_corpus():
        lowered = doc["text"].lower()
        for phrase in forbidden:
            assert phrase.lower() not in lowered, f"{doc['source_id']}: {phrase}"


def test_corpus_excluded_fields_never_leak():
    """The values of performance fields in EXCLUDED_FIELDS must not appear.

    AUM for HDFC Small Cap is 41890.8613 and NAV moves daily, so their absence is
    checkable even though the field names never appear in prose anyway.
    """
    for doc in _load_corpus():
        text = doc["text"]
        assert "41890.86" not in text, doc["source_id"]
        assert "return_stats" not in text


def test_seo_meta_description_is_dropped_from_the_corpus():
    """Regression: the Groww meta description is performance-laden boilerplate.

    All five pages serve the identical string
    "<Name> - Get latest NAV, SIP Returns & Rankings, Ratings, Fund Performance,
    Portfolio, Expense Ratio, Holding Analysis, and Peers. Invest in ... Online".
    It contains four C4-banned terms and no scheme-specific content beyond the
    name, which is already the H1. If it ever reappears, the C4 corpus guard is
    being satisfied only by luck.

    Every probe below must be a substring that occurs ONLY in the meta
    description, or the test fires on correct corpus content:

    * "fund performance" - the legitimate page *title* is "<Name> - NAV, Mutual
      Fund Performance & Portfolio", and "mutual fund performance" contains it.
    * "invest in hdfc" - collides with the real FAQ heading "How to invest in
      HDFC Large Cap Fund Direct Growth?".

    Four unambiguous markers from the template are enough: if the meta
    description leaked at all, "get latest nav" would be present with it.
    """
    template_bits = ["get latest nav", "sip returns", "rankings",
                     "holding analysis"]
    for doc in _load_corpus():
        lowered = doc["text"].lower()
        for bit in template_bits:
            assert bit not in lowered, f"{doc['source_id']}: meta description leaked '{bit}'"


def test_endstop_does_not_double_punctuate():
    """Regression: a value already ending in '.' must not gain a second one.

    The Balanced Advantage exit-load string ships as
    "... redemption within 1 year." and naively appending a stop produced "..".
    """
    assert loader._endstop("within 1 year") == "within 1 year."
    assert loader._endstop("within 1 year.") == "within 1 year."
    assert loader._endstop("Nil") == "Nil."
    assert loader._endstop("Nil.") == "Nil."
    assert loader._endstop("really?") == "really?"
    assert loader._endstop("  spaced  ") == "spaced."


def test_exit_load_line_has_exactly_one_terminal_stop():
    """Every rendered exit-load line must be a single well-formed sentence."""
    for doc in _load_corpus():
        for line in doc["text"].split("\n"):
            if line.startswith("Exit load of "):
                assert not line.endswith(".."), f"{doc['source_id']}: {line}"
                assert line.endswith("."), f"{doc['source_id']}: {line}"


def test_corpus_has_headings_for_the_heading_bounded_chunker():
    for doc in _load_corpus():
        assert doc["text"].startswith("# ")
        assert "## Scheme facts" in doc["text"]
        assert "Source page: " in doc["text"]


def test_corpus_is_free_of_app_boilerplate():
    for doc in _load_corpus():
        lowered = doc["text"].lower()
        assert "download the groww app" not in lowered
        assert "all rights reserved" not in lowered


def test_sources_csv_lists_five_rows():
    import csv as csv_mod

    from app.config import SETTINGS

    path = SETTINGS.path("loading.sources_csv_path", "data/sources.csv")
    if not path.exists():
        pytest.skip("data/sources.csv not present — run ingest first")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert len(rows) == 5
    assert {r["url"] for r in rows} == {s.url for s in ALLOWLIST}
    assert all(r["http_status"] == "200" for r in rows)


def test_offline_replay_requires_no_network(monkeypatch):
    docs = _load_corpus()
    if not docs:
        pytest.skip("no corpus")

    def explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("offline mode must not touch the network")

    monkeypatch.setattr(loader, "fetch", explode)
    replayed = loader.load_all(offline=True)
    assert len(replayed) == 5
    assert {d.source_id for d in replayed} == {d["source_id"] for d in docs}
