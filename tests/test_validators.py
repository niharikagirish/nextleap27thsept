"""Ordered-validator tests (C4, C5, C6, C7) — Phase 6.

Three groups, in increasing order of how much they protect:

1. **Order.** The seven steps run in the specified sequence and short-circuit.
   A test asserts the exact list of ``StepResult.step`` values, because a
   reordering is invisible in any single-answer test and would quietly change
   which failure wins.
2. **Individual checks.** C4, C5, C6, C7 each get their own tests, including
   the cases that are easy to get wrong (a legitimate ``%``, an uncited answer,
   a fifth sentence).
3. **Probes.** ``eval/opinion_probes.json`` and ``eval/ooc_probes.json`` drive
   the intent tier behaviour, and a synthetic model output stands in for a real
   LLM so the suite needs no key.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import disclaimers
from app.models import Chunk, RetrievedChunk
from app.pipeline import validators
from app.pipeline.validators import (
    MAX_SENTENCES,
    count_sentences,
    run_steps,
    strip_markdown,
    truncate_to_sentences,
    validate,
    violates_c4,
)

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
ALLOWED_URL = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
ALLOWED_URL_2 = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _chunk(url: str = ALLOWED_URL, source_id: str = "hdfc-large-cap",
           text: str = "Exit load is 1% if redeemed within 1 year.",
           fetched_at: str | None = None) -> RetrievedChunk:
    stamp = fetched_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")
    chunk = Chunk(
        chunk_id=f"{source_id}-0", text=text, embed_text=text, source_id=source_id,
        scheme_name="HDFC Large Cap Fund", scheme_slug=source_id,
        category="Large Cap", plan="Direct-Growth", url=url, page_title="Large Cap",
        heading_trail=["HDFC Large Cap Fund", "Exit load"],
        chunk_index=0, char_len=len(text), word_count=len(text.split()),
        content_hash="deadbeef", fetched_at=stamp,
    )
    return RetrievedChunk(chunk=chunk, score=0.81, rank=1)


def _chunks(*urls: str) -> list[RetrievedChunk]:
    return [_chunk(url=u, source_id=f"s{i}") for i, u in enumerate(urls or (ALLOWED_URL,))]


def _json(answer: str, sources: list[str] | None = None, **extra) -> str:
    payload = {"answer": answer, "sources": sources if sources is not None else [ALLOWED_URL],
               "refused": False, "reason": ""}
    payload.update(extra)
    return json.dumps(payload)


# --------------------------------------------------------------------------- #
# 1. The order, asserted directly
# --------------------------------------------------------------------------- #


def test_step_order_is_exactly_as_specified() -> None:
    result, steps = run_steps(_json("Exit load is 1% within 1 year."), _chunks())
    assert result  # no failure
    assert [s.step for s in steps] == [
        "parse", "pii_output", "c4_performance",
        "citation_present", "citation_provenance", "sentences", "freshness",
    ]


def test_parse_failure_short_circuits_everything_else() -> None:
    result, steps = run_steps("I'm not JSON at all", _chunks())
    assert result == {}
    assert [s.step for s in steps] == ["parse"]
    assert steps[0].ok is False


def test_c4_runs_before_citation_checks() -> None:
    """A well-sourced return claim is still blocked, so C4 must fire first.

    C4 short-circuits, so the assertion is that ``citation_present`` never runs:
    the return sentence was rejected on its own merits, not because the
    citation happened to be missing.
    """
    answer = "HDFC Large Cap Fund has given 18% returns in the last year."
    result, steps = run_steps(_json(answer, [ALLOWED_URL]), _chunks())
    names = [s.step for s in steps]
    assert names == ["parse", "pii_output", "c4_performance"]
    assert "citation_present" not in names
    assert "c4_violation" in result


def test_pii_runs_before_citation_checks() -> None:
    answer = "For PAN ABCDE1234F the exit load is 1%."
    result, steps = run_steps(_json(answer, [ALLOWED_URL]), _chunks())
    assert [s.step for s in steps] == ["parse", "pii_output"]
    assert "pii_categories" in result


def test_sentences_runs_before_freshness() -> None:
    long_answer = "One. Two. Three. Four. Five."
    _result, steps = run_steps(_json(long_answer, [ALLOWED_URL]), _chunks())
    names = [s.step for s in steps]
    assert names.index("sentences") < names.index("freshness")


# --------------------------------------------------------------------------- #
# 2a. C4 - no performance claims, but percentages are fine
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "answer",
    [
        "The fund has given 18% returns over one year.",
        "Its CAGR since inception is 14.2%.",
        "It outperformed the Nifty 50 last year.",
        "Performance has been strong so far.",
        "If you invest Rs 5000 a month for 10 years you will get about Rs 10 lakh.",
        "The fund guarantees returns.",
        "This is a risk-free investment.",
        "My SIP will become Rs 500000 in 10 years.",
    ],
)
def test_c4_blocks_performance_and_projection(answer: str) -> None:
    assert violates_c4(answer) is True
    out = validate(_json(answer, [ALLOWED_URL]), _chunks(), "1 year return?")
    assert out.answer.kind == "perf_redirect"
    assert out.answer.text == disclaimers.PERF_REDIRECT
    assert "CAGR" not in out.answer.text


@pytest.mark.parametrize(
    "answer",
    [
        "The expense ratio is 0.87%.",
        "The exit load is 1% if redeemed within 1 year.",
        "The total expense ratio is 1.19%, of which the management fee is 0.72%.",
        "The minimum SIP is Rs 500.",
    ],
)
def test_c4_allows_legitimate_percentages(answer: str) -> None:
    """The regression this guard is most likely to ship: banning every '%'."""
    assert violates_c4(answer) is False
    out = validate(_json(answer, [ALLOWED_URL]), _chunks(), "expense ratio?")
    assert out.answer.kind == "factual"
    # The figure must survive verbatim -- including the '%' where there is one.
    assert out.answer.text == answer


def test_c4_blocks_advice_language_from_the_model() -> None:
    out = validate(
        _json("You should buy this fund for your goals.", [ALLOWED_URL]),
        _chunks(), "exit load?",
    )
    assert out.answer.kind in {"perf_redirect", "refusal"}


def test_unexplained_percentage_is_treated_as_a_violation() -> None:
    assert violates_c4("The value is 42% today.") is True


# --------------------------------------------------------------------------- #
# 2a-bis. C4 scoping - a fee percentage near return vocabulary
#
# Found against a live model: a correct answer such as "The expense ratio ...
# is 0.78%." was rejected as `percentage_near_return_language` whenever the model
# also used the word "returns" or "performance" somewhere, because the backwards
# window spanned a sentence boundary and no fee label got the benefit of the
# doubt. A correct answer must not be lost to a heuristic meant to catch a
# different one.
#
# Every input below contains return *vocabulary* but makes no return *claim* -
# which is precisely the distinction the old window could not make.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "answer",
    [
        "This is not a returns figure. The expense ratio is 0.78%.",
        "Performance figures are on the scheme page. The TER is 0.63%.",
        "Returns can be negative. The exit load is 1%.",
        "Performance varies with the market. The management fee is 0.72%.",
    ],
)
def test_c4_scopes_return_window_to_the_owning_sentence(answer: str) -> None:
    """Return vocabulary in another sentence does not taint this fee figure."""
    assert violates_c4(answer) is False


def test_c4_exempts_a_percentage_whose_own_label_is_a_fee() -> None:
    """The fee label owns the number even with return words in between.

    Exercises the label check rather than the sentence split: here the return
    vocabulary sits *between* the fee label and its value, inside one sentence, so
    only looking at the label can tell the number is a fee.
    """
    assert violates_c4("The expense ratio, which varies with yields, is 0.78%.") is False


@pytest.mark.parametrize(
    "answer",
    [
        "The expense ratio is 0.78% and returns are 15.2%.",
        "The fund returned 15.2% last year.",
        "Its 1-year returns were 15.2%.",
        "The fund has given 18% returns over one year.",
        "Returns have been strong and consistent performance.",
    ],
)
def test_c4_still_blocks_real_return_claims(answer: str) -> None:
    """The fix must not defuse the rule it was fixing.

    Exempting fee-labelled percentages is only safe while genuine return claims
    are still blocked - otherwise the exemption is just a loophole. These mix a
    legitimate fee with an actual return figure in one answer, which is the case
    most at risk of being over-corrected.
    """
    assert violates_c4(answer) is True


def test_c4_explains_a_number_with_no_label_and_no_fee_context() -> None:
    """A % with neither a fee label nor return vocabulary stays blocked.

    Guards the interaction with the ``unexplained_percentage`` fallback: skipping
    a fee-labelled number must not accidentally exempt every number.
    """
    assert violates_c4("The value is 42% today.") is True
    assert violates_c4("It was 42%.") is True


# --------------------------------------------------------------------------- #
# 2b. C5 - citation present and provenance-checked
# --------------------------------------------------------------------------- #


def test_uncited_answer_becomes_an_abstention() -> None:
    out = validate(_json("Exit load is 1% within 1 year.", []), _chunks(), "exit load?")
    assert out.answer.kind == "abstain"
    assert out.answer.sources == []
    assert disclaimers.ABSTAIN_PREFIX in out.answer.text


def test_factual_answer_always_carries_a_citation() -> None:
    out = validate(_json("Exit load is 1% within 1 year."), _chunks(), "exit load?")
    assert out.answer.kind == "factual"
    assert out.answer.sources, "C5: a factual answer must have >=1 source"
    assert out.answer.sources[0] == ALLOWED_URL


def test_citation_outside_the_retrieved_set_is_dropped() -> None:
    out = validate(
        _json("Exit load is 1% within 1 year.", ["https://example.com/other"]),
        _chunks(), "exit load?",
    )
    assert out.answer.kind == "abstain", "all citations were fabricated"
    assert out.answer.sources == []


def test_mixed_citations_keep_only_the_retrieved_one() -> None:
    out = validate(
        _json("Exit load is 1% within 1 year.", ["https://example.com/x", ALLOWED_URL]),
        _chunks(), "exit load?",
    )
    assert out.answer.kind == "factual"
    assert out.answer.sources == [ALLOWED_URL]


def test_model_refusal_uses_the_canonical_refusal_string() -> None:
    raw = json.dumps({"answer": "I will not help with that", "sources": [],
                      "refused": True, "reason": "policy"})
    out = validate(raw, _chunks(), "should I buy?")
    assert out.answer.kind == "refusal"
    assert out.answer.text == disclaimers.REFUSAL
    assert "I will not help" not in out.answer.text


# --------------------------------------------------------------------------- #
# 2c. C6 - at most three sentences
# --------------------------------------------------------------------------- #


def test_sentence_counting_survives_decimals_and_abbreviations() -> None:
    assert count_sentences("The ratio is 1.19% today.") == 1
    assert count_sentences("Rs. 500 is the minimum. Lock-in is 3 years.") == 2


def test_five_sentences_are_truncated_to_three() -> None:
    out = validate(
        _json("One fact. Two facts. Three facts. Four facts. Five facts.", [ALLOWED_URL]),
        _chunks(), "q?",
    )
    assert out.answer.kind == "factual"
    assert count_sentences(out.answer.text) == MAX_SENTENCES
    assert "Four facts" not in out.answer.text


def test_truncation_never_leaves_a_partial_sentence() -> None:
    text = "Alpha. Bravo. Charlie. Delta. Echo."
    truncated = truncate_to_sentences(text, 2)
    assert truncated == "Alpha. Bravo."
    assert count_sentences(truncated) == 2


def test_markdown_is_stripped_before_counting() -> None:
    raw = _json("**Exit load** is 1%.\n- within 1 year\n- after 1 year", [ALLOWED_URL])
    out = validate(raw, _chunks(), "exit load?")
    assert "**" not in out.answer.text
    assert strip_markdown("```json\nExit load is 1%.\n```") == "Exit load is 1%."


def test_code_fenced_answer_is_still_validated() -> None:
    raw = ('```json\n{"answer": "Exit load is 1%.", "sources": ["' + ALLOWED_URL
           + '"]}\n```')
    out = validate(raw, _chunks(), "exit load?")
    assert out.answer.kind == "factual"
    assert out.answer.text == "Exit load is 1%."


# --------------------------------------------------------------------------- #
# 2d. C7 - freshness
# --------------------------------------------------------------------------- #


def _stale(days: int) -> RetrievedChunk:
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
        timespec="seconds").replace("+00:00", "Z")
    return _chunk(fetched_at=stamp)


def test_fresh_source_gets_no_caveat() -> None:
    out = validate(_json("Exit load is 1% within 1 year."), _chunks(), "exit load?")
    assert out.answer.kind == "factual"
    assert disclaimers.FRESHNESS_CAVEAT not in out.answer.text
    assert out.answer.last_updated


def test_stale_source_spends_a_sentence_on_the_caveat() -> None:
    out = validate(
        _json("Exit load is 1% within 1 year. Lock-in is 3 years.", [ALLOWED_URL]),
        [_stale(400)], "exit load?",
    )
    assert out.answer.kind == "factual"
    assert disclaimers.FRESHNESS_CAVEAT in out.answer.text
    assert count_sentences(out.answer.text) <= MAX_SENTENCES, (
        "C6 still holds when the freshness caveat is appended"
    )


def test_unparseable_timestamp_does_not_crash_or_claim_freshness() -> None:
    chunk = _chunk()
    object.__setattr__(chunk.chunk, "fetched_at", "not-a-date")
    out = validate(_json("Exit load is 1%."), [chunk], "exit load?")
    assert out.answer.kind == "factual"


# --------------------------------------------------------------------------- #
# 2e. Output PII never reaches the user
# --------------------------------------------------------------------------- #


def test_pii_in_model_output_is_replaced_not_echoed() -> None:
    raw = _json("For PAN ABCDE1234F the exit load is 1%.", [ALLOWED_URL])
    out = validate(raw, _chunks(), "exit load?")
    assert out.answer.kind == "pii_notice"
    assert out.answer.text == disclaimers.PII_NOTICE
    assert "ABCDE1234F" not in out.answer.text
    assert out.answer.sources == []


# --------------------------------------------------------------------------- #
# 3. Probe files
# --------------------------------------------------------------------------- #


def _load(name: str) -> list[dict]:
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))["probes"]


def test_opinion_probes_are_wellformed() -> None:
    probes = _load("opinion_probes.json")
    assert len(probes) >= 15
    kinds = {p["expect_kind"] for p in probes}
    assert kinds <= {"refusal", "perf_redirect", "abstain"}
    assert "refusal" in kinds and "perf_redirect" in kinds


def test_ooc_probes_are_wellformed() -> None:
    probes = _load("ooc_probes.json")
    assert len(probes) >= 10
    assert all(p["expect_kind"] == "abstain" for p in probes)
    assert all(p["tier"] == 2 for p in probes)


def test_probe_ids_are_unique_across_all_three_files() -> None:
    ids: list[str] = []
    for name in ("opinion_probes.json", "pii_probes.json", "ooc_probes.json"):
        ids += [p["id"] for p in _load(name)]
    assert len(ids) == len(set(ids)), "duplicate probe id across probe files"
