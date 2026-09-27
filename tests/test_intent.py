"""Intent-gate tests (FR-03, FR-04, C4) — the two tiers must not be conflated.

The distinction under test is the one that is easiest to get wrong and hardest
to notice: a **refusal** says "I won't" and a **performance redirect** says "I
won't state returns, look here", while an **abstention** says "that isn't in my
sources". Answering "How do I download my capital gains statement?" with a
refusal is a worse product than abstaining, and it is invisible in a test that
only asserts `kind != "factual"`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.guards.intent import classify

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def _probes(name: str) -> list[dict]:
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))["probes"]


# --------------------------------------------------------------------------- #
# Tier 1 - out of scope by policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question",
    [
        "Should I buy the HDFC Large Cap Fund?",
        "Would you recommend HDFC Small Cap Fund?",
        "Is HDFC Balanced Advantage Fund a good investment?",
        "Which is better, HDFC Large Cap or HDFC Small Cap?",
        "Is it a good time to exit HDFC Flexi Cap?",
        "How much should I invest monthly?",
        "Is this fund safe for my retirement?",
        "Does HDFC ELSS suit me for tax saving?",
    ],
)
def test_advice_requests_are_refused(question: str) -> None:
    assert classify(question).intent == "refusal"


@pytest.mark.parametrize(
    "question",
    [
        "What is the 1 year return?",
        "What is the CAGR since inception?",
        "How has it performed vs Nifty 50?",
        "Which fund has the best returns?",
        "What is the current NAV?",
        "If I invest 5000 a month for 10 years what will I get?",
        "What is the worst drawdown?",
        "Does it guarantee returns?",
        "What is the AUM and has it grown?",
    ],
)
def test_performance_requests_are_redirected(question: str) -> None:
    assert classify(question).intent == "perf_redirect"


def test_advice_outranks_performance() -> None:
    """'Should I buy the best performing fund?' is a refusal, not a redirect."""
    result = classify("Should I buy the best performing HDFC fund?")
    assert result.intent == "refusal"


# --------------------------------------------------------------------------- #
# Tier 2 - in scope, absent from the corpus
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question",
    [
        "How do I download a capital gains statement?",
        "Where can I get my consolidated account statement?",
        "How do I download the tax statement for FY 2024-25?",
        "What is the value of my folio?",
        "What is my folio number?",
        "How do I update my bank mandate?",
        "What is the toll-free customer care number?",
        "How do I register a complaint?",
        "How long does a redemption take to credit?",
        "What is the weather in Mumbai?",
    ],
)
def test_out_of_corpus_questions_abstain(question: str) -> None:
    assert classify(question).intent == "abstain"


def test_abstain_is_not_a_refusal() -> None:
    """The whole point of two tiers: statement questions abstain, they don't refuse."""
    statement = classify("How do I download a capital gains statement?")
    advice = classify("Should I buy HDFC Small Cap Fund?")
    assert statement.intent == "abstain" and advice.intent == "refusal"
    assert statement.intent != advice.intent


def test_facts_pass_through() -> None:
    for question in (
        "What is the expense ratio of the HDFC Large Cap Fund?",
        "Does the HDFC ELSS Tax Saver Fund have a lock-in period?",
        "What is the exit load of HDFC Flexi Cap?",
        "What is the minimum SIP amount?",
        "What is the benchmark of HDFC Balanced Advantage Fund?",
        "Who manages HDFC Large Cap Fund?",
        "What is the riskometer category?",
    ):
        assert classify(question).intent == "factual", f"wrongly gated: {question}"


def test_empty_question_abstains() -> None:
    assert classify("").intent == "abstain"
    assert classify("   ").intent == "abstain"


# --------------------------------------------------------------------------- #
# The precision that keeps the gate usable
# --------------------------------------------------------------------------- #


def test_bookkeeping_uses_of_return_are_not_performance_questions() -> None:
    """"What is the return period for exit load?" is about exit load."""
    for question in (
        "What is the exit load return period?",
        "What is the return period for the exit load table?",
    ):
        assert classify(question).intent == "factual", f"wrongly redirected: {question}"


def test_fund_facts_with_digits_are_not_mistaken_for_performance() -> None:
    assert classify("Is the minimum SIP Rs 500 or Rs 1000?").intent == "factual"


def test_scheme_mentions_do_not_trip_the_gate() -> None:
    assert classify("What is the expense ratio of HDFC Small Cap Fund?").intent == "factual"


# --------------------------------------------------------------------------- #
# Probe files drive the suite
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "probe", _probes("opinion_probes.json"), ids=lambda p: p["id"]
)
def test_opinion_probe_expectation_holds(probe: dict) -> None:
    result = classify(probe["question"])
    assert result.intent == probe["expect_kind"], (
        f"{probe['id']}: expected {probe['expect_kind']}, got {result.intent}"
    )


@pytest.mark.parametrize("probe", _probes("ooc_probes.json"), ids=lambda p: p["id"])
def test_ooc_probe_abstains(probe: dict) -> None:
    result = classify(probe["question"])
    assert result.intent == "abstain", (
        f"{probe['id']}: expected abstain, got {result.intent} ({result.matched})"
    )


# --------------------------------------------------------------------------- #
# The gate must not abstain on a question the corpus CAN answer
# --------------------------------------------------------------------------- #


def _golden() -> list[dict]:
    return json.loads(
        (EVAL_DIR / "golden.json").read_text(encoding="utf-8"))["questions"]


@pytest.mark.parametrize("question", [q["question"] for q in _golden()],
                         ids=[q["id"] for q in _golden()])
def test_every_golden_question_passes_the_gate(question: str) -> None:
    """The 20 labelled questions are the corpus's known-answer set.

    A single one of these being gated would mean the gate is firing on a fact
    the corpus contains, which is the most damaging precision failure available.
    """
    assert classify(question).intent == "factual"


def test_scheme_aliases_are_recognised() -> None:
    """The Flexi Cap scheme is allowlisted as 'HDFC Equity Fund (Flexi Cap)'."""
    for question in (
        "What is the expense ratio of the HDFC Flexi Cap Fund?",
        "What is the expense ratio of HDFC Equity Fund (Flexi Cap)?",
        "Exit load of HDFC Equity Fund?",
    ):
        assert classify(question).intent == "factual", f"wrongly gated: {question}"


# --------------------------------------------------------------------------- #
# Unlisted schemes: refuse to answer about a fund we do not have
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question",
    [
        "What is the expense ratio of the HDFC Mid-Cap Opportunity Fund?",
        "What is the exit load of the SBI Bluechip Fund?",
        "Tell me about the Axis Small Cap Fund.",
    ],
)
def test_unlisted_schemes_abstain(question: str) -> None:
    """The wrong-fund failure mode: real citation, wrong number, confident tone."""
    result = classify(question)
    assert result.intent == "abstain"
    assert result.matched == "unlisted_scheme"


def test_an_invented_fund_name_is_not_accepted_as_a_substring() -> None:
    """'hdfc large cap' is inside 'hdfc large cap mid cap fund' - it must not pass."""
    assert classify("Exit load of the HDFC Large Cap Mid Cap Fund?").intent == "abstain"


def test_off_topic_questions_abstain() -> None:
    for question in ("What is the weather in Mumbai?",
                     "Who won the last World Cup?",
                     "What is the capital of France?"):
        assert classify(question).intent == "abstain", f"not gated: {question}"


def test_a_scheme_name_alone_keeps_a_vague_question_in_scope() -> None:
    assert classify("What about HDFC Small Cap Fund?").intent == "factual"


def test_judging_a_fee_is_a_refusal_but_stating_it_is_a_fact() -> None:
    """The exact pair that separates 'what is' from 'what do you think'."""
    assert classify("What is the expense ratio of HDFC Large Cap Fund?").intent == "factual"
    assert classify("Is the expense ratio of 0.87% high or low?").intent == "refusal"
    assert classify("Is the exit load too high?").intent == "refusal"
