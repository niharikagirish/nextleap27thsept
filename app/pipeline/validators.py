"""STAGE 6 — ordered output validators (architecture.md §6, C4, C5, C6, C7).

**The order below is the specification, not a style preference.** Each step
exists because a *later* check cannot be trusted once an *earlier* one has
failed, and a user seeing a malformed or unsourced answer is the worst outcome
this system can produce.

    1. parse              — never interpret a string you have not parsed
    2. pii_output         — strip identifiers BEFORE they reach a user
    3. c4_performance     — no returns/advice, overriding everything below
    4. citation_present   — a factual answer with no citation is not factual
    5. citation_provenance— every URL must be one we actually retrieved
    6. sentences          — <=3 sentences (C6), a hard answer invariant
    7. freshness          — how old is the page behind this answer (C7)

Why these relative positions:

* **PII before everything.** Redacting after the answer is displayed, logged, or
  used to build a citation is a failure. It is also the only step that must
  *replace* the answer outright.
* **C4 before citations.** A return claim must be removed even if it is
  perfectly, provably sourced. Running citation checks first would let a
  well-sourced performance sentence survive by having a good URL attached.
* **Citations before sentence limits.** If the answer is going to be abandoned
  for lack of a source, truncating its sentences first is wasted work and can
  change *which* sentences get cut, producing a different, unreviewed answer.
* **Sentence limit before freshness.** The limit is a hard invariant on the
  output, so it is established first; freshness then spends whatever sentence
  budget remains.

Two deliberate non-goals:

* **A validator never invents content.** A failed check produces a canonical
  refusal/abstention/redirect, or a truncation, never an invented number or a
  substituted URL.
* **Percent signs are not banned.** C4 targets return and performance *claims*,
  not the character ``%``. "The expense ratio is 0.87%" is a legitimate fact
  from the corpus and must survive. :func:`violates_c4` keys on return language
  and growth projections, so fees and TER percentages pass untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from .. import disclaimers
from ..config import SETTINGS
from ..guards import pii
from ..logging_utils import emit, q_hash
from ..models import Answer, RetrievedChunk
from .chunker import split_sentences
from .generator import ParseError, _parse

MAX_SENTENCES = 3

# --- C4 detection ---------------------------------------------------------- #
# Return/performance *claims*. Deliberately not keyed on "%".
_RETURN_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(?:cagr|annualized?\s+return|average\s+return|"
               r"rolling\s+return|yearly\s+return|returns?\s+(?:are|of|is|was|"
               r"since|over|for|and)\b)"),
    re.compile(r"(?i)\b(?:outperform(?:ed|s|ing)?|underperform(?:ed|s|ing)?|"
               r"beat\s+the\s+(?:benchmark|index|nifty)|"
               r"topped\s+the\s+benchmark)\b"),
    re.compile(r"(?i)\b(?:performance\s+has|performance\s+is|performance\s+of|"
               r"strong\s+performance|weak\s+performance|good\s+performance|"
               r"poor\s+performance|consistent\s+performance)\b"),
    re.compile(r"(?i)\b(?:gave|given|returns?|returned|returning|earning|earned|"
               r"yielded|yielding|rose|risen)\b[^.!?\n]{0,25}?"
               r"\d+(?:\.\d+)?\s*%"),
    # Growth projections: the "what will I get" family, which is advice by
    # another name and is the most common way a facts bot leaks a forecast.
    re.compile(r"(?i)\b(?:in\s+\d+\s+years?|after\s+\d+\s+years?|"
               r"your\s+(?:investment|money|wealth|sip)\s+will|"
               r"you\s+will\s+(?:get|earn|have)|will\s+grow\s+to|"
               r"grow\s+to\s+rs|become\s+rs\s*\d|double[sd]?\s+(?:in|within)|"
               r"triple[sd]?\s+(?:in|within)|target\s+amount\s+of|"
               r"expected\s+(?:returns?|value|growth)|projected\s+value|"
               r"wealth\s+of|worth\s+rs\s*\d)\b"),
    re.compile(r"(?i)\b(?:guarantee[sd]?|risk[\s\-]?free|assured\s+returns?|"
               r"safe\s+investment|will\s+not\s+lose)\b"),
    re.compile(r"(?i)\b(?:should\s+i|would\s+you|recommend|"
               r"you\s+should|one\s+should|you\s+ought\s+to)\b"
               r"[^.!?\n]{0,40}?"
               r"\b(?:buy|sell|invest|hold|redeem|allocate|switch|start)\b"),
)

# A percentage is *explained* when one of these appears anywhere in the answer.
# Checked against the WHOLE text rather than a sliding window: a window-based
# check splits a long, perfectly legitimate fee sentence ("total expense ratio
# 1.19%, of which the management fee is 0.72%") into slices that each lack the
# fee vocabulary, and then blocks a correct answer.
_FEE_CONTEXT = re.compile(
    r"(?i)\b(?:expense\s+ratio|ter\b|total\s+expense|exit\s+load|entry\s+load|"
    r"maintenance\s+charge|asset\s+management\s+fee|fees?|charges?)\b"
)

# A percentage sitting close after any of these is a return claim, whatever the
# rest of the sentence says. Checked in a backwards window per "%".
_RETURN_WORD = re.compile(
    r"(?i)\b(?:return|returns|returned|returning|cagr|yield|yielded|grew|grown|"
    r"growth|profit|profits|gains?|earned|appreciation|appreciated|surge[ds]?|"
    r"rally|rallied|upside|performance)\b"
)

_BACK_WINDOW = 60

# How far back to look for the label that owns a given "%". Shorter than
# _BACK_WINDOW on purpose: this asks "what is this number?", not "is a return
# mentioned nearby?".
_FEE_LABEL_WINDOW = 40

_SENTENCE_SPLIT = re.compile(r"[.!?\n]")


def _owns_a_fee(answer: str, percent_index: int) -> bool:
    """True when the ``%`` at *percent_index* is the value of a fee quantity.

    An expense ratio, exit load or TER is a percentage, so the "percentages near
    return vocabulary are suspicious" heuristic cannot tell ``0.78%`` (a fee) from
    ``15.2%`` (a return) by looking at the number. It has to look at the label.
    Without this, a perfectly correct answer like "The expense ratio is 0.78%" is
    rejected purely because the number is a percentage at all.
    """
    head = answer[max(0, percent_index - _FEE_LABEL_WINDOW):percent_index]
    return bool(_FEE_CONTEXT.search(head))


def _violates_return(answer: str) -> str | None:
    """Return the first C4 violation label, or ``None``.

    A ``%`` on its own is never a violation. The question is always "is this
    percentage part of a return or growth claim?", and there are two ways to
    answer yes: a return phrase matches outright, or a ``%`` is preceded closely
    by return vocabulary. A ``%`` with *no* return vocabulary anywhere near it is
    also treated as a violation when the answer contains no fee vocabulary at
    all, because an unexplained number is either a return or a hallucination and
    neither should be shown.

    Two scoping rules keep the heuristic from firing on correct answers:

    * A percentage labelled as a fee is exempt (:func:`_owns_a_fee`).
    * The backwards window is clipped to the current sentence. Return vocabulary
      in a *previous* sentence describes a different number - otherwise "Its
      1-year returns were 15.2%. The expense ratio is 0.78%." would be rejected
      for the second percentage, which is exactly the shape of a good answer.
    """
    for pattern in _RETURN_CLAIM_PATTERNS:
        if pattern.search(answer):
            return pattern.pattern[:48]

    if "%" in answer:
        for match in re.finditer(r"%", answer):
            if _owns_a_fee(answer, match.start()):
                continue
            window = answer[max(0, match.start() - _BACK_WINDOW):match.start()]
            # Only the tail of the current sentence can describe this number.
            window = _SENTENCE_SPLIT.split(window)[-1]
            if _RETURN_WORD.search(window):
                return "percentage_near_return_language"
        if not _FEE_CONTEXT.search(answer):
            return "unexplained_percentage"
    return None


def count_sentences(text: str) -> int:
    """Sentence count, delegating to the chunker's abbreviation-aware splitter."""
    return len([s for s in split_sentences(text or "") if s.strip()])


def truncate_to_sentences(text: str, limit: int = MAX_SENTENCES) -> str:
    """Keep at most ``limit`` sentences. Never leaves a partial final sentence."""
    parts = [s.strip() for s in split_sentences(text or "") if s.strip()]
    return " ".join(parts[:limit])


def strip_markdown(text: str) -> str:
    """Remove fences, bullet markers and bold from a model answer.

    Models wrap answers in ``` fences and ``**`` even when told not to, and a
    bullet list would inflate the sentence count while adding nothing.
    """
    out = re.sub(r"^\s*```(?:json|text)?\s*", "", text or "")
    out = re.sub(r"\s*```\s*$", "", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"(?m)^\s*[-*•]\s+", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


# --- C7 freshness ---------------------------------------------------------- #


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _newest_fetched_at(chunks: Sequence[RetrievedChunk]) -> str:
    stamps = [c.chunk.fetched_at for c in chunks if c.chunk.fetched_at]
    return max(stamps) if stamps else ""


def _age_days(value: str, now: datetime | None = None) -> float | None:
    stamp = _parse_ts(value)
    if stamp is None:
        return None
    reference = now or datetime.now(timezone.utc)
    return (reference - stamp).total_seconds() / 86400.0


# --- result plumbing ------------------------------------------------------- #

ABSENT_REASON = "no_citation"
ABSTAIN_TEXT = f"{disclaimers.ABSTAIN_PREFIX} {disclaimers.ABSTAIN_SUGGESTIONS}"


@dataclass(frozen=True)
class StepResult:
    """Outcome of one ordered check. ``step`` values are asserted by tests."""

    step: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Validation:
    answer: Answer
    steps: list[StepResult]


def _log(step: str, question: str, ok: bool, **fields: Any) -> None:
    emit("validate", step=step, q_hash=q_hash(question), fresh=ok, **fields)


def run_steps(raw: str, chunks: Sequence[RetrievedChunk]) -> tuple[dict, list[StepResult]]:
    """Run the 7 ordered checks, short-circuiting on the first hard failure.

    Returns ``(parsed_or_empty, steps)``. The caller (:func:`validate`) turns the
    failing step into the final :class:`Answer`. Splitting it this way is what
    lets ``tests/test_validators.py`` assert the *order* and the *short-circuit*
    without re-implementing the policy.
    """
    steps: list[StepResult] = []

    def record(name: str, ok: bool, detail: str = "") -> bool:
        steps.append(StepResult(name, ok, detail))
        return ok

    # ---- 1. parse ------------------------------------------------------- #
    try:
        parsed = _parse(raw)
    except ParseError as exc:
        record("parse", False, str(exc))
        return {}, steps
    record("parse", True)

    answer_text = strip_markdown(parsed["answer"])

    # ---- 2. pii_output -------------------------------------------------- #
    hits = pii.scan(answer_text)
    if hits:
        pii.note_block()
        record("pii_output", False, ",".join(hits))
        return {"pii_categories": hits}, steps
    record("pii_output", True)

    # ---- 3. c4_performance --------------------------------------------- #
    violation = _violates_return(answer_text)
    if violation:
        record("c4_performance", False, violation)
        return {"c4_violation": violation}, steps
    record("c4_performance", True)

    # ---- 4. citation_present -------------------------------------------- #
    retrieved_urls = {c.chunk.url for c in chunks if c.chunk.url}
    if not parsed["sources"]:
        # An empty citation list plus a non-empty answer is an unsourced claim.
        # Refusals are honoured here with the canonical string, not the model's.
        if parsed["refused"] or not answer_text:
            record("citation_present", False, "model_refused")
            return {"model_refused": True}, steps
        record("citation_present", False, ABSENT_REASON)
        return {"no_citation": True}, steps
    record("citation_present", True)

    # ---- 5. citation_provenance ----------------------------------------- #
    bad = [url for url in parsed["sources"] if url not in retrieved_urls]
    kept = [url for url in parsed["sources"] if url in retrieved_urls]
    if bad:
        record("citation_provenance", False, f"dropped={len(bad)}")
        if not kept:
            return {"no_provenance": True, "dropped": len(bad)}, steps
    else:
        record("citation_provenance", True)

    # ---- 6. sentences --------------------------------------------------- #
    limit = int(SETTINGS.get("generation.max_sentences", MAX_SENTENCES) or MAX_SENTENCES)
    sentences = count_sentences(answer_text)
    if sentences > limit:
        answer_text = truncate_to_sentences(answer_text, limit)
        record("sentences", False, f"truncated {sentences}->{limit}")
        n_truncated = True
    else:
        record("sentences", True)
        n_truncated = False

    if not answer_text:
        record("non_empty", False, "empty_after_trim")
        return {"empty_after_trim": True}, steps

    # ---- 7. freshness --------------------------------------------------- #
    max_age = float(SETTINGS.get("freshness.max_age_days", 30) or 30)
    cited = [c for c in chunks if c.chunk.url in kept]
    newest = _newest_fetched_at(cited) or _newest_fetched_at(chunks)
    age = _age_days(newest)
    stale = age is not None and age > max_age
    record("freshness", not stale,
           f"age_days={age:.1f}" if age is not None else "unknown_timestamp")
    emit("validate", step="freshness", max_age_days=max_age, fresh=not stale)

    return {
        "answer_text": answer_text,
        "sources": kept,
        "last_updated": newest,
        "stale": stale,
        "truncated": n_truncated,
        "dropped_citations": len(bad),
        "sentence_count": count_sentences(answer_text),
    }, steps


def validate(
    raw: str,
    chunks: Sequence[RetrievedChunk],
    question: str = "",
) -> Validation:
    """The full ordered guardrail pass. Returns the final user-facing answer.

    This is the single authority on what a user may see. The orchestrator's job
    is to hand it the raw model string and honour the result; it must not
    re-implement or reorder any of these checks.
    """
    result, steps = run_steps(raw, chunks)
    failed = next((s for s in steps if not s.ok), None)

    def finish(kind: str, text: str, sources: list[str], last_updated: str,
               reason: str) -> Validation:
        _log(failed.step if failed else "done", question, failed is None,
             kind=kind, cites=len(sources),
             abstained=kind == "abstain", refused=kind == "refusal",
             perf_redirected=kind == "perf_redirect",
             repaired=bool(result.get("truncated") or result.get("dropped_citations")),
             sentences=count_sentences(text) if kind == "factual" else 0)
        return Validation(
            answer=Answer(
                text=text,
                sources=sources,
                last_updated=last_updated or _newest_fetched_at(chunks),
                kind=kind,  # type: ignore[arg-type]
                reason=reason,
                debug=list(chunks),
            ),
            steps=list(steps),
        )

    if not result:
        # Step 1 failed: we never had an answer to guard.
        return finish("error", disclaimers.ERROR_GENERIC, [],
                      _newest_fetched_at(chunks), "generation_parse_failed")

    if "pii_categories" in result:
        return finish("pii_notice", disclaimers.PII_NOTICE, [],
                      _newest_fetched_at(chunks),
                      f"output_pii:{','.join(result['pii_categories'])}")

    if "c4_violation" in result:
        # A pointer to the source page is honest and useful here; the redirect
        # text makes no performance claim of its own.
        pointer = [chunks[0].chunk.url] if chunks and chunks[0].chunk.url else []
        return finish("perf_redirect", disclaimers.PERF_REDIRECT, pointer,
                      _newest_fetched_at(chunks),
                      f"c4_block:{result['c4_violation']}")

    if result.get("model_refused"):
        return finish("refusal", disclaimers.REFUSAL, [],
                      _newest_fetched_at(chunks), "model_refused")

    if result.get("no_citation") or result.get("no_provenance"):
        reason = (ABSENT_REASON if result.get("no_citation")
                  else f"citation_provenance_dropped:{result.get('dropped', 0)}")
        return finish("abstain", ABSTAIN_TEXT, [],
                      _newest_fetched_at(chunks), reason)

    if result.get("empty_after_trim"):
        return finish("abstain", ABSTAIN_TEXT, [],
                      _newest_fetched_at(chunks), "empty_answer")

    text = result["answer_text"]
    sources = result["sources"]
    if result.get("stale"):
        # Hard to spend one of three sentences on a staleness note, but C7
        # requires saying so. Trim the body to make room rather than break the
        # <=3-sentence invariant.
        budget = int(SETTINGS.get("generation.max_sentences", MAX_SENTENCES)
                     or MAX_SENTENCES) - 1
        text = truncate_to_sentences(text, budget)
        text = f"{text} {disclaimers.FRESHNESS_CAVEAT}".strip()

    return finish("factual", text, sources, result["last_updated"],
                  "stale_source" if result.get("stale") else "ok")


__all__ = [
    "validate", "run_steps", "Validation", "StepResult", "MAX_SENTENCES",
    "count_sentences", "truncate_to_sentences", "strip_markdown", "violates_c4",
    "_violates_return", "_RETURN_CLAIM_PATTERNS",
]


def violates_c4(answer: str) -> bool:
    """Public helper: does this answer text break C4? Used by tests and the CLI."""
    return _violates_return(answer) is not None
