"""The query pipeline: Stage 4 -> Stage 5 -> Stage 6, in that order (architecture.md Â§5).

This module is *sequencing only*. It decides what runs next and how to present
the result; it makes no judgement of its own about content. Every policy
decision belongs to a guard (``intent``, ``pii``, ``injection``) or to
:mod:`app.pipeline.validators`, which is what makes each of those independently
testable (C10).

The order below is the safety order, and each step exists to stop a specific
failure before it can happen:

1. **PII on the input, before anything else.** The LLM is never called with an
   identifier, not even to be told to ignore it. No retrieval either â€” there is
   no reason to spend a query on a PAN.
2. **Intent gate, before retrieval.** A performance or advice question is
   refused without a vector search or a prompt, and an out-of-corpus question
   abstains without a prompt.
3. **Retrieval.** Empty result -> abstain. The score floor already guarantees
   nothing weak reaches this point.
4. **Injection filter on retrieved text.** Hostile chunks are dropped before
   they are ever placed between the ``<<<SOURCE>>>`` markers.
5. **Generation.** Parsed JSON with sanitised citations; one repair retry.
6. **Ordered validators.** The final word. The orchestrator may only *downgrade*
   an answer, never upgrade or edit one.

Nothing here logs the question or the answer text â€” only ``q_hash`` and counts
(C3/NFR-08).
"""

from __future__ import annotations

from typing import Any

from ..disclaimers import ERROR_GENERIC, PERF_REDIRECT, PII_NOTICE, REFUSAL
from ..guards import injection, intent as intent_mod, pii
from ..llm.base import LLMError, ParseError
from ..logging_utils import emit, q_hash
from ..models import Answer, RetrievedChunk
from . import retriever
from .generator import generate
from .validators import ABSTAIN_TEXT, count_sentences, validate

ABSTAIN_REASON_NO_RETRIEVAL = "no_chunk_above_floor"


def _answer(
    kind: str,
    text: str,
    *,
    reason: str,
    debug: list[RetrievedChunk] | None = None,
    last_updated: str = "",
) -> Answer:
    return Answer(
        text=text,
        sources=[],
        last_updated=last_updated or (debug[0].chunk.fetched_at if debug else ""),
        kind=kind,  # type: ignore[arg-type]
        reason=reason,
        debug=list(debug or []),
    )


def answer_question(
    question: str,
    *,
    llm: Any | None = None,
    scheme_slug: str | None = None,
    skip_intent: bool = False,
) -> Answer:
    """Run one question end to end and return a user-safe :class:`Answer`.

    ``skip_intent=True`` exists for the eval harness, which needs to measure
    *retrieval and generation* quality on questions the intent gate would refuse
    for policy reasons. It is never set by the CLI or the UI.
    """
    qh = q_hash(question)
    text_in = (question or "").strip()

    # ---- 1. input PII: block before retrieval, before the LLM -------------- #
    hits = pii.scan(text_in)
    if hits:
        pii.note_block()
        emit("answer", q_hash=qh, kind="pii_notice", refused=True,
             pii_categories=",".join(hits),
             pii_blocks_total=pii.blocks_total())
        return _answer("pii_notice", PII_NOTICE,
                       reason=f"input_pii:{','.join(hits)}")

    if not text_in:
        return _answer("abstain", ABSTAIN_TEXT, reason="empty_input")

    # ---- 2. intent gate: no search, no prompt, for out-of-scope questions -- #
    if not skip_intent:
        decision = intent_mod.classify(text_in)
        if decision.intent == "refusal":
            emit("answer", q_hash=qh, kind="refusal", intent=decision.label, refused=True)
            return _answer("refusal", REFUSAL,
                           reason=f"intent:{decision.matched}")
        if decision.intent == "perf_redirect":
            emit("answer", q_hash=qh, kind="perf_redirect", intent=decision.label,
                 perf_redirected=True)
            return _answer("perf_redirect", PERF_REDIRECT,
                           reason=f"intent:{decision.matched}")
        if decision.intent == "abstain":
            emit("answer", q_hash=qh, kind="abstain", intent=decision.label,
                 abstained=True)
            return _answer("abstain", ABSTAIN_TEXT,
                           reason=f"intent:{decision.matched}")

    # ---- 3. retrieval ---------------------------------------------------- #
    if scheme_slug is None:
        scheme_slug = retriever.detect_scheme_slug(text_in)
    try:
        retrieved = retriever.search(text_in, scheme_slug=scheme_slug)
    except retriever.RetrievalError as exc:
        emit("answer", q_hash=qh, kind="error", refused=False)
        return _answer("error", str(exc), reason="retrieval_error")

    if not retrieved:
        emit("answer", q_hash=qh, kind="abstain", abstained=True, n_kept=0)
        return _answer("abstain", ABSTAIN_TEXT,
                       reason=ABSTAIN_REASON_NO_RETRIEVAL, debug=retrieved)

    # ---- 4. injection filter on retrieved text --------------------------- #
    safe, dropped = injection.filter_chunks(retrieved)
    if dropped:
        emit("answer", q_hash=qh, injection_dropped=len(dropped), n_kept=len(safe))
    if not safe:
        emit("answer", q_hash=qh, kind="abstain", abstained=True)
        return _answer("abstain", ABSTAIN_TEXT,
                       reason="all_chunks_dropped_by_injection_guard", debug=retrieved)

    # ---- 5. generation --------------------------------------------------- #
    try:
        generation = generate(text_in, safe, client=llm)
    except ParseError:
        # Unparseable after one repair retry. Surface it as an error rather than
        # an abstention: the sources were fine, the model was not.
        return _answer("error", ERROR_GENERIC,
                       reason="generation_parse_failed", debug=safe)
    except LLMError as exc:
        # Keep the provider's own cause. A flat "llm_unavailable" made a dead
        # model name indistinguishable from a rate limit, so the advice shown
        # to the user was always a guess about the wrong thing.
        return _answer("error", str(exc),
                       reason=getattr(exc, "code", "llm_unavailable"),
                       debug=safe)

    emit("generate", q_hash=qh,
         backend=(getattr(llm, "name", "unknown") if llm is not None
                  else _backend_name()),
         sentences=count_sentences(generation.answer),
         cites=len(generation.sources), repaired=generation.repaired)

    # ---- 6. ordered validators: the final word --------------------------- #
    result = validate(generation.raw, safe, question=text_in)
    if result.answer.kind == "factual":
        return result.answer

    # A validated non-factual answer keeps its debug trail so --debug still works.
    return Answer(
        text=result.answer.text,
        sources=result.answer.sources,
        last_updated=result.answer.last_updated,
        kind=result.answer.kind,
        reason=result.answer.reason,
        debug=safe,
    )


# --- small helpers --------------------------------------------------------- #


def _backend_name() -> str:
    try:
        from ..llm import describe

        return describe()
    except Exception:  # noqa: BLE001 - diagnostics must not break answering
        return "unknown"


__all__ = ["answer_question", "ABSTAIN_TEXT", "ABSTAIN_REASON_NO_RETRIEVAL"]
