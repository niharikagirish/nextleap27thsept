"""``python -m app.eval_retrieval`` — measure the index you actually ship.

Why this exists separately from ``app.experiment``
--------------------------------------------------
``app.experiment`` answers "which *chunking* is best?". It re-chunks the raw
documents in memory and scores them with brute-force numpy cosine. That is the
right instrument for its question and the wrong one for this one, because it
bypasses everything the production retriever actually does:

* the persistent Chroma/HNSW index, not exact numpy search
* ``retrieval.score_floor`` - a gold chunk scoring 0.30 is dropped, and a
  perfect chunker still reads as a miss
* near-duplicate de-duplication, which can promote a worse chunk
* the ``scheme_slug`` metadata filter chosen by :func:`detect_scheme_slug`
* truncation of the context budget to ``retrieval.top_k``

A chunking sweep reporting Recall@5 = 1.000 says nothing about whether the
deployed pipeline returns that chunk. Only this does. It calls the same
:func:`~app.pipeline.retriever.search` the orchestrator calls, so a number here
is a number about the product.

Metrics
-------
``hit@k``
    A gold question counts as a hit when some top-k chunk has the expected
    ``source_id`` *and* contains the verbatim ``answer_span``. Requiring the
    source id stops a chunk from the wrong scheme that repeats the same
    boilerplate ("Minimum SIP amount ... is Rs 100") from scoring a hit.

``scheme@k``
    Relaxed: the expected scheme appears anywhere in the top k, regardless of
    whether the specific span was found. The gap between ``scheme@k`` and
    ``hit@k`` is the diagnostic that matters - a low ``scheme@k`` is a filtering
    or embedding bug, while a high ``scheme@k`` with a low ``hit@k`` means the
    right page was retrieved but split or ranked wrongly.

``MRR``
    Mean reciprocal rank of the first hit. Distinguishes "the right chunk is
    there but ranked 6th" from "the right chunk is not there at all", which
    ``hit@k`` alone cannot.

No LLM is called, so this is free and deterministic. ``--strict`` turns any
imperfect score into a non-zero exit code for use in a check.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import enable_utf8_console
from .config import ROOT
from .pipeline.retriever import RetrievalError, detect_scheme_slug, search

DEFAULT_GOLD = ROOT / "eval" / "golden.json"
DEFAULT_OUT = ROOT / "docs" / "retrieval_eval.md"


@dataclass
class GoldQuestion:
    """One row of ``eval/golden.json``."""

    id: str
    question: str
    fact_type: str
    expected_source_id: str
    answer_span: str
    must_include: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    """Per-question result, including *why* it missed."""

    gold: GoldQuestion
    hit_rank: int | None
    scheme_rank: int | None
    n_returned: int
    top_score: float | None
    detected_scheme: str | None
    scheme_mismatch: bool
    failure: str | None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.hit_rank is None else 1.0 / self.hit_rank


def load_gold(path: Path) -> list[GoldQuestion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldQuestion(
            id=row["id"],
            question=row["question"],
            fact_type=row.get("fact_type", ""),
            expected_source_id=row["expected_source_id"],
            answer_span=row["answer_span"],
            must_include=list(row.get("must_include") or []),
        )
        for row in payload["questions"]
    ]


def _excerpt(text: str, width: int = 90) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "..."


def evaluate_question(
    gold: GoldQuestion,
    *,
    top_k: int | None = None,
    n_results: int | None = None,
    scheme_filter: bool = True,
) -> Outcome:
    """Run one question through the production retriever and score it.

    Mirrors the orchestrator's own call sequence - detect the scheme, then
    search with it - rather than inventing a retrieval path of its own.
    """
    detected = detect_scheme_slug(gold.question) if scheme_filter else None
    results = search(gold.question, n_results=n_results, top_k=top_k,
                     scheme_slug=detected)

    hit_rank: int | None = None
    scheme_rank: int | None = None
    for rc in results:
        if scheme_rank is None and rc.chunk.source_id == gold.expected_source_id:
            scheme_rank = rc.rank
        if (hit_rank is None
                and rc.chunk.source_id == gold.expected_source_id
                and gold.answer_span in rc.chunk.text):
            hit_rank = rc.rank

    mismatch = detected is not None and detected != gold.expected_source_id
    if hit_rank is not None:
        failure = None
    elif mismatch:
        # The filter itself chose the wrong scheme: retrieval never had a chance.
        failure = f"scheme filter chose {detected!r}, expected {gold.expected_source_id!r}"
    elif not results:
        failure = "nothing cleared the score floor"
    elif scheme_rank is None:
        failure = "expected scheme absent from results"
    else:
        failure = f"right scheme at rank {scheme_rank}, but answer_span not in any chunk"

    return Outcome(
        gold=gold,
        hit_rank=hit_rank,
        scheme_rank=scheme_rank,
        n_returned=len(results),
        top_score=results[0].score if results else None,
        detected_scheme=detected,
        scheme_mismatch=bool(mismatch),
        failure=failure,
    )


def _at_k(outcomes: Sequence[Outcome], k: int, attr: str) -> float:
    if not outcomes:
        return 0.0
    hits = sum(1 for o in outcomes if getattr(o, attr) is not None
               and getattr(o, attr) <= k)
    return hits / len(outcomes)


def render(
    outcomes: Sequence[Outcome],
    ks: Sequence[int],
    *,
    gold_path: Path,
    scheme_filter: bool,
) -> str:
    n = len(outcomes)
    max_k = max(ks) if ks else 0
    lines: list[str] = []
    add = lines.append

    add("# Retrieval evaluation (live index)")
    add("")
    add("Generated by `python -m app.eval_retrieval`. Do not edit by hand.")
    add("")
    add(f"- Gold set: `{gold_path.name}`, {n} questions")
    add("- Index: the persistent Chroma collection, via "
        "`app.pipeline.retriever.search`")
    add("- Includes: score floor, near-duplicate de-duplication, context "
        "truncation to `retrieval.top_k`")
    add(f"- Scheme filter: {'auto-detected per question' if scheme_filter else 'DISABLED'}")
    add("- Metric: hit = top-k chunk from the correct `source_id` contains the "
        "verbatim `answer_span`")
    add("- No LLM is called, so this measures retrieval alone.")
    add("")

    add("## Scores")
    add("")
    add("| metric | value |")
    add("|--------|------:|")
    for k in ks:
        add(f"| hit@{k} | {_at_k(outcomes, k, 'hit_rank'):.3f} |")
    for k in ks:
        add(f"| scheme@{k} | {_at_k(outcomes, k, 'scheme_rank'):.3f} |")
    mrr = sum(o.reciprocal_rank for o in outcomes) / n if n else 0.0
    add(f"| MRR | {mrr:.3f} |")
    add(f"| exact hit @1 | {sum(1 for o in outcomes if o.hit_rank == 1)}/{n} |")
    add("")

    add("## Reading these numbers")
    add("")
    add("- **`scheme@k` vs `hit@k` is the diagnostic.** If `scheme@k` is high and")
    add("  `hit@k` is low, the right page was retrieved but the answer was split")
    add("  across chunks or ranked too low — a chunking problem, not an embedding")
    add("  problem. If `scheme@k` is low too, the scheme filter or the query")
    add("  itself is at fault.")
    add("- **Misses listed below are not automatically bugs.** A gold")
    add("  `answer_span` is a verbatim string; if a chunker split that sentence,")
    add("  the span genuinely is not in any single chunk, and no retriever can")
    add("  find it. That is a gold-set or chunking-boundary artefact.")
    add("")

    add("## Per-question results")
    add("")
    add("| id | fact_type | scheme | hit@rank | top | result |")
    add("|----|-----------|--------|----------:|----:|--------|")
    for o in outcomes:
        add(f"| {o.gold.id} | {o.gold.fact_type} | {o.gold.expected_source_id} "
            f"| {o.hit_rank if o.hit_rank else '-'} "
            f"| {o.top_score:.3f} | {'hit' if o.hit_rank else 'MISS'} |"
            if o.top_score is not None else
            f"| {o.gold.id} | {o.gold.fact_type} | {o.gold.expected_source_id} "
            f"| - | - | MISS |")
    add("")

    misses = [o for o in outcomes if o.hit_rank is None]
    if misses:
        add("## Misses")
        add("")
        for o in misses:
            add(f"### {o.gold.id} — {o.gold.question}")
            add("")
            add(f"- fact_type: `{o.gold.fact_type}`")
            add(f"- expected source: `{o.gold.expected_source_id}`")
            add(f"- detected scheme: `{o.detected_scheme or 'none'}`")
            add(f"- chunks returned: {o.n_returned}")
            if o.top_score is not None:
                add(f"- top score: {o.top_score:.3f}")
            add(f"- diagnosis: **{o.failure}**")
            add(f"- wanted span: `{_excerpt(o.gold.answer_span, 120)}`")
            add("")
    else:
        add("## Misses")
        add("")
        add("None. Every gold `answer_span` was retrieved.")
        add("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    enable_utf8_console()

    parser = argparse.ArgumentParser(
        prog="python -m app.eval_retrieval",
        description="Score the live Chroma index against eval/golden.json. "
                    "No LLM is called.",
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD,
                        help="gold set JSON (default: eval/golden.json)")
    parser.add_argument("--ks", type=int, nargs="*", default=[1, 3, 5, 6],
                        help="cutoffs to report (default: 1 3 5 6)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="override retrieval.top_k for every question")
    parser.add_argument("--n-results", type=int, default=None,
                        help="override retrieval.n_results")
    parser.add_argument("--no-scheme-filter", dest="scheme_filter",
                        action="store_false",
                        help="disable scheme auto-detection, to measure what the "
                             "filter is contributing")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="write the markdown report here")
    parser.add_argument("--no-write", action="store_true",
                        help="print only, do not write the report")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero unless every question hits at the "
                             "largest k (for CI)")
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    ks = sorted({k for k in args.ks if k > 0})

    outcomes: list[Outcome] = []
    try:
        for item in gold:
            outcomes.append(
                evaluate_question(
                    item,
                    top_k=args.top_k,
                    n_results=args.n_results,
                    scheme_filter=args.scheme_filter,
                )
            )
    except RetrievalError as exc:
        print(f"RETRIEVAL FAILED: {exc}", file=sys.stderr)
        return 1

    report = render(outcomes, ks, gold_path=args.gold,
                    scheme_filter=args.scheme_filter)
    print(report)

    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nreport written to {args.out}")

    if args.strict and any(o.hit_rank is None or o.hit_rank > max(ks) for o in outcomes):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
