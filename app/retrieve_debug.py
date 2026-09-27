"""``python -m app.retrieve_debug "question"`` — retrieval quality, no LLM.

This is how retrieval gets judged *in isolation* from generation (Phase 4's
whole point). If an answer is wrong, this tells you whether the chunk was never
retrieved or whether the model misused a chunk it did get — and those are very
different bugs.

Prints a ranked table of ``rank / score / source_id / heading_trail / excerpt``
and never calls an LLM. Queries are hashed for the log; the question itself is
only ever printed to this terminal, never written to ``logs/run.jsonl``.

    python -m app.retrieve_debug "What is the expense ratio of the HDFC Large Cap Fund?"
    python -m app.retrieve_debug "What is today's NAV?"      # expected: nothing clears the floor
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import enable_utf8_console
from .pipeline.retriever import RetrievalError, search


def _excerpt(text: str, width: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def render(results: Sequence, question: str, floor: float,
           scheme_slug: str | None = None,
           detected: bool = False) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"Q: {question}")
    if scheme_slug:
        how = "auto-detected" if detected else "forced"
        add(f"   scheme filter: {scheme_slug} ({how})")
    else:
        add("   scheme filter: none - all five schemes searched")
    add("")
    if not results:
        add(f"No chunk cleared the score floor ({floor}).")
        add("This is the CORRECT outcome for an out-of-corpus or performance")
        add("question — the orchestrator will abstain rather than guess.")
        return "\n".join(lines)


    add(f"{'#':>2}  {'score':>6}  {'source_id':<18} heading_trail")
    add("-" * 100)
    for rc in results:
        trail = " > ".join(rc.chunk.heading_trail)
        add(f"{rc.rank:>2}  {rc.score:>6.3f}  {rc.chunk.source_id:<18} {trail}")
        add(f"{'':>2}  {'':>6}  {_excerpt(rc.chunk.text)}")
        add("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    enable_utf8_console()

    parser = argparse.ArgumentParser(
        prog="python -m app.retrieve_debug",
        description="Print ranked retrieved chunks with scores. No LLM is called.",
    )
    parser.add_argument("question", nargs="+", help="the question to retrieve for")
    parser.add_argument("--n-results", type=int, default=None,
                        help="override retrieval.n_results (default 12)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="override retrieval.top_k (default 6)")
    parser.add_argument("--scheme", default=None,
                        help="force a scheme_slug metadata filter; overrides "
                             "auto-detection")
    parser.add_argument("--no-scheme-filter", dest="scheme_filter",
                        action="store_false",
                        help="skip scheme auto-detection and search all five "
                             "schemes. Matches no production path - use it only "
                             "to see what the filter is hiding")
    args = parser.parse_args(argv)
    parser.set_defaults(scheme_filter=True)

    question = " ".join(args.question)

    from .config import SETTINGS

    floor = float(SETTINGS.get("retrieval.score_floor", 0.35))

    # Default to the scheme the orchestrator would pick, so this tool measures
    # production retrieval rather than a stricter-than-real one. Without it,
    # "What is the exit load of HDFC Small Cap?" returns two HDFC Large Cap
    # chunks and burns a third of the context budget on a question that named
    # its scheme - a problem you would spend a day chasing that the live app
    # does not have.
    from .pipeline import retriever

    if args.scheme is not None:
        scheme_slug = args.scheme
        detected = False
    elif not args.scheme_filter:
        scheme_slug = None
        detected = False
    else:
        scheme_slug = retriever.detect_scheme_slug(question)
        detected = scheme_slug is not None

    try:
        results = search(
            question,
            n_results=args.n_results,
            top_k=args.top_k,
            scheme_slug=scheme_slug,
        )
    except RetrievalError as exc:
        print(f"RETRIEVAL FAILED: {exc}", file=sys.stderr)
        return 1

    print(render(results, question, floor, scheme_slug, detected))
    if results:
        wrong = [rc for rc in results if scheme_slug and rc.chunk.source_id != scheme_slug]
        note = f"  [{len(wrong)} from other schemes]" if wrong else ""
        print(f"{len(results)} chunk(s) cleared floor={floor}; "
              f"top_score={results[0].score:.3f}{note}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
