"""``python -m app.chat "question"`` — the grounded answer CLI (FR-13, FR-07).

One question, one answer, one exit code. Everything the UI shows is here:
the answer text, the source chips, and the ``last updated`` line, assembled from
``Answer`` alone. The UI imports this module's rendering helpers rather than
re-implementing them, so a change to what a user sees lands in both surfaces at
once (the same reasoning as C9 and ``app/disclaimers.py``).

    python -m app.chat "What is the exit load of HDFC Large Cap Fund?"
    python -m app.chat "What is the 1 year return?"          # perf redirect
    python -m app.chat "Should I buy HDFC Small Cap Fund?"   # refusal
    python -m app.chat "My PAN is ABCDE1234F"                # PII notice
    python -m app.chat "What is the expense ratio?" --debug  # + ranked chunks

Exit codes: ``0`` a factual or directed answer, ``1`` an abstention, ``2`` an
error. Scripts can branch on whether the bot actually answered, which
``retrieval-only mode`` and the eval harness both need.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import enable_utf8_console
from .disclaimers import (
    FULL as FULL_DISCLAIMER,
    LAST_UPDATED_LABEL,
    SCOPE_SUMMARY,
    SHORT,
)
from .models import Answer
from .pipeline.orchestrator import answer_question

EXIT_OK = 0
EXIT_ABSTAINED = 1
EXIT_ERROR = 2


ERROR_HEADLINE = "The assistant could not reach the language model."

# What to actually do about each cause. Keyed by LLMError.code; the old flat
# "llm_unavailable" forced one guess (check your key) onto every failure, which
# is actively misleading when the key is fine and the model id is dead.
ERROR_ADVICE: dict[str, str] = {
    "llm_unreachable": "Could not reach the provider at all. Check network access.",
    "llm_auth_failed": "The API key was rejected. Check LLM_API_KEY in .env.",
    "llm_rate_limited": "The free tier is throttling. Wait a few seconds and "
                        "retry - the key and model are fine.",
    "llm_model_not_found": "The provider does not recognise that model. Set a "
                           "current model id in GROQ_MODEL / GEMINI_MODEL.",
    "llm_bad_request": "The provider rejected the request. Re-run with --debug.",
    "llm_server_error": "The provider returned a server error. Retry shortly.",
    "llm_bad_response": "The provider replied in an unexpected shape. Re-run "
                        "with --debug, or switch model.",
    "llm_unsupported": "That model does not return text at this endpoint. "
                       "Pick a chat-completions model.",
    "llm_unavailable": "Re-run with --debug for the provider error, or check "
                       "the LLM_API_KEY / GROQ_MODEL values in .env.",
}


def render(answer: Answer, debug: bool = False) -> str:
    """Format one answer for a terminal."""
    lines: list[str] = []
    add = lines.append

    if answer.kind == "error":
        # An infrastructure failure must never be dressed up as an answer.
        # Rendering it through the normal path emitted the provider's raw 404
        # body followed by "Last updated from sources" and the facts-only
        # disclaimer, so a dead API key looked like a sourced, authoritative
        # reply. The technical detail is kept, but labelled and gated behind
        # --debug so the default output says what actually happened.
        reason = answer.reason or "llm_unavailable"
        add(ERROR_HEADLINE)
        add("")
        add(f"Reason: {reason}")
        add("")
        advice = ERROR_ADVICE.get(reason)
        if advice is None:
            # An unrecognised code means the provider layer gained a cause the
            # CLI has not learned yet. Show it verbatim rather than guessing.
            advice = f"({reason}) Re-run with --debug for detail."
        add(advice)
        if debug:
            add("")
            add("Technical detail (--debug):")
            add(f"  {answer.text}")
        return "\n".join(lines)

    add(answer.text)
    add("")

    if answer.sources:
        add("Sources:")
        for url in answer.sources:
            add(f"  - {url}")
        add("")

    if answer.last_updated:
        add(f"{LAST_UPDATED_LABEL} {answer.last_updated}")
        add("")

    add(SHORT)
    if debug and answer.debug:
        add("")
        add("Retrieved chunks:")
        for rc in answer.debug:
            trail = " > ".join(rc.chunk.heading_trail)
            excerpt = " ".join(rc.chunk.text.split())[:100]
            add(f"  #{rc.rank} score={rc.score:.3f} {rc.chunk.source_id} | {trail}")
            add(f"      {excerpt}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    enable_utf8_console()
    parser = argparse.ArgumentParser(
        prog="python -m app.chat",
        description="Ask a grounded question about the 5 allowlisted HDFC schemes.",
    )
    parser.add_argument("question", nargs="+", help="the question to ask")
    parser.add_argument("--debug", action="store_true",
                        help="print the ranked retrieved chunks after the answer")
    parser.add_argument("--scheme", default=None,
                        help="force a scheme_slug filter (hdfc-large-cap, ...)")
    parser.add_argument("--skip-intent", action="store_true",
                        help=argparse.SUPPRESS)  # eval harness only
    parser.add_argument("--disclaimer", action="store_true",
                        help="print the full disclaimer and exit")
    args = parser.parse_args(argv)

    if args.disclaimer:
        print(FULL_DISCLAIMER)
        print()
        print(SCOPE_SUMMARY)
        return EXIT_OK

    question = " ".join(args.question)
    answer = answer_question(
        question,
        scheme_slug=args.scheme,
        skip_intent=args.skip_intent,
    )
    print()
    print(render(answer, debug=args.debug))

    if answer.kind == "error":
        return EXIT_ERROR
    if answer.kind == "abstain":
        return EXIT_ABSTAINED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
