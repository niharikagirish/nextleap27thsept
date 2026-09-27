"""``echo`` backend — no LLM at all (architecture.md ADR A9).

Returns the top retrieved chunk verbatim, clearly labelled as retrieval-only
mode. This exists for one reason: **the demo must never die because a key is
missing, expired, or rate-limited.** With this backend the whole pipeline —
retrieval, guardrails, citation repair, sentence limits, the UI — is runnable
and testable with zero credentials, so Phases 5 through 8 can be completed and
demonstrable on a machine that has never seen an API key.

It is also genuinely useful for debugging: when an answer is wrong, seeing the
raw passage that was retrieved tells you immediately whether the fault is
retrieval or generation.

``app.llm.get_llm()`` prints a one-line notice when it selects this backend, so
nobody mistakes its output for a real generated answer.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .base import LLMError

# The orchestrator hands the echo backend the already-built messages, so it pulls
# the source blocks back out of them. This is a retrieval echo, not an
# interpretation: no text is generated, reordered, or paraphrased.
#
# Every attribute is matched as a quoted string rather than with a loose
# ``[^>]*`` tail. The heading is a " > "-joined trail, so the open tag routinely
# contains a ">" *inside* a quoted value:
#
#     <<<SOURCE url="..." heading="HDFC Large Cap Fund > Exit load">>>
#
# A ``[^>]*`` tail stops at that ">", never reaches the closing ">>>", and the
# block silently fails to match - which then surfaces as "retrieval returned
# nothing" on a request that had retrieved something. Keep the attribute names
# in step with prompt.SOURCE_OPEN.
_SOURCE_BLOCK = re.compile(
    r'<<<SOURCE\s+url="(?P<url>[^"]*)"\s+scheme="(?P<scheme>[^"]*)"\s+'
    r'fetched_at="(?P<fetched_at>[^"]*)"\s+heading="(?P<heading>[^"]*)">>>'
    r'\s*(?P<text>.*?)\s*<<<END SOURCE>>>',
    re.DOTALL,
)


class EchoClient:
    """Retrieval-only stub. See module docstring."""

    name = "echo"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = int(max_sentences)

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 250,
    ) -> str:
        user_text = ""
        for message in messages:
            if message.get("role") == "user":
                user_text = str(message.get("content", ""))
        if not user_text:
            raise LLMError("echo backend received no user message")

        blocks = list(_SOURCE_BLOCK.finditer(user_text))
        if not blocks:
            raise LLMError(
                "echo backend found no <<<SOURCE ... >>> blocks in the prompt; "
                "retrieval returned nothing, so there is nothing to echo."
            )

        first = blocks[0]
        url = first.group("url")
        sentences = _first_sentences(first.group("text"), self.max_sentences)

        # Returned in the same JSON shape the real backends must produce, so the
        # generator and validators exercise exactly the same code path.
        return (
            '{"answer": "[retrieval-only mode] ' + _escape(sentences) + '", '
            '"sources": ["' + _escape(url) + '"], '
            '"refused": false, "reason": "echo backend: no LLM configured"}'
        )


def _first_sentences(text: str, n: int) -> str:
    from ..pipeline.validators import count_sentences, truncate_to_sentences

    flat = " ".join(text.split())
    if count_sentences(flat) <= n:
        return flat
    return truncate_to_sentences(flat, n)


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


__all__ = ["EchoClient"]
