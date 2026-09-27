"""Prompt-injection filtering for retrieved text (FR-12).

The threat is specific, and worth stating precisely: an attacker who controls a
page in the corpus (a scraped mirror, a user-generated field on the source site,
a page fetched after a compromise) puts text in the page that reads like an
instruction. The RAG system then faithfully retrieves it and the model, having
been told to follow the sources, obeys it.

Two layers address this, and neither is sufficient alone:

1. **This module — drop the hostile chunk before it reaches the prompt.** The
   cheapest possible layer: a dropped chunk cannot be obeyed, and the score floor
   plus dedupe mean a lost chunk degrades one answer rather than breaking it.
2. **The framing in :mod:`app.pipeline.prompt`** — every block is delimited by
   ``<<<SOURCE ... >>>`` / ``<<<END SOURCE>>>`` and the system prompt states the
   marked region is data, never instructions. This is what defends against
   injection that does *not* look like an injection, e.g. a subtly-worded
   sentence that reads plausibly as a fact.

**Precision matters more than recall here.** A fund page is full of text that
*contains* instruction-adjacent words — "systematic withdrawal plan", "riskometer
category", "exit load instructions", "steps to invest". Matching the bare word
"instructions" would delete the exit-load chunk, which is one of the most
important facts in the corpus. So every pattern below requires an *imperative
jailbreak shape*, not just a keyword. When in doubt, keep the chunk and let the
prompt framing do the work; a false negative costs one answer, a false positive
costs a correct fact.

Test with ``eval/ooc_probes.json``.
"""

from __future__ import annotations

import re
from typing import Sequence

# Each pattern is an explicit instruction-overriding attempt. Multi-word
# "ignore the previous instructions" shapes only — never the bare word "ignore".
_JAILBREAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override_prior",
     re.compile(r"(?i)\b(?:ignore|disregard|forget|discard)\b[^.!?\n]{0,40}?"
                r"\b(?:previous|prior|above|earlier|preceding|all|any)\b"
                r"[^.!?\n]{0,20}?\b(?:instruction|prompt|rule|direction|message)s?\b")),
    ("override_output",
     re.compile(r"(?i)\b(?:ignore|disregard|forget)\b[^.!?\n]{0,30}?"
                r"\b(?:everything|all)\b[^.!?\n]{0,20}?\byou(?:'ve| have)? been told\b")),
    ("reveal_system",
     re.compile(r"(?i)\b(?:reveal|show|print|repeat|output|display)\b[^.!?\n]{0,30}?"
                r"\b(?:system|initial|original|hidden)\b[^.!?\n]{0,10}?"
                r"\b(?:prompt|instructions?|message)\b")),
    ("role_reassign",
     re.compile(r"(?i)\byou\s+are\s+now\b|\bnew\s+(?:system\s+)?"
                r"(?:instruction|persona|role)s?\b|\bact\s+as\s+(?:a|an|the)\b")),
    ("jailbreak_persona",
     re.compile(r"(?i)\b(?:developer\s+mode|dan\s+mode|jailbreak|"
                r"do\s+anything\s+now)\b")),
    ("fake_turn_marker",
     re.compile(r"(?i)(?:<\|?\s*(?:im_start|im_end|system|endoftext)\s*\|?>|"
                r"\[/?INST\]|<<<\s*SYS\s*>>>|\b###\s*(?:instruction|system)\s*:)")),
    ("prompt_leak_request",
     re.compile(r"(?i)\bwhat\s+(?:were|are)\s+your\s+(?:original\s+)?"
                r"(?:instructions?|system\s+prompt|guidelines)\b")),
    ("citation_spoof",
     re.compile(r"(?i)\b(?:cite|source|according\s+to)\s*:\s*https?://")),
)

# Patterns that must NOT trip the filter. Asserted by tests, and documented here
# because these are the mistakes this module is most likely to grow into.
BENIGN_EQUIVALENCES: tuple[str, ...] = (
    "Follow the steps to invest in the fund",
    "Exit load instructions are given on the scheme page",
    "Systematic withdrawal plan is available",
    "Read the riskometer category before investing",
    "The fund manager's investment philosophy",
)


def scan(text: str) -> list[str]:
    """Return the *names* of jailbreak patterns found. Never the matched text."""
    if not text:
        return []
    found: list[str] = []
    for name, pattern in _JAILBREAK_PATTERNS:
        if pattern.search(text) and name not in found:
            found.append(name)
    return found


def contains_injection(text: str) -> bool:
    return bool(scan(text))


def is_benign(text: str) -> bool:
    """True when the text looks like ordinary fund prose.

    Only used by tests, to keep the filter's precision honest as patterns are
    added over time.
    """
    return not contains_injection(text)


def filter_chunks(chunks: Sequence) -> tuple[list, list]:
    """Split ``chunks`` into ``(safe, dropped)`` by injection scan.

    Dropped chunks are reported by count and reason only — never by echoing
    their text into a log line (C3 applies to injected content too, and a
    hostile chunk is exactly the content you would least like written to disk).
    """
    safe, dropped = [], []
    for rc in chunks:
        if contains_injection(rc.chunk.text):
            dropped.append(rc)
        else:
            safe.append(rc)
    return safe, dropped


def pattern_names() -> list[str]:
    return [name for name, _ in _JAILBREAK_PATTERNS]


__all__ = [
    "scan", "contains_injection", "is_benign", "filter_chunks", "pattern_names",
    "BENIGN_EQUIVALENCES", "_JAILBREAK_PATTERNS",
]
