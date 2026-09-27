"""PII detection and blocking (constraint C3, FR-10, FR-11).

This is the one module where the *output* matters more than the behaviour, so
the design rule is blunt:

    **Never log, echo, return, or store the matched value. Only the category name.**

``scan`` returns ``["pan", "phone"]``, never ``"ABCDE1234F"``. The orchestrator
does not call the LLM at all when a hit occurs, and the user receives a fixed
notice with no echo of what they typed. A PII guard that reports *what* it found
is a second copy of the secret.

**Pattern order is load-bearing.** PAN and Aadhaar are resolved before any long
digit run, because the looser 10-digit rule would otherwise match the digits
inside them first and report the wrong — less specific — category. Same for
IFSC: ``HDFC0001234`` contains digits but not ten of them, while PAN
``ABCDE1234F`` does contain a 5-digit run.

**A long digit run is labelled by context, not by length.** A 10-digit number
starting 6-9 satisfies both the account and the phone shape. ``9876543210`` on
its own is reported as an account number; ``call 9999999999`` is reported as a
phone number. Either way it is blocked — only the label shown to the user
changes, and mislabelling it would send someone to the wrong document.

**False positives are accepted deliberately.** ``\\b[0-9]{10,}\\b`` will catch a
10+ digit number that is not really an account number. Over-blocking a
legitimate question is embarrassing; letting a PAN reach a third-party LLM
provider is a reportable breach. The trade is asymmetric, so it goes this way.

Detected on **input** (before the LLM) and again on **output** (before the user),
because a model can echo a number that was in its context.
"""

from __future__ import annotations

import re
from typing import Iterable

# Order matters. Specific before generic. See the module docstring.
# PAN/Aadhaar allow internal spaces and hyphens, and PAN is case-insensitive,
# because identifiers get pasted in every format the forms use: "ABCDE1234F",
# "ABC DE 1234 F", "abcde1234f". All three are the same PAN.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("pan", re.compile(
        r"(?i)\b[A-Z](?:[\s\-]?[A-Z]){4}[\s\-]?[0-9](?:[\s\-]?[0-9]){3}[\s\-]?[A-Z]\b"
    )),
    ("aadhaar", re.compile(r"\b[2-9][0-9]{3}(?:[\s\-]?[0-9]{4}){2,3}\b")),
    ("otp", re.compile(
        r"(?i)\b(?:otp|one[\s\-]?time|verification|passcode|pin)\b\D{0,20}\b[0-9]{4,6}\b"
    )),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
)

# A long digit run is either an account number or a phone number, and the two
# forms overlap on a 10-digit number starting 6-9. Length alone cannot break
# the tie: "9876543210" in bare context is an account number, while
# "call 9999999999" is plainly a phone number. So the surrounding vocabulary
# decides. Both categories block, so this only changes the label shown to a
# user, never whether they are stopped.
_ACCOUNT_RUN = re.compile(r"\b[0-9]{10,}\b")
_PHONE_RUN = re.compile(r"(?:\+?91[\s\-.]?)?\b[6-9][0-9]{4}[\s\-]?[0-9]{5}\b")

_ACCOUNT_CONTEXT = re.compile(r"(?i)\b(?:bank\s+account|account|acct|a\s*\/\s*c)\b")
_PHONE_CONTEXT = re.compile(
    r"(?i)(?:\+\s?91\b|\b(?:phone|mobile|call|contact|whats?app|tel|telephone)\b)"
)
# "number" on its own is not phone vocabulary, but a 10-digit Indian number
# introduced by it almost always is ("number 9999999999"). Restricting this to
# exactly 10 digits keeps "number 123456789012" on the account side.
_GENERIC_NUMBER = re.compile(r"(?i)\b(?:number|no\.)\b")
_CONTEXT_WINDOW = 32
_NATIONAL_DIGITS = 10


# Count-only metric (C3). Incremented on every block, never per-value.
PII_BLOCKS_TOTAL = 0


def _national_digit_count(span_text: str) -> int:
    """Digits in the national part of a number, ignoring any +91 country code."""
    core = re.sub(r"^\s*\+?91[\s\-.]?", "", span_text)
    return sum(character.isdigit() for character in core)


def _digit_label(text: str, start: int, end: int) -> str:
    """Classify one long digit run as ``account`` or ``phone`` from its context.

    An explicit account keyword wins. Then an explicit phone marker. Then a
    10-digit run introduced by a bare "number". Otherwise ``account`` — the
    safer label, because an unlabelled long digit run is more often an account
    than a phone.
    """
    before = text[max(0, start - _CONTEXT_WINDOW):start]
    if _ACCOUNT_CONTEXT.search(before):
        return "account"
    if _PHONE_CONTEXT.search(before) or _PHONE_CONTEXT.search(text[start:end]):
        return "phone"
    if (_national_digit_count(text[start:end]) == _NATIONAL_DIGITS
            and _GENERIC_NUMBER.search(before)):
        return "phone"
    return "account"


def _spans(text: str) -> list[tuple[int, int, str]]:
    """Resolve every PII span, most specific first, dropping overlaps.

    Both :func:`scan` and :func:`redact` go through this so they can never
    disagree about which value belongs to which category — a mismatch there
    would mean redacting the wrong span, or reporting a category while leaving
    the value in place.
    """
    resolved: list[tuple[int, int, str]] = []
    claimed: list[tuple[int, int]] = []

    def claim(start: int, end: int, name: str) -> None:
        if any(low <= start and end <= high for low, high in claimed):
            return  # already reported by a more specific pattern
        claimed.append((start, end))
        resolved.append((start, end, name))

    for name, pattern in PATTERNS:
        for match in pattern.finditer(text):
            claim(match.start(), match.end(), name)

    for pattern in (_ACCOUNT_RUN, _PHONE_RUN):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(low <= start and end <= high for low, high in claimed):
                continue
            claim(start, end, _digit_label(text, start, end))

    return resolved


def scan(text: str) -> list[str]:
    """Return the *category names* found in ``text``. Never the values.

    Patterns are resolved most-specific-first (email, PAN, Aadhaar, OTP, IFSC)
    and a later match is discarded when its span sits entirely inside one
    already claimed, so:

    * a PAN is claimed as ``pan`` and the digits inside it are not *also*
      reported as an account number;
    * an Aadhaar number is reported as ``aadhaar``, not as an account;
    * a long digit run is reported once, as ``account`` or ``phone`` depending
      on the words around it.

    Reporting the broadest overlapping label would tell a user their Aadhaar
    number is a "bank account", which is both useless and alarming.
    """
    if not text:
        return []

    found: list[str] = []
    for _, _, name in _spans(text):
        if name not in found:
            found.append(name)
    return found


def contains_pii(text: str) -> bool:
    """True if any category matches."""
    return bool(scan(text))


def categories(text: str) -> str:
    """Comma-joined categories, for the count-only log line. No values."""
    return ",".join(scan(text))


def note_block(count: int = 1) -> int:
    """Increment the count-only block metric and return the new total."""
    global PII_BLOCKS_TOTAL
    PII_BLOCKS_TOTAL += int(count)
    return PII_BLOCKS_TOTAL


def blocks_total() -> int:
    return PII_BLOCKS_TOTAL


def reset_counter() -> None:
    """Zero the metric. Tests only."""
    global PII_BLOCKS_TOTAL
    PII_BLOCKS_TOTAL = 0


def redact(text: str) -> tuple[str, list[str]]:
    """Replace every matched value with a ``[REDACTED:<category>]`` marker.

    Used on the *output* path, where a model may have echoed an identifier back.
    The category is preserved so the notice can say what was removed without
    repeating it.
    """
    if not text:
        return "", []
    spans = _spans(text)
    hits: list[str] = []
    for _, _, name in spans:
        if name not in hits:
            hits.append(name)
    # Right to left, so earlier offsets stay valid as the string is rewritten.
    redacted = text
    for start, end, name in reversed(spans):
        redacted = redacted[:start] + f"[REDACTED:{name}]" + redacted[end:]
    return redacted, hits


def benign_finance_question(text: str) -> bool:
    """True when ``text`` is a fund-facts question with no identifier in it.

    Used by the test suite to assert the guard does not over-block ordinary
    questions, and by the orchestrator to keep ``Rs 500``-style amounts working:
    a 3-digit amount must never trip the 10-digit account rule.
    """
    return not contains_pii(text)


def describe_categories(names: Iterable[str]) -> str:
    """Human-readable category list for the user notice."""
    friendly = {
        "email": "email address",
        "pan": "PAN",
        "aadhaar": "Aadhaar number",
        "account": "account number",
        "phone": "phone number",
        "otp": "OTP or PIN",
        "ifsc": "IFSC code",
    }
    return ", ".join(friendly.get(name, name) for name in names)


__all__ = [
    "PATTERNS", "scan", "contains_pii", "categories", "redact", "note_block",
    "blocks_total", "reset_counter", "benign_finance_question",
    "describe_categories",
]
