"""STAGE 5b — generation (architecture.md §5.4).

The backend is a black box that returns a *string*. Making it also return valid
JSON is a wish, not a contract, so the real job of this module is turning
probable model behaviour into a dependable result:

* **Tolerate, then insist.** Common decorations (```json fences, a sentence of
  preamble before the object, trailing commentary) are stripped rather than
  rejected, because rejecting them would mean abstaining on answers that were
  perfectly good.
* **Retry exactly once**, and the retry says what went wrong. Small models fail
  in consistent, correctable ways — an unquoted value, a missing ``sources``
  key, prose instead of JSON. One targeted retry recovers most of them; a second
  one is not worth the latency and starts looking like flailing.
* **Never guess the schema.** A response missing ``answer`` is a parse failure,
  not an empty answer. Fabricating ``""`` here would produce a confident-looking
  factual answer with no content and a citation attached.
* **Cite only what was retrieved.** ``sources`` is intersected with the URLs
  actually in the retrieved set before it leaves this module. The model is
  instructed not to invent URLs, but instruction is not enforcement, and this is
  the cheapest place to enforce it (validators re-check as defence in depth).

On an unrecoverable parse failure this raises :class:`ParseError`; the
orchestrator turns that into an abstention, because a bot that says "I couldn't
find that" when it actually failed to parse is far more defensible than one that
passes a malformed string through to a user.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import SETTINGS
from ..llm.base import LLMClient, LLMError, ParseError
from ..models import RetrievedChunk
from . import prompt as prompt_mod

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
# A short, harmless preamble: "Here is the JSON you asked for:" and friends.
_PREAMBLE = re.compile(r"^[\s\w:,\.\-–—]*?(?=\{|\[)", re.DOTALL)


@dataclass(frozen=True)
class Generation:
    """One parsed, citation-sanitised model response."""

    answer: str
    sources: list[str] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    raw: str = ""
    attempts: int = 1
    repaired: bool = False


def _settings_int(dotted: str, default: int) -> int:
    try:
        return int(SETTINGS.get(dotted, default))
    except (TypeError, ValueError):
        return default


def _settings_float(dotted: str, default: float) -> float:
    try:
        return float(SETTINGS.get(dotted, default))
    except (TypeError, ValueError):
        return default


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, string-aware.

    Brace counting has to skip braces inside string literals, or an answer
    containing ``"}"`` — plausible in prose about fund categories — truncates the
    object and produces a baffling JSON error.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:position + 1]
    return None


def _clean_raw(raw: str) -> str:
    text = _FENCE.sub("", (raw or "").strip())
    text = _PREAMBLE.sub("", text, count=1)
    return text.strip()


def _parse(raw: str) -> dict[str, Any]:
    """Parse model output into the required shape, or raise :class:`ParseError`."""
    cleaned = _clean_raw(raw)
    if not cleaned:
        raise ParseError("model returned an empty response")

    block = _first_json_object(cleaned)
    if block is None:
        raise ParseError("no JSON object found in the model response")

    try:
        data = json.loads(block)
    except ValueError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ParseError(f"expected a JSON object, got {type(data).__name__}")

    if "answer" not in data:
        raise ParseError("JSON object has no 'answer' key")
    if not isinstance(data["answer"], str):
        raise ParseError(
            f"'answer' must be a string, got {type(data['answer']).__name__}"
        )

    sources = data.get("sources", [])
    if sources is None:
        sources = []
    if not isinstance(sources, list) or any(not isinstance(s, str) for s in sources):
        raise ParseError("'sources' must be a list of strings")

    refused = data.get("refused", False)
    if not isinstance(refused, bool):
        refused = bool(refused) if isinstance(refused, (int, float)) else False

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    return {
        "answer": data["answer"].strip(),
        "sources": [s.strip() for s in sources if s.strip()],
        "refused": refused,
        "reason": reason.strip(),
    }


def _repair_message(raw: str, error: str) -> dict[str, str]:
    """The follow-up turn. Names the error; does not re-send the whole prompt."""
    return {
        "role": "user",
        "content": (
            "Your previous reply could not be used: "
            f"{error}\n\n"
            "Reply with a single JSON object and nothing else, in exactly this "
            'shape: {"answer": "<up to 3 sentences>", "sources": ["<exact url>"], '
            '"refused": false, "reason": ""}\n'
            "No markdown fences, no preamble, no commentary. Use only URLs from "
            "the source blocks above."
        ),
    }


def _sanitise_sources(sources: Sequence[str], chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Intersect model-supplied URLs with the retrieved set, preserving order.

    Dropped, not repaired: if the model wants to cite a URL that was not
    retrieved, the correct response is to lose the citation and let the citation
    validator downgrade the answer — not to helpfully substitute a nearby URL,
    which would attach a real-looking link to an unsupported claim.
    """
    allowed = {rc.chunk.url for rc in chunks if rc.chunk.url}
    seen: set[str] = set()
    kept: list[str] = []
    for url in sources:
        if url in allowed and url not in seen:
            seen.add(url)
            kept.append(url)
    return kept


def generate(
    question: str,
    chunks: Sequence[RetrievedChunk],
    client: LLMClient | None = None,
) -> Generation:
    """Run the model, parse its JSON, and sanitise the citations. One retry max."""
    if not chunks:
        raise LLMError("nothing retrieved; refusing to call the LLM with no sources")

    if client is None:
        from ..llm import get_llm

        client = get_llm()

    temperature = _settings_float("generation.temperature", 0.0)
    max_tokens = _settings_int("generation.max_tokens", 250)

    base_messages = prompt_mod.build_messages(question, chunks)
    raw = client.generate(base_messages, temperature=temperature,
                          max_tokens=max_tokens)
    try:
        parsed = _parse(raw)
        return Generation(
            answer=parsed["answer"],
            sources=_sanitise_sources(parsed["sources"], chunks),
            refused=parsed["refused"],
            reason=parsed["reason"],
            raw=raw,
            attempts=1,
            repaired=False,
        )
    except ParseError as exc:
        first_error = str(exc)

    # Exactly one repair attempt, with the failing text in the conversation.
    messages = list(base_messages) + [
        {"role": "assistant", "content": raw[:2000]},
        _repair_message(raw, first_error),
    ]
    raw2 = client.generate(messages, temperature=temperature, max_tokens=max_tokens)
    try:
        parsed = _parse(raw2)
    except ParseError as exc:
        raise ParseError(
            f"{first_error}; retry also failed: {exc}"
        ) from exc

    return Generation(
        answer=parsed["answer"],
        sources=_sanitise_sources(parsed["sources"], chunks),
        refused=parsed["refused"],
        reason=parsed["reason"],
        raw=raw2,
        attempts=2,
        repaired=True,
    )


__all__ = ["generate", "Generation", "ParseError", "LLMError", "_parse",
           "_first_json_object", "_sanitise_sources"]
