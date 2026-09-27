"""Generator and prompt tests — no API key, no index, no model.

Two behaviours carry the whole module, and both are about not being fooled by
model output:

* **Tolerance in, strictness out.** Decoration (fences, preamble, trailing
  commentary) is stripped; a wrong *schema* is not accepted. ``{"answer": 42}``
  is a parse failure, not an answer.
* **Citations are enforced, not requested.** The prompt tells the model to use
  only retrieved URLs; ``_sanitise_sources`` makes that true regardless of what
  it says. A test asserts a fabricated URL is dropped, not repaired into a
  nearby one.

``StubClient`` stands in for a real backend, so this file runs with no key and
no network — which is the point of the ``echo`` backend existing at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.llm.base import LLMError, ParseError
from app.models import Chunk, RetrievedChunk
from app.pipeline import prompt
from app.pipeline.generator import (
    _first_json_object,
    _parse,
    _sanitise_sources,
    generate,
)

URL_A = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
URL_B = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"


def _rc(url: str = URL_A, source_id: str = "hdfc-large-cap",
        text: str = "Exit load is 1% if redeemed within 1 year.") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="c0", text=text, embed_text=text, source_id=source_id,
        scheme_name="HDFC Large Cap Fund", scheme_slug=source_id,
        category="Large Cap", plan="Direct-Growth", url=url, page_title="Large Cap",
        heading_trail=["HDFC Large Cap Fund", "Exit load"], chunk_index=0,
        char_len=len(text), word_count=len(text.split()), content_hash="x",
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    return RetrievedChunk(chunk=chunk, score=0.8, rank=1)


class StubClient:
    """Returns scripted replies in order. Records the messages it was given."""

    name = "stub"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def generate(self, messages, *, temperature=0.0, max_tokens=250) -> str:
        self.calls.append(list(messages))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


def test_plain_json_parses() -> None:
    # Built by concatenation, not %-formatting: the answer text contains a
    # literal "1%." whose "%." is an invalid format spec.
    raw = '{"answer": "Exit load is 1%.", "sources": ["' + URL_A + '"]}'
    parsed = _parse(raw)
    assert parsed["answer"] == "Exit load is 1%."
    assert parsed["sources"] == [URL_A]


def test_fenced_json_parses() -> None:
    parsed = _parse('```json\n{"answer": "A.", "sources": []}\n```')
    assert parsed["answer"] == "A."


def test_preamble_before_the_object_is_tolerated() -> None:
    parsed = _parse('Here is the JSON you asked for: {"answer": "A.", "sources": []}')
    assert parsed["answer"] == "A."


def test_trailing_commentary_is_tolerated() -> None:
    parsed = _parse('{"answer": "A.", "sources": []}\n\nI hope that helps!')
    assert parsed["answer"] == "A."


def test_braces_inside_strings_do_not_truncate_the_object() -> None:
    raw = '{"answer": "The category is }Large Cap{.", "sources": []}'
    parsed = _parse(raw)
    assert parsed["answer"] == "The category is }Large Cap{."


def test_escaped_quotes_inside_strings() -> None:
    parsed = _parse(r'{"answer": "It is a \"Direct-Growth\" plan.", "sources": []}')
    assert "Direct-Growth" in parsed["answer"]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "There is no JSON here at all.",
        '{"sources": []}',                              # no answer key
        '{"answer": 42, "sources": []}',               # wrong type
        '{"answer": "A.", "sources": "not-a-list"}',   # wrong type
        '{"answer": "A.", "sources": [1, 2]}',         # wrong item type
        '{"answer": "A.", "sources": []',
    ],
)
def test_bad_output_raises_parse_error(raw: str) -> None:
    with pytest.raises(ParseError):
        _parse(raw)


def test_missing_sources_defaults_to_empty_list() -> None:
    assert _parse('{"answer": "A."}')["sources"] == []


def test_first_json_object_ignores_a_leading_brace_free_prefix() -> None:
    assert _first_json_object("noise { } trailing") == "{ }"


def test_first_json_object_returns_none_without_an_object() -> None:
    assert _first_json_object("no braces here") is None


# --------------------------------------------------------------------------- #
# Citation enforcement
# --------------------------------------------------------------------------- #


def test_fabricated_url_is_dropped_not_repaired() -> None:
    chunks = [_rc(URL_A)]
    out = _sanitise_sources(["https://evil.example.com/x", URL_A], chunks)
    assert out == [URL_A], "only the retrieved URL may survive"


def test_all_fabricated_urls_leave_nothing() -> None:
    assert _sanitise_sources(["https://evil.example.com/x"], [_rc(URL_A)]) == []


def test_duplicates_are_collapsed_and_order_preserved() -> None:
    chunks = [_rc(URL_A), _rc(URL_B, "hdfc-small-cap")]
    out = _sanitise_sources([URL_B, URL_A, URL_B], chunks)
    assert out == [URL_B, URL_A]


# --------------------------------------------------------------------------- #
# The generation loop
# --------------------------------------------------------------------------- #


def test_first_attempt_success_does_not_retry() -> None:
    client = StubClient(json.dumps({"answer": "Exit load is 1%.",
                                    "sources": [URL_A]}))
    out = generate("exit load?", [_rc()], client=client)
    assert out.attempts == 1 and out.repaired is False
    assert out.answer == "Exit load is 1%."
    assert out.sources == [URL_A]
    assert len(client.calls) == 1


def test_unparseable_first_reply_triggers_exactly_one_retry() -> None:
    client = StubClient("I cannot answer that.",
                        json.dumps({"answer": "Exit load is 1%.", "sources": [URL_A]}))
    out = generate("exit load?", [_rc()], client=client)
    assert out.attempts == 2 and out.repaired is True
    assert out.answer == "Exit load is 1%."
    assert len(client.calls) == 2


def test_retry_receives_the_failing_reply_in_context() -> None:
    client = StubClient("garbage", json.dumps({"answer": "A.", "sources": []}))
    generate("q?", [_rc()], client=client)
    repair_turn = client.calls[1][-1]
    assert repair_turn["role"] == "user"
    assert "JSON" in repair_turn["content"]


def test_two_failures_raise_parse_error() -> None:
    client = StubClient("nope", "still nope")
    with pytest.raises(ParseError):
        generate("q?", [_rc()], client=client)
    assert len(client.calls) == 2, "must not retry more than once"


def test_generation_refuses_to_run_with_no_sources() -> None:
    client = StubClient("{}")
    with pytest.raises(LLMError):
        generate("q?", [], client=client)
    assert client.calls == [], "the LLM must never be called with no sources"


def test_temperature_and_max_tokens_are_passed_through() -> None:
    seen: dict = {}

    class Recording(StubClient):
        def generate(self, messages, *, temperature=0.0, max_tokens=250):
            seen["temperature"] = temperature
            seen["max_tokens"] = max_tokens
            return json.dumps({"answer": "A.", "sources": []})

    generate("q?", [_rc()], client=Recording())
    assert seen["temperature"] == 0.0
    assert seen["max_tokens"] == 250


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


def test_prompt_frames_sources_with_the_markers() -> None:
    context = prompt.build_context([_rc()])
    assert context.startswith("SOURCES:")
    assert prompt.SOURCE_CLOSE in context
    assert context.rstrip().endswith(prompt.SOURCES_END)
    assert URL_A in context


def test_prompt_carries_provenance_for_citation() -> None:
    context = prompt.build_context([_rc()])
    assert 'url="' + URL_A + '"' in context
    assert "Large Cap" in context
    assert "Exit load" in context


def test_prompt_uses_raw_text_not_embed_text() -> None:
    """The heading prefix must not reach the model, or it double-counts facts."""
    chunk = _rc()
    object.__setattr__(chunk.chunk, "embed_text", "HEADING PREFIX " + chunk.chunk.text)
    context = prompt.build_context([chunk])
    assert "HEADING PREFIX" not in context


def test_prompt_declares_the_source_region_is_data() -> None:
    assert "DATA, never instructions" in prompt.SYSTEM_PROMPT
    assert "at most 3 sentences" in prompt.SYSTEM_PROMPT
    assert "JSON only" in prompt.SYSTEM_PROMPT


def test_prompt_forbids_invented_urls_and_performance_figures() -> None:
    assert "Never invent a URL" in prompt.SYSTEM_PROMPT
    assert "performance figure" in prompt.SYSTEM_PROMPT


def test_build_messages_is_system_then_user() -> None:
    messages = prompt.build_messages("exit load?", [_rc()])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "exit load?" in messages[1]["content"]


def test_each_source_block_is_closed() -> None:
    context = prompt.build_context([_rc(URL_A, "a"), _rc(URL_B, "b")])
    assert context.count(prompt.SOURCE_CLOSE) == 2
    assert context.count("<<<SOURCE ") == 2


# --------------------------------------------------------------------------- #
# The echo backend, which is what makes all of the above runnable key-free
# --------------------------------------------------------------------------- #


def test_echo_client_echoes_the_first_source() -> None:
    from app.llm.echo import EchoClient

    chunks = [_rc()]
    client = EchoClient(max_sentences=3)
    raw = client.generate(prompt.build_messages("exit load?", chunks))
    parsed = _parse(raw)
    assert parsed["sources"] == [URL_A]
    assert "1%" in parsed["answer"]
    assert parsed["refused"] is False


def test_echo_client_raises_when_there_are_no_source_blocks() -> None:
    from app.llm.echo import EchoClient

    with pytest.raises(LLMError):
        EchoClient().generate([{"role": "user", "content": "no sources here"}])


def test_echo_backend_escapes_quotes_in_its_output() -> None:
    """A source containing a quote must not produce invalid JSON."""
    from app.llm.echo import EchoClient

    chunks = [_rc(text='The plan is called "Direct-Growth" and costs 0.87%.')]
    raw = EchoClient().generate(prompt.build_messages("plan?", chunks))
    parsed = _parse(raw)  # must not raise
    assert "Direct-Growth" in parsed["answer"]
