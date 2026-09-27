"""Answer rendering tests (``app.chat.render``).

The bug these exist to prevent: an infrastructure failure rendered as a sourced,
authoritative answer. ``kind="error"`` carries the provider's raw error string as
its text, and the normal render path printed that verbatim - a Groq 404 JSON body
and all - followed by "Last updated from sources" and the facts-only disclaimer.
On a demo machine with a dead model name, the failure mode looked like the app
working perfectly and citing its sources.

An error is the one kind where the styling actively lies, so it is the one kind
with its own branch.
"""

from __future__ import annotations

import inspect
import re

from app.chat import ERROR_ADVICE, ERROR_HEADLINE, render
from app.disclaimers import LAST_UPDATED_LABEL
from app.models import Answer

# What Groq actually returns for a model that does not exist. Used verbatim
# because the point is that provider internals must not reach the user by default.
GROQ_404 = (
    'https://api.groq.com/openai/v1 returned 404: {"error":{"message":"The '
    'model `llama-3.3-70b-versatile` does not exist or you do not have access '
    'to it.","type":"invalid_request_error","code":"model_not_found"}}'
)


def _error(**kwargs) -> Answer:
    base = dict(
        text=GROQ_404,
        sources=["https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"],
        last_updated="2026-09-27T09:41:53Z",
        kind="error",
        reason="llm_unavailable",
    )
    base.update(kwargs)
    return Answer(**base)


def test_error_does_not_look_like_a_sourced_answer() -> None:
    out = render(_error())

    assert ERROR_HEADLINE in out
    assert "llm_unavailable" in out
    # The three things that made the outage look authoritative.
    assert "Last updated" not in out
    assert "Sources:" not in out
    assert "groww.in" not in out
    assert "No investment advice" not in out


def test_error_hides_provider_internals_by_default() -> None:
    out = render(_error())

    assert GROQ_404 not in out
    assert "api.groq.com" not in out
    assert "model_not_found" not in out
    # It must still tell the user what to do about it.
    assert ".env" in out


def test_error_keeps_the_technical_detail_under_debug() -> None:
    out = render(_error(), debug=True)

    assert GROQ_404 in out
    assert "--debug" in out
    # Still no false provenance, even with the detail shown.
    assert "Last updated" not in out
    assert "Sources:" not in out


def test_error_without_a_reason_still_renders() -> None:
    out = render(_error(reason=None))

    assert ERROR_HEADLINE in out
    assert "llm_unavailable" in out


def test_rate_limit_is_not_reported_as_a_key_problem() -> None:
    """A 429 with a valid key must not send the user off to check their key.

    This is the confusion that prompted the change: one flat "llm_unavailable"
    for every cause meant the advice was always a guess, and usually about the
    wrong variable.
    """
    out = render(_error(kind="error", reason="llm_rate_limited",
                        text="rate limited (429)"))

    assert "llm_rate_limited" in out
    assert "the key and model are fine" in out
    # The key/model advice belongs to a different failure and would mislead.
    assert ".env" not in out


def test_dead_model_names_the_model_not_the_key() -> None:
    out = render(_error(reason="llm_model_not_found"))

    assert "llm_model_not_found" in out
    assert "GROQ_MODEL" in out or "GEMINI_MODEL" in out
    assert "rejected the API key" not in out


def test_unknown_reason_is_shown_not_guessed() -> None:
    """A cause the CLI has not learned about must surface, not be flattened."""
    out = render(_error(reason="llm_something_new"))

    assert "llm_something_new" in out
    assert "--debug" in out


def test_advice_exists_for_every_code_the_client_can_raise() -> None:
    """New LLMError.code values need an advice line, or users get a dead end."""
    from app.llm.openai_compat import OpenAICompatClient

    source = inspect.getsource(OpenAICompatClient)
    codes = set(re.findall(r'code="([a-z_]+)"', source))
    assert codes, "no codes found - the scan itself is broken"
    for code in codes:
        assert code in ERROR_ADVICE, f"{code} has no advice in ERROR_ADVICE"


def test_factual_answer_keeps_its_sources_and_disclaimer() -> None:
    """The error branch must not have cost the normal path anything."""
    answer = Answer(
        text="The exit load is 1% if redeemed within 1 year.",
        sources=["https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"],
        last_updated="2026-09-27T09:41:53Z",
        kind="factual",
        reason="ok",
    )
    out = render(answer)

    assert "The exit load is 1%" in out
    assert "Sources:" in out
    assert "groww.in" in out
    # Asserted against the constant, not a literal, so relabelling the freshness
    # line cannot silently stop testing that it is rendered at all.
    assert LAST_UPDATED_LABEL in out
    assert "2026-09-27T09:41:53Z" in out
    assert "No investment advice" in out
    assert ERROR_HEADLINE not in out


def test_abstention_is_not_labelled_as_an_error() -> None:
    """A refusal and an abstention are normal answers, not outages."""
    answer = Answer(
        text="I could not find that in the official sources.",
        sources=[],
        last_updated=None,
        kind="abstain",
        reason="out_of_corpus",
    )
    out = render(answer)

    assert "could not find" in out
    assert ERROR_HEADLINE not in out
