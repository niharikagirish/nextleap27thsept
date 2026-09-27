"""UI tests for the Phase 7 Streamlit surface (implementation.md Gate 7).

These run the real script through Streamlit's ``AppTest``, so they exercise the
actual widget tree rather than a mock of it. Most of them inject a synthetic
``Answer`` into ``session_state`` instead of asking a question, which keeps the
suite offline and deterministic - the pipeline's own behaviour is covered by
``test_chat_render.py`` and the orchestrator tests.

What is being defended here is mostly about *trustworthiness of the surface*:
that a refusal is not dressed up as a crash, that a citation is a full readable
URL, and that nothing the user typed is written to disk.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import disclaimers, ui
from app.models import Answer, Chunk, RetrievedChunk
from app.sources import ALLOWLIST, EXAMPLE_QUESTIONS

UI_PATH = Path(ui.__file__)


def _chunk(index: int = 0, score: float = 0.5, rank: int = 1) -> RetrievedChunk:
    spec = ALLOWLIST[0]
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"c{index:015d}",
            text=f"Expense ratio of {spec.scheme_name} is 1.03%.",
            embed_text=f"{spec.scheme_name} > Scheme facts\n\nExpense ratio",
            source_id=spec.source_id,
            scheme_name=spec.scheme_name,
            scheme_slug=spec.source_id,
            category=spec.category,
            plan=spec.plan,
            url=spec.url,
            page_title="HDFC Large Cap Fund Direct Growth",
            heading_trail=[spec.scheme_name, "Scheme facts"],
            chunk_index=index,
            char_len=48,
            word_count=9,
            content_hash="deadbeef",
            fetched_at="2026-09-27T09:41:46Z",
        ),
        score=score,
        rank=rank,
    )


def _answer(kind: str = "factual", **kwargs) -> Answer:
    base = dict(
        text="The expense ratio is 1.03%.",
        sources=[ALLOWLIST[0].url],
        last_updated="2026-09-27T09:41:46Z",
        kind=kind,
        reason="ok",
    )
    base.update(kwargs)
    return Answer(**base)


def _rendered(at: AppTest) -> str:
    """Every piece of text the user can see, flattened."""
    parts: list[str] = []
    for group in (at.markdown, at.caption, at.info, at.warning, at.error,
                  at.success, at.subheader, at.header):
        parts.extend(str(getattr(e, "value", "")) for e in group)
    return "\n".join(parts)


@pytest.fixture(scope="module")
def landing() -> AppTest:
    """The app in its first-load state, with no model loaded."""
    at = AppTest.from_file(str(UI_PATH), default_timeout=180)
    at.run()
    return at


# --------------------------------------------------------------------------- #
# Gate 7 - the app starts clean
# --------------------------------------------------------------------------- #


def test_app_starts_with_no_exception(landing: AppTest) -> None:
    assert [e.value for e in landing.exception] == []


def test_welcome_line_is_present_on_first_load(landing: AppTest) -> None:
    assert disclaimers.WELCOME in _rendered(landing)


def test_exactly_three_example_questions_and_they_are_the_real_ones(
    landing: AppTest,
) -> None:
    labels = [b.label for b in landing.button]
    assert len(labels) == 3
    assert tuple(labels) == EXAMPLE_QUESTIONS


# --------------------------------------------------------------------------- #
# C9 - the disclaimer is visible without scrolling
# --------------------------------------------------------------------------- #


def test_short_disclaimer_is_above_the_fold_and_pinned_in_the_sidebar(
    landing: AppTest,
) -> None:
    assert any(disclaimers.SHORT in w.value for w in landing.warning)
    sidebar_text = "\n".join(
        str(getattr(e, "value", "")) for e in landing.sidebar.caption
    )
    assert disclaimers.SHORT in sidebar_text


# --------------------------------------------------------------------------- #
# FR-14 - the debug drawer
# --------------------------------------------------------------------------- #


def test_debug_toggle_exists_and_defaults_off(landing: AppTest) -> None:
    boxes = landing.checkbox
    assert len(boxes) == 1
    assert boxes[0].label == "Show retrieval debug"
    assert boxes[0].value is False


def test_debug_drawer_shows_scores_for_every_chunk_including_low_ones() -> None:
    at = AppTest.from_file(str(UI_PATH), default_timeout=180)
    at.run()
    at.session_state["messages"] = [
        {"role": "user", "text": "expense ratio?"},
        {
            "role": "assistant",
            "answer": _answer(
                debug=[
                    _chunk(0, score=0.812, rank=1),
                    # A marginal hit just above the floor. The drawer must not
                    # hide it - showing it is what proves the floor is real.
                    _chunk(1, score=0.361, rank=2),
                ]
            ),
        },
    ]
    at.checkbox[0].set_value(True).run()

    assert [e.value for e in at.exception] == []
    text = _rendered(at)
    assert "0.812" in text
    assert "0.361" in text
    assert "Retrieved 2 chunk" in text
    assert "floor" in text


# --------------------------------------------------------------------------- #
# C5 / C7 - citations and freshness
# --------------------------------------------------------------------------- #


def test_factual_answer_shows_full_clickable_url_and_freshness() -> None:
    at = AppTest.from_file(str(UI_PATH), default_timeout=180)
    at.run()
    at.session_state["messages"] = [
        {"role": "user", "text": "expense ratio?"},
        {"role": "assistant", "answer": _answer()},
    ]
    at.run()

    assert [e.value for e in at.exception] == []
    text = _rendered(at)
    # The full URL, not a truncated anchor: the reviewer has to be able to read
    # where the answer came from.
    assert ALLOWLIST[0].url in text
    assert "2026-09-27T09:41:46Z" in text
    assert f"{disclaimers.LAST_UPDATED_LABEL} 2026-09-27T09:41:46Z" in text


def test_freshness_line_is_omitted_rather_than_shown_dangling() -> None:
    at = AppTest.from_file(str(UI_PATH), default_timeout=180)
    at.run()
    at.session_state["messages"] = [
        {"role": "user", "text": "q"},
        {"role": "assistant", "answer": _answer(last_updated="")},
    ]
    at.run()

    assert disclaimers.LAST_UPDATED_LABEL not in _rendered(at)


# --------------------------------------------------------------------------- #
# Refusals and abstentions are not crashes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kind",
    ["refusal", "abstain", "perf_redirect", "pii_notice"],
)
def test_non_factual_kinds_render_as_a_callout_not_an_error(kind: str) -> None:
    at = AppTest.from_file(str(UI_PATH), default_timeout=180)
    at.run()
    at.session_state["messages"] = [
        {"role": "user", "text": "should I buy it?"},
        {"role": "assistant", "answer": _answer(kind=kind)},
    ]
    at.run()

    assert [e.value for e in at.exception] == []
    # The whole point: a refusal working correctly must not look like a bug.
    assert [e.value for e in at.error] == []
    assert at.info or at.warning


def test_error_kind_is_the_only_one_rendered_as_an_error() -> None:
    at = AppTest.from_file(str(UI_PATH), default_timeout=180)
    at.run()
    at.session_state["messages"] = [
        {"role": "user", "text": "q"},
        {
            "role": "assistant",
            "answer": _answer(
                kind="error",
                text=disclaimers.ERROR_GENERIC,
                sources=[],
                last_updated="",
                reason="llm_unavailable",
            ),
        },
    ]
    at.run()

    assert len(at.error) == 1
    assert disclaimers.ERROR_GENERIC in at.error[0].value
    # The reason is diagnostic, so it is gated behind the debug toggle.
    assert "llm_unavailable" not in _rendered(at)


def test_pii_notice_does_not_echo_the_sensitive_value() -> None:
    at = AppTest.from_file(str(UI_PATH), default_timeout=180)
    at.run()
    at.session_state["messages"] = [
        {"role": "user", "text": "my PAN is ABCDE1234F"},
        {"role": "assistant", "answer": _answer(kind="pii_notice", sources=[], last_updated="")},
    ]
    at.run()

    text = _rendered(at)
    assert "ABCDE1234F" in text  # only the user's own bubble
    assert text.count("ABCDE1234F") == 1


# --------------------------------------------------------------------------- #
# FR-18 - one definition per string
# --------------------------------------------------------------------------- #


def test_no_user_facing_string_is_inlined_in_the_ui() -> None:
    """FR-18: the disclaimer must not be copy-pasted into the UI.

    A second copy is how the CLI and the UI drift apart, and C9 is a
    constraint rather than decoration.
    """
    source = UI_PATH.read_text(encoding="utf-8")
    assert disclaimers.SHORT not in source
    assert disclaimers.WELCOME not in source
    assert disclaimers.FULL not in source
    assert disclaimers.PERF_REDIRECT not in source
    assert disclaimers.REFUSAL not in source
    assert disclaimers.PII_NOTICE not in source


def test_scope_and_examples_come_from_the_allowlist() -> None:
    source = UI_PATH.read_text(encoding="utf-8")
    assert "from app.sources import ALLOWLIST, EXAMPLE_QUESTIONS" in source
    for spec in ALLOWLIST:
        # Not hardcoded: every scheme the sidebar shows must trace to the
        # allowlist, which is the single source of truth for the corpus.
        assert spec.scheme_name not in source


def test_ui_calls_the_shared_pipeline_rather_than_its_own_copy() -> None:
    source = UI_PATH.read_text(encoding="utf-8")
    assert "from app.pipeline.orchestrator import answer_question" in source
    # A UI that re-implemented retrieval or generation would drift from the CLI.
    for forbidden in ("def embed_query", "SYSTEM_PROMPT =", "def build_context"):
        assert forbidden not in source


# --------------------------------------------------------------------------- #
# NFR-08 - nothing the user typed is persisted
# --------------------------------------------------------------------------- #


def test_ui_writes_nothing_to_disk() -> None:
    """The transcript must live only in session_state.

    PII screening and the no-logging rule are enforced upstream, so the UI's
    job is simply not to add a second write path. A stray ``open(...)`` here
    would quietly undo that.
    """
    source = UI_PATH.read_text(encoding="utf-8")
    assert "open(" not in source
    assert "to_csv" not in source
    assert "json.dump" not in source
    assert 'st.session_state["messages"]' in source


def test_score_floor_shown_in_the_drawer_matches_the_retriever() -> None:
    """The drawer must not quote a floor the retriever does not actually use."""
    from app.config import SETTINGS

    assert ui.SCORE_FLOOR == SETTINGS.get("retrieval.score_floor")


def test_render_helpers_are_callable_without_a_running_streamlit() -> None:
    """A smoke test that the module's own surface is importable and bound."""
    assert callable(inspect.unwrap(ui._ask))
    assert callable(ui._render_answer)
    assert callable(ui._render_debug)
