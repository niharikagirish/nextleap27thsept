"""Phase 7 - the Streamlit demo surface (implementation.md 7.A-7.C, PRD M6).

A reviewer has three minutes. The screen has to make the whole system legible:
what the scope is, that it is facts-only, where every answer came from, and -
via the debug drawer - which passages were actually retrieved and what they
scored.

Three rules shape this module.

1. **Do not re-implement the pipeline.** Everything routes through
   :func:`app.pipeline.orchestrator.answer_question`, the same function the CLI
   uses. A second copy of the retrieval or guard logic in here is how the UI and
   the CLI start disagreeing about what the system does.

2. **Every user-facing string is imported from :mod:`app.disclaimers`.** None
   is inlined (FR-18). The disclaimer is a constraint (C9), not decoration, and
   a copy that drifts from the CLI is a constraint that quietly stops holding.

3. **User input is never persisted.** The transcript lives in
   ``st.session_state`` only. Stage logging records a salted hash of the
   question, never the question (C3/NFR-08), and that is enforced upstream in
   ``app.logging_utils`` - this module adds no new write path.
"""

from __future__ import annotations

import streamlit as st

from app.disclaimers import (
    ERROR_GENERIC,
    FRESHNESS_CAVEAT,
    LAST_UPDATED_LABEL,
    NO_LLM_KEY,
    SCOPE_SUMMARY,
    SHORT,
    WELCOME,
)
from app.sources import ALLOWLIST, EXAMPLE_QUESTIONS

# The score floor the retriever applies. Shown in the debug drawer so the
# numbers there are interpretable rather than arbitrary.
SCORE_FLOOR = 0.35

st.set_page_config(
    page_title="HDFC MF FAQ - facts-only assistant",
    page_icon="📄",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
# Streamlit re-runs this whole script on every keystroke and every widget
# interaction. Without caching, the ~90 MB SentenceTransformer and the Chroma
# client would be rebuilt each time and the app would appear to hang.
#
# Both are already process-level singletons (app.embedder_singleton,
# app.pipeline.store), so these wrappers mostly serve to load them exactly once
# and to surface a readable error instead of a raw traceback when the index has
# not been built yet.


@st.cache_resource(show_spinner="Loading the embedding model and index (first run only)...")
def _index() -> tuple[object, int]:
    """Return (collection, chunk_count), loading both once per process.

    The warm-up call is the point of this wrapper: the embedder is a ~90 MB
    SentenceTransformer, and the retriever would otherwise build it on the first
    query of every Streamlit session.
    """
    from app.embedder_singleton import get_model
    from app.pipeline.store import count, get_collection

    get_model()
    collection = get_collection()
    return collection, count(collection)


@st.cache_resource(show_spinner="Connecting to the language model...")
def _llm() -> object:
    """Return the configured LLM client, or the retrieval-only echo fallback."""
    from app.llm import get_llm

    return get_llm()


# --------------------------------------------------------------------------- #
# The one funnel from a question to an Answer
# --------------------------------------------------------------------------- #


def _ask(question: str):
    """Run a question through the pipeline. Nothing else in here calls the model."""
    from app.pipeline.orchestrator import answer_question

    return answer_question(question, llm=_llm())


def _error_answer(detail: str):
    """Wrap an unexpected failure so the transcript stays renderable."""
    from app.models import Answer

    return Answer(
        text=ERROR_GENERIC,
        sources=[],
        last_updated="",
        kind="error",
        reason=detail,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

# A refusal or an abstention is the system working, not failing. Only a genuine
# infrastructure failure gets the error treatment.
_KIND_NOTE = {
    "refusal": ("info", "I won't give investment advice"),
    "abstain": ("info", "Not in the source pages"),
    "perf_redirect": ("warning", "Performance question - redirected"),
    "pii_notice": ("warning", "Personal details removed"),
}


def _render_sources(answer) -> None:
    """Show every source as a full, clickable URL.

    The full URL is the link *text* on purpose. A truncated anchor would hide
    exactly what a reviewer needs to check (C5): that the citation is the page
    the answer actually came from.
    """
    if not answer.sources:
        return
    st.markdown("**Sources**")
    for url in answer.sources:
        st.markdown(f"- [{url}]({url})")


def _render_last_updated(answer) -> None:
    """C7 - the freshness stamp. Skipped when empty rather than shown dangling."""
    if answer.last_updated:
        st.caption(f"{LAST_UPDATED_LABEL} {answer.last_updated}")


def _render_debug(answer) -> None:
    """The debug drawer (FR-14): what was retrieved, and what it scored.

    Every kept chunk is shown, including the marginal ones just above the
    floor. Those are the interesting ones - they demonstrate the floor is a
    real threshold doing work rather than a number that was never applied.
    """
    chunks = answer.debug or []
    if not chunks:
        st.caption("No chunks were retrieved for this question.")
        return

    top = chunks[0].score
    st.markdown(
        f"**Retrieved {len(chunks)} chunk(s)** · floor {SCORE_FLOOR} · "
        f"top score {top:.3f}"
    )
    for rc in chunks:
        trail = " > ".join(rc.chunk.heading_trail)
        excerpt = " ".join(rc.chunk.text.split())
        st.markdown(
            f"`#{rc.rank}` **score {rc.score:.3f}** · `{rc.chunk.source_id}`\n\n"
            f"<small>{trail}</small>\n\n"
            f"> {excerpt}\n"
        )
    st.caption(
        "Scores are cosine similarity (1.0 = identical). Chunks below the floor "
        "are discarded before the model ever sees them."
    )


def _render_answer(answer, *, show_debug: bool) -> None:
    """Render one answer: text, kind-appropriate framing, sources, freshness."""
    with st.chat_message("assistant"):
        if answer.kind == "error":
            st.error(answer.text)
            if show_debug and answer.reason:
                st.caption(answer.reason)
        elif answer.kind in _KIND_NOTE:
            level, headline = _KIND_NOTE[answer.kind]
            getattr(st, level)(headline)
            st.markdown(answer.text)
        else:
            st.markdown(answer.text)
            st.caption(FRESHNESS_CAVEAT)

        _render_sources(answer)
        _render_last_updated(answer)

        if show_debug:
            with st.expander("Retrieval debug", expanded=False):
                _render_debug(answer)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Scope")
    st.markdown(SCOPE_SUMMARY)
    st.markdown("")
    for spec in ALLOWLIST:
        st.markdown(f"- **{spec.scheme_name}** — {spec.category}, {spec.plan}")

    st.divider()

    # FR-14. Off by default: the drawer is for demonstrating provenance, not
    # for being the main reading experience.
    show_debug = st.checkbox(
        "Show retrieval debug",
        value=False,
        help="Score and source of every chunk retrieved.",
    )

    st.divider()
    st.caption(SHORT)
    st.caption(
        "The index is built with `python -m app.ingest`. "
        "No LLM API key is needed to build it."
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("HDFC Mutual Fund FAQ Assistant")
st.write(WELCOME)
# C9 - must be visible without scrolling, so it sits near the top as well as in
# the sidebar.
st.warning(SHORT)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Both example chips and st.chat_input drop a question into ``_pending`` and
# rerun, so there is exactly one place where a question becomes an Answer.
if "_pending" in st.session_state:
    question = st.session_state.pop("_pending")
    st.session_state["messages"].append({"role": "user", "text": question})

    try:
        _collection, n_chunks = _index()
        if n_chunks == 0:
            st.error(
                "The retrieval index is empty. Build it first:\n\n"
                "```\npython -m app.ingest --stage all\n```"
            )
        else:
            st.session_state["messages"].append(
                {"role": "assistant", "answer": _ask(question)}
            )
    except Exception as exc:  # noqa: BLE001 - one bad question must not kill the app
        st.session_state["messages"].append(
            {"role": "assistant", "answer": _error_answer(f"{type(exc).__name__}: {exc}")}
        )
    st.rerun()


# --------------------------------------------------------------------------- #
# Landing: exactly three example questions
# --------------------------------------------------------------------------- #

if not st.session_state["messages"]:
    st.subheader("Try one of these")
    # Straight from app.sources.EXAMPLE_QUESTIONS (FR-18). Stacked in one column
    # rather than three, so the layout stays readable at mobile width.
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state["_pending"] = example
            st.rerun()

st.divider()

# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #

for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["text"])
    else:
        _render_answer(message["answer"], show_debug=show_debug)


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #

typed = st.chat_input("Ask a fact about the 5 HDFC schemes")
if typed:
    st.session_state["_pending"] = typed
    st.rerun()

# Retrieval-only mode is a supported configuration, not a failure, so it is
# announced once at the bottom rather than hidden.
if "echo" in type(_llm()).__name__.lower():
    st.info(NO_LLM_KEY)
