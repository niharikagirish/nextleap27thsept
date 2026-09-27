"""STAGE 4 — retrieval (architecture.md §5.3).

    score = 1.0 - distance for cosine space with normalized embeddings.

The shape of this stage is deliberate and slightly counter-intuitive: **fetch
wide, filter hard.** Twelve candidates come back from ChromaDB, the score floor
throws away everything under 0.35, and only the survivors are trimmed to six.
Fetching six and then filtering can leave three, and a genuine two-part question
("what is the benchmark and the riskometer?") then goes unanswered on its second
half.

Three rules that must not be relaxed:

* **Never return a below-floor chunk.** If nothing clears the floor the correct
  answer is ``[]``, and the orchestrator abstains. Padding the result with weak
  matches is how a facts-only bot ends up inventing something.
* **Never lower the floor to make a result appear.** The floor is a
  correctness control, not a quality dial.
* **Log ``q_hash``, never the question.** C3/NFR-08: enough to correlate a slow
  request, not enough to reconstruct what someone asked about a mutual fund.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ..config import SETTINGS
from ..logging_utils import emit, q_hash
from ..models import Chunk, RetrievedChunk
from . import store as store_mod

# Jaccard is computed on word sets, so punctuation and case must not create
# spurious differences between two copies of the same fact.
_TOKEN = re.compile(r"[a-z0-9]+")


def _normalize_for_dedupe(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    """Word-set Jaccard similarity in [0, 1]."""
    set_a, set_b = _normalize_for_dedupe(a), _normalize_for_dedupe(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _count_tokens(text: str) -> int:
    """Word-piece length via the shared tokenizer, with a word-count fallback.

    The fallback is only used if the tokenizer cannot be loaded, which on a
    configured install should never happen — but a retrieval path that hard-fails
    on a missing tokenizer would take the whole demo down over a token budget.
    """
    try:
        from .chunker import count_wordpieces

        return count_wordpieces(text)
    except Exception:  # noqa: BLE001 - degrade to an estimate, never crash
        return int(len(text.split()) * 1.35) + 2


def rebuild_chunk(meta: dict[str, Any], document: str) -> Chunk:
    """Reconstruct a :class:`Chunk` from a Chroma row.

    ``heading_trail`` arrives as a ``" > "``-delimited string and is split back
    into a list here. ``text`` is the raw stored document, so citations stay
    clean. ``embed_text`` is rebuilt rather than stored: it is only needed at
    ingest time, and duplicating it in the index would be a second copy to drift.
    """
    fields = store_mod.from_metadata(meta, document)
    source_id = fields["source_id"]
    index = fields["chunk_index"]
    chunk_id = store_mod.chunk_id_for(source_id, index)
    content_hash = fields["content_hash"] or ""
    trail = fields["heading_trail"]
    text = fields["text"]
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        embed_text=_embed_text(trail, text),
        source_id=source_id,
        scheme_name=fields["scheme_name"],
        scheme_slug=fields["scheme_slug"],
        category=fields["category"],
        plan=fields["plan"],
        url=fields["url"],
        page_title=fields["page_title"],
        heading_trail=trail,
        chunk_index=index,
        char_len=fields["char_len"] or len(text),
        word_count=fields["word_count"] or len(text.split()),
        content_hash=content_hash,
        fetched_at=fields["fetched_at"],
    )


def _embed_text(heading_trail: Sequence[str], text: str) -> str:
    from .chunker import build_embed_text

    return build_embed_text(heading_trail, text)


def embed_query(text: str) -> list[float]:
    """Embed a query with the SAME singleton and normalization as Stage 3.

    Never instantiate a model here. A second instance with different
    normalization would make every score meaningless while still looking
    plausible.
    """
    from .embedder import EmbeddingError, embed_query as _embed

    try:
        return _embed(text)
    except EmbeddingError as exc:
        raise RetrievalError(str(exc)) from exc


class RetrievalError(RuntimeError):
    """Retrieval could not run, with a remedy attached."""


# --------------------------------------------------------------------------- #
# Core search
# --------------------------------------------------------------------------- #


def search(
    question: str,
    n_results: int | None = None,
    top_k: int | None = None,
    scheme_slug: str | None = None,
    collection: Any | None = None,
) -> list[RetrievedChunk]:
    """Retrieve scored chunks for ``question``.

    Returns ``[]`` — never a padded or below-threshold list — when nothing
    clears the score floor. The caller turns that into an abstention.
    """
    wide = int(n_results or SETTINGS.get("retrieval.n_results", 12))
    keep = int(top_k or SETTINGS.get("retrieval.top_k", 6))
    floor = float(SETTINGS.get("retrieval.score_floor", 0.35))
    dedupe_threshold = float(SETTINGS.get("retrieval.dedupe_jaccard", 0.9))
    budget = int(SETTINGS.get("retrieval.max_context_tokens", 2000))

    where = {"scheme_slug": scheme_slug} if scheme_slug else None

    try:
        coll = collection or store_mod.get_collection()
    except store_mod.StoreError as exc:
        raise RetrievalError(str(exc)) from exc

    try:
        vector = embed_query(question)
    except RetrievalError as exc:
        raise RetrievalError(
            f"{exc}\nRemedy: run `python -m app.ingest --stage embed --stage store` first."
        ) from exc

    raw = coll.query(
        query_embeddings=[vector],
        n_results=wide,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    fetched = len(documents)

    # ---- score: cosine space, normalized vectors -> score = 1.0 - distance ----
    scored: list[RetrievedChunk] = []
    for position, document in enumerate(documents):
        if position >= len(distances):
            break
        meta = metadatas[position] if position < len(metadatas) else {}
        chunk = rebuild_chunk(meta, document)
        scored.append(
            RetrievedChunk(
                chunk=chunk,
                score=round(1.0 - float(distances[position]), 6),
                rank=0,
            )
        )

    scored.sort(key=lambda rc: rc.score, reverse=True)
    kept = [rc for rc in scored if rc.score >= floor]
    dropped_floor = len(scored) - len(kept)

    # ---- dedupe near-identical chunks, keeping the higher-scoring one --------
    deduped: list[RetrievedChunk] = []
    n_deduped = 0
    for candidate in kept:
        duplicate = False
        for existing in deduped:
            if jaccard(existing.chunk.text, candidate.chunk.text) > dedupe_threshold:
                duplicate = True
                break
        if duplicate:
            n_deduped += 1
        else:
            deduped.append(candidate)

    # ---- trim to top_k, then enforce the context budget ----------------------
    trimmed = deduped[:keep]
    n_trimmed = max(0, len(deduped) - len(trimmed))

    within_budget: list[RetrievedChunk] = []
    used = 0
    for rc in trimmed:
        cost = _count_tokens(rc.chunk.text)
        if within_budget and used + cost > budget:
            break
        within_budget.append(rc)
        used += cost
    n_budget = len(trimmed) - len(within_budget)

    ranked = [
        RetrievedChunk(chunk=rc.chunk, score=rc.score, rank=index)
        for index, rc in enumerate(within_budget, start=1)
    ]

    emit(
        "retrieve",
        q_hash=q_hash(question),
        n_retrieved=fetched,
        n_kept=len(ranked),
        dropped=dropped_floor + n_trimmed + n_budget,
        filtered=dropped_floor,
        n_deduped=n_deduped,
        truncated=n_budget,
        top_score=ranked[0].score if ranked else 0.0,
        floor=floor,
        reranked=False,
    )
    return ranked


def search_top(
    question: str, scheme_slug: str | None = None, k: int = 1
) -> list[RetrievedChunk]:
    """Convenience wrapper used by the intent gate and abstention paths.

    Deliberately bypasses the score floor: the abstention message wants to point
    at the *closest* scheme page even when nothing was a confident match. This
    is never used to produce an answer.
    """
    coll = store_mod.get_collection()
    vector = embed_query(question)
    raw = coll.query(
        query_embeddings=[vector],
        n_results=max(k, 1),
        where={"scheme_slug": scheme_slug} if scheme_slug else None,
        include=["documents", "metadatas", "distances"],
    )
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    out: list[RetrievedChunk] = []
    for position, document in enumerate(documents[:k]):
        meta = metadatas[position] if position < len(metadatas) else {}
        out.append(
            RetrievedChunk(
                chunk=rebuild_chunk(meta, document),
                score=round(1.0 - float(distances[position]), 6),
                rank=position + 1,
            )
        )
    return out


def closest_scheme_url(question: str) -> str:
    """Best-guess scheme page URL, for abstention and refusal messages.

    Falls back to the first allowlisted scheme so callers always get a valid
    allowlist URL (C1) rather than an empty citation list.
    """
    from ..sources import ALLOWLIST

    try:
        hits = search_top(question, k=1)
        if hits and hits[0].chunk.url:
            return hits[0].chunk.url
    except Exception:  # noqa: BLE001 - a hint must never break the response
        pass
    return ALLOWLIST[0].url


def detect_scheme_slug(question: str) -> str | None:
    """Infer a ``scheme_slug`` from an explicit scheme mention in the question.

    Returns ``None`` when the question does not name a scheme, in which case no
    metadata filter is applied and the retriever relies on the embedded scheme
    name instead.

    Matching reuses the allowlist logic in :mod:`app.guards.intent` rather than
    repeating a token list here. A local "count the distinctive words" heuristic
    gets "HDFC Mid Cap fund" wrong: "cap" is shared with HDFC Large Cap, so the
    Mid Cap question silently filtered retrieval down to the Large Cap page and
    answered with that fund's expense ratio under a real citation. Requiring the
    mentioned name's tokens to be a *subset* of a known scheme's name fixes it —
    "mid" is in no allowlisted name, so the mention is rejected outright instead
    of resolved to the wrong scheme.
    """
    from ..guards.intent import _FUND_MENTION, _matches_allowlist, _normalise
    from ..sources import ALLOWLIST

    specs = [(spec.source_id, set(_normalise(spec.scheme_name).split()))
             for spec in ALLOWLIST]
    best: tuple[int, str] | None = None  # (token count, source_id)

    for candidate in _FUND_MENTION.findall(question or ""):
        tokens = set(_normalise(candidate).split())
        if not tokens:
            continue
        for source_id, known in specs:
            if not tokens <= known:
                continue
            # Prefer the tightest known name, so an abbreviation resolves to the
            # most specific scheme rather than the first that happens to fit.
            if best is None or len(known) < best[0]:
                best = (len(known), source_id)
    return best[1] if best else None


__all__ = [
    "search", "search_top", "embed_query", "rebuild_chunk", "jaccard",
    "closest_scheme_url", "detect_scheme_slug", "RetrievalError",
]
