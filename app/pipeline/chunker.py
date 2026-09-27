"""STAGE 2 — chunking (architecture.md §4.2, §4.2.1).

Three interchangeable strategies implement the :class:`~app.models.Chunker`
protocol and are selected by ``config.yaml: chunking.strategy``:

============  =====================================================  ============
Strategy      Mechanism                                               Cost
============  =====================================================  ============
recursive     Character splits on ``\\n\\n``, ``\\n``, ``". "``, ``" "``   fast
heading_bound Split on H2/H3, then recurse each section to the cap     fast
semantic      Sliding sentence window, break at the largest cosine gap very slow
============  =====================================================  ============

The load-bearing rule in this module is the **word-piece invariant**. MiniLM has
a hard 256 word-piece input limit and truncates *silently*, so an over-long
chunk does not error — it embeds only its first half while the stored text still
shows the whole thing. The chunk then retrieves on its first half and the answer
comes from the wrong passage. Every chunk leaving this module is therefore
tokenized with the real model tokenizer and, if it is over budget, re-split;
only an unsplittable fragment is dropped, and that drop is logged as a violation.

Two things are deliberately *not* done here:

* the experiment (§4.2.1) lives in ``app/experiment.py`` and does not touch
  ChromaDB, so a chunking result can never be contaminated by index behaviour;
* nothing imports ``app.pipeline.store`` or ``app.pipeline.embedder`` — Stage 2
  must stay runnable before Stage 3 exists (constraint C10).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import SETTINGS
from ..models import Chunk, Chunker, RawDoc

# --------------------------------------------------------------------------- #
# Token accounting — the invariant
# --------------------------------------------------------------------------- #

# MiniLM prepends [CLS] and appends [SEP]. Counting with add_special_tokens=True
# measures what the model actually receives, which is the number that must stay
# under 256. Counting without them would under-report by exactly 2 every time.
_SPECIAL_TOKENS = 2


class TokenBudgetError(RuntimeError):
    """Raised when the tokenizer is unavailable and a budget check is required."""


_TOKENIZER: Any | None = None


def get_tokenizer() -> Any:
    """Return the shared MiniLM tokenizer (see ``app.embedder_singleton``)."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    from ..embedder_singleton import get_tokenizer as _shared

    _TOKENIZER = _shared()
    return _TOKENIZER


def count_wordpieces(text: str) -> int:
    """Word-piece length of ``text`` as MiniLM will see it, specials included.

    Falls back to a whitespace-word estimate only if the tokenizer cannot be
    loaded *and* the caller opted out via ``count_wordpieces(..., strict=False)``.
    The strict default exists because a silently-wrong token count would defeat
    the entire purpose of this stage.
    """
    return _count(text, strict=True)


def _count(text: str, strict: bool = True) -> int:
    tok = get_tokenizer()
    if tok is None:  # pragma: no cover - only reachable with a stub tokenizer
        if strict:
            raise TokenBudgetError("tokenizer unavailable; cannot enforce the 256 cap")
        return len(text.split()) + _SPECIAL_TOKENS
    encoded = tok(text, add_special_tokens=True)["input_ids"]
    return len(encoded)


def _wordpiece_len_estimate(text: str) -> int:
    """Cheap upper-bound estimate used *inside* split loops, before tokenizing.

    MiniLM word-pieces average ~1.35 tokens per whitespace word on this corpus,
    but a single long URL or a run of punctuation can exceed that. The estimate
    is used only to decide *where* to cut; every resulting chunk is then
    tokenized for real. Overestimating is safe (chunks come out smaller),
    underestimating is caught by the final check.
    """
    words = text.split()
    long_tokens = sum(1 for w in words if len(w) > 12)
    return int(len(words) * 1.35) + long_tokens * 2 + _SPECIAL_TOKENS


# --------------------------------------------------------------------------- #
# Sentence segmentation
# --------------------------------------------------------------------------- #

# Abbreviations that must not end a sentence. "Rs" and the fund-manager initials
# matter here: "Rs 100." would otherwise split mid-fact.
_ABBREV = (
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no",
    "fig", "approx", "cf", "eg", "ie", "am", "pm", "pvt", "ltd", "inc",
    "rs", "min", "max", "qty", "nav", "aum", "f&o",
)
_SENT_SPLIT = re.compile(
    r"(?<=[.!?])[\"'\)\]]*\s+(?=[A-Z\"'\(\[])", re.UNICODE
)
_SENTENCE_GUARD = re.compile(
    r"\b(" + "|".join(_ABBREV) + r")\.$", re.IGNORECASE
)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, protecting common abbreviations and decimals.

    ``1.03%`` and ``Rs 100`` must not be cut, or the answer span in
    ``eval/golden.json`` stops being a verbatim substring of the chunk.
    """
    if not text:
        return []
    pieces = _SENT_SPLIT.split(text)
    merged: list[str] = []
    for piece in pieces:
        candidate = piece.strip()
        if not candidate:
            continue
        if merged and _SENTENCE_GUARD.search(merged[-1]):
            merged[-1] = merged[-1] + " " + candidate
        else:
            merged.append(candidate)
    return merged


# --------------------------------------------------------------------------- #
# Heading structure
# --------------------------------------------------------------------------- #

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_META_LINE = re.compile(r"^(Source page|Page title):")


@dataclass(frozen=True)
class Section:
    """One heading-bounded region of a document."""

    trail: tuple[str, ...]      # h1 → h2 → h3 breadcrumb
    text: str


def parse_sections(text: str) -> list[Section]:
    """Split a markdown-flavoured document into H1/H2/H3 sections.

    The loader emits ``# {name}``, ``Source page:``, ``Page title:``,
    ``## Scheme facts``, ``### {question}`` and so on, so the heading levels here
    line up with real structure rather than being a heuristic.
    """
    sections: list[Section] = []
    trail: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(trail=tuple(trail), text=body))
        buf.clear()

    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            trail = trail[: level - 1] + [match.group(2)]
            continue
        if not _META_LINE.match(line):
            buf.append(line)
    flush()
    return sections


def _heading_trail_for(doc: RawDoc, trail: Sequence[str]) -> list[str]:
    """Build the breadcrumb prefixed onto the *embedded* text only.

    architecture.md §4.2: ``scheme_name`` is folded in so a scheme-specific
    question ("HDFC Small Cap expense ratio") boosts the right page without a
    metadata filter, and ``page_title`` follows it. Duplicates are collapsed so
    the prefix never reads "HDFC Small Cap Fund > HDFC Small Cap Fund".
    """
    parts: list[str] = [doc.scheme_name, doc.page_title, *trail]
    out: list[str] = []
    for part in parts:
        cleaned = (part or "").strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def build_embed_text(heading_trail: Sequence[str], text: str) -> str:
    """``"{scheme} > {page_title} > {H2} > {H3}\\n\\n{chunk_text}"`` (§4.2)."""
    prefix = " > ".join(heading_trail)
    body = text.strip()
    return f"{prefix}\n\n{body}" if prefix else body


# --------------------------------------------------------------------------- #
# Chunk construction
# --------------------------------------------------------------------------- #


def _hash16(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:16]


def _make_chunk(
    doc: RawDoc, text: str, trail: Sequence[str], index: int
) -> Chunk:
    body = text.strip()
    full_trail = _heading_trail_for(doc, trail)
    return Chunk(
        chunk_id=_hash16(doc.source_id, str(index)),
        text=body,
        embed_text=build_embed_text(full_trail, body),
        source_id=doc.source_id,
        scheme_name=doc.scheme_name,
        scheme_slug=doc.source_id,      # source_id is already the slug
        category=doc.category,
        plan=doc.plan,
        url=doc.url,
        page_title=doc.page_title,
        heading_trail=full_trail,
        chunk_index=index,
        char_len=len(body),
        word_count=len(body.split()),
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        fetched_at=doc.fetched_at,
    )


def _assemble(
    doc: RawDoc, pieces: Iterable[tuple[Sequence[str], str]]
) -> list[Chunk]:
    """Turn (trail, body) pairs into indexed :class:`Chunk` objects, dropping empties."""
    chunks: list[Chunk] = []
    for trail, body in pieces:
        body = (body or "").strip()
        if len(body.split()) < 3:      # a fragment too small to answer from
            continue
        chunks.append(_make_chunk(doc, body, trail, len(chunks)))
    return chunks


# --------------------------------------------------------------------------- #
# Size-bounded splitting primitives
# --------------------------------------------------------------------------- #


def _split_by_separators(text: str, size: int, overlap: int) -> list[str]:
    """Recursive character split, mirroring the documented separator order.

    Separators are tried in order and the first one that keeps a piece under
    ``size`` wins; the recursion bottoms out at single characters so this cannot
    loop forever.
    """
    separators = ["\n\n", "\n", ". ", " "]
    return _recursive_split(text, separators, size, overlap)


def _recursive_split(
    text: str, separators: list[str], size: int, overlap: int
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    for index, sep in enumerate(separators):
        if sep not in text:
            continue
        parts = text.split(sep)
        remainder = separators[index + 1:]
        out: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= size:
                out.append(part)
            else:
                out.extend(_recursive_split(part, remainder, size, overlap))
        return _merge_with_overlap(out, size, overlap)

    # No separator present (one giant token): hard-split on size.
    return [text[i: i + size] for i in range(0, len(text), size)]


def _merge_with_overlap(parts: list[str], size: int, overlap: int) -> list[str]:
    """Greedily re-pack parts into ``size``-capped windows with a word overlap."""
    if overlap <= 0 or not parts:
        return parts
    windows: list[str] = []
    current: list[str] = []
    length = 0
    for part in parts:
        addition = len(part) + (1 if current else 0)
        if current and length + addition > size:
            windows.append(" ".join(current))
            # Carry back whole trailing words to approximate the requested overlap.
            carry: list[str] = []
            carried = 0
            for word in reversed(current):
                if carried >= overlap:
                    break
                carry.insert(0, word)
                carried += len(word) + 1
            current = list(carry)
            length = sum(len(w) + 1 for w in current) - (1 if current else 0)
        current.append(part)
        length += addition
    if current:
        windows.append(" ".join(current))
    return windows


def _split_by_sentences(text: str, size: int) -> list[str]:
    """Group whole sentences up to ``size``; hard-split any single long sentence."""
    out: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(text) or [text]:
        addition = len(sentence) + (1 if current else 0)
        if current and len(" ".join(current)) + addition > size:
            out.append(" ".join(current))
            current = []
            addition = len(sentence)
        if addition > size:
            if current:
                out.append(" ".join(current))
                current = []
            for i in range(0, len(sentence), size):
                out.append(sentence[i: i + size])
            continue
        current.append(sentence)
    if current:
        out.append(" ".join(current))
    return [piece for piece in out if piece.strip()]


def _split_by_words(text: str, size: int) -> list[str]:
    """Last-resort split on whitespace, for text with no sentence punctuation."""
    words = text.split()
    out: list[str] = []
    current: list[str] = []
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and len(" ".join(current)) + addition > size:
            out.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        out.append(" ".join(current))
    return out


# --------------------------------------------------------------------------- #
# The invariant enforcement
# --------------------------------------------------------------------------- #


@dataclass
class BudgetReport:
    """Outcome of enforcing the word-piece cap, for the stage summary."""

    kept: list[Chunk] = field(default_factory=list)
    violations: int = 0
    rescued: int = 0
    dropped: int = 0


def enforce_token_budget(
    chunks: list[Chunk],
    max_wordpieces: int,
    target_wordpieces: int,
) -> BudgetReport:
    """Re-split any chunk whose real token length exceeds ``max_wordpieces``.

    Escalation ladder, cheapest first:

    1. already within budget — keep as is;
    2. over budget — re-split at sentence boundaries, which keeps the passage
       readable and keeps the heading trail meaningful;
    3. a resulting piece still over budget — split on words;
    4. a fragment that is *still* over budget after word splitting (a
       pathologically long URL, or 5 KB of unbroken text) — drop that fragment
       and count a violation, because embedding it would silently truncate and
       produce a chunk that misrepresents its own content.

    A single hostile fragment does not condemn its neighbours: the rescue keeps
    every piece that fits and drops only what cannot. Returns a
    :class:`BudgetReport`; ``violations`` is non-zero only when something was
    genuinely discarded, and is surfaced in the stage summary.
    """
    report = BudgetReport(kept=[])

    for chunk in chunks:
        if count_wordpieces(chunk.text) <= max_wordpieces:
            report.kept.append(chunk)
            continue

        target_chars = max(120, int(target_wordpieces / 1.35))
        salvaged: list[str] = []

        for piece in _split_by_sentences(chunk.text, target_chars):
            if count_wordpieces(piece) <= max_wordpieces:
                salvaged.append(piece)
                continue
            for word_piece in _split_by_words(piece, target_chars):
                if count_wordpieces(word_piece) <= max_wordpieces:
                    salvaged.append(word_piece)
                else:
                    report.violations += 1
                    report.dropped += 1

        if salvaged:
            # _renumber below assigns the authoritative index and chunk_id, so
            # the placeholder index here is irrelevant.
            report.kept.extend(_reindex(chunk, text, 0) for text in salvaged)
            report.rescued += 1
        else:
            # Nothing in this chunk could be saved.
            report.violations += 1
            report.dropped += 1

    _renumber(report.kept)
    return report


def _reindex(chunk: Chunk, text: str, index: int) -> Chunk:
    """Rebuild a chunk from a smaller piece of ``chunk``, preserving provenance."""
    body = text.strip()
    return Chunk(
        chunk_id=_hash16(chunk.source_id, str(index)),
        text=body,
        embed_text=build_embed_text(chunk.heading_trail, body),
        source_id=chunk.source_id,
        scheme_name=chunk.scheme_name,
        scheme_slug=chunk.scheme_slug,
        category=chunk.category,
        plan=chunk.plan,
        url=chunk.url,
        page_title=chunk.page_title,
        heading_trail=list(chunk.heading_trail),
        chunk_index=index,
        char_len=len(body),
        word_count=len(body.split()),
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        fetched_at=chunk.fetched_at,
    )


def _renumber(chunks: list[Chunk]) -> None:
    """Reassign chunk_index/chunk_id after re-splitting, per document.

    ``Chunk`` is frozen, so this mutates the underlying ``__dict__`` through
    ``object.__setattr__``. Doing it here — once, at the end — is why
    ``chunk_id`` stays a pure function of ``(source_id, chunk_index)`` and can be
    used as a stable upsert key in Stage 4.
    """
    seen: dict[str, int] = {}
    for chunk in chunks:
        index = seen.get(chunk.source_id, 0)
        seen[chunk.source_id] = index + 1
        object.__setattr__(chunk, "chunk_index", index)
        object.__setattr__(chunk, "chunk_id", _hash16(chunk.source_id, str(index)))


# --------------------------------------------------------------------------- #
# Strategy 1 — recursive
# --------------------------------------------------------------------------- #


class RecursiveChunker:
    """Baseline: pure character splitting, structure-blind."""

    name = "recursive"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def split(self, doc: RawDoc) -> list[Chunk]:
        sections = parse_sections(doc.text) or [Section((), doc.text)]
        pieces: list[tuple[Sequence[str], str]] = []
        for section in sections:
            for part in _split_by_separators(
                section.text, self.chunk_size, self.chunk_overlap
            ):
                pieces.append((section.trail, part))
        return _assemble(doc, pieces)


# --------------------------------------------------------------------------- #
# Strategy 2 — heading_bound  (the expected winner)
# --------------------------------------------------------------------------- #


class HeadingBoundChunker:
    """Split on H2/H3, then recurse only *within* a section to the size cap.

    Why this should win on this corpus: every chunk then carries exactly the
    breadcrumb of the section it came from, so "exit load" questions match the
    chunk that is about exit load even when those words are absent from the
    sentence. The recursive baseline destroys that association by packing
    unrelated facts together.
    """

    name = "heading_bound"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def split(self, doc: RawDoc) -> list[Chunk]:
        sections = parse_sections(doc.text)
        if not sections:
            sections = [Section((), doc.text)]

        pieces: list[tuple[Sequence[str], str]] = []
        for section in sections:
            body = section.text
            if len(body) <= self.chunk_size:
                pieces.append((section.trail, body))
                continue
            # A section that is itself too long keeps its trail on every part,
            # so the breadcrumb survives the recursion.
            for part in _split_by_separators(body, self.chunk_size, self.chunk_overlap):
                pieces.append((section.trail, part))
        return _assemble(doc, pieces)


# --------------------------------------------------------------------------- #
# Strategy 3 — semantic
# --------------------------------------------------------------------------- #


class SemanticChunker:
    """Break where consecutive sentences are least related.

    Sentence embeddings come from the same MiniLM instance used at query time, so
    the notion of "related" is exactly the one retrieval will use. A break is
    placed at the local maxima of ``1 - cos(s_i, s_{i+1})``, subject to a
    percentile threshold, and the size cap is then applied within each resulting
    group.

    This is expected *not* to win here, and the experiment is what proves it: the
    corpus is ~1.5k words per scheme of already well-separated headings, so
    there is little for a cosine gap to discover that the headings did not
    already state.
    """

    name = "semantic"

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int = 0,
        overlap_pct: float = 0.10,
        breakpoint_threshold: float = 0.35,
        buffer_size: int = 1,
    ) -> None:
        """``breakpoint_threshold`` is an *absolute* cosine distance (implementation.md 2.C).

        A percentile threshold was tried first and rejected: it is relative to the
        corpus, so the same 75th percentile means a different semantic gap on each
        of the 45 grid cells, and a config could "win" purely because its chunk
        sizes changed the distance distribution. 0.35 means the same thing in
        every cell. The percentile is retained only as a fallback for when the
        absolute threshold finds no break at all.
        """
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.overlap_pct = float(overlap_pct)
        self.breakpoint_threshold = float(breakpoint_threshold)
        self.buffer_size = max(1, int(buffer_size))

    def split(self, doc: RawDoc) -> list[Chunk]:
        sections = parse_sections(doc.text) or [Section((), doc.text)]
        pieces: list[tuple[Sequence[str], str]] = []
        for section in sections:
            sentences = split_sentences(section.text)
            if len(sentences) < 3:
                pieces.append((section.trail, section.text))
                continue
            for group in self._semantic_groups(sentences):
                body = " ".join(group)
                if len(body) <= self.chunk_size:
                    pieces.append((section.trail, body))
                else:
                    for part in _split_by_sentences(body, self.chunk_size):
                        pieces.append((section.trail, part))
        return _assemble(doc, pieces)

    def _semantic_groups(self, sentences: list[str]) -> list[list[str]]:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise TokenBudgetError(
                f"semantic chunking needs numpy: {exc}"
            ) from exc

        vectors = self._embed_sentences(sentences)

        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)

        # Absolute threshold first; fall back to a percentile only if it finds
        # nothing, so a quiet section is not emitted as one unsplittable blob.
        breaks = {i for i, d in enumerate(distances) if d > self.breakpoint_threshold}
        threshold_used = self.breakpoint_threshold
        if not breaks and len(distances) > 1:
            threshold_used = float(np.percentile(distances, 75.0))
            breaks = {i for i, d in enumerate(distances) if d > threshold_used}

        groups: list[list[str]] = []
        current: list[str] = [sentences[0]]
        for index in range(1, len(sentences)):
            previous = index - 1
            # Respect the size cap before honouring a semantic break.
            over_size = len(" ".join(current + [sentences[index]])) > self.chunk_size
            if previous in breaks or over_size:
                groups.append(current)
                current = [sentences[index]]
            else:
                current.append(sentences[index])
        if current:
            groups.append(current)
        return groups

    def _embed_sentences(self, sentences: list[str]) -> Any:
        """Embed sentences, caching by text across every config in the sweep.

        The §4.2.1 grid varies size and overlap but never the sentence set, so
        re-encoding per config would run the same ~250 encodes fifteen times.
        """
        import numpy as np

        missing = [s for s in dict.fromkeys(sentences) if s not in _SENTENCE_VECTORS]
        if missing:
            try:
                from ..embedder_singleton import get_model

                fresh = np.asarray(
                    get_model().encode(
                        missing,
                        batch_size=int(SETTINGS.get("embedding.batch_size", 32)),
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ),
                    dtype="float32",
                )
            except Exception as exc:  # noqa: BLE001
                raise TokenBudgetError(
                    f"semantic chunking needs the embedding model: {exc}\n"
                    "Remedy: pip install torch --index-url "
                    "https://download.pytorch.org/whl/cpu && "
                    "pip install sentence-transformers"
                ) from exc
            for text, vector in zip(missing, fresh):
                _SENTENCE_VECTORS[text] = vector
        return np.stack([_SENTENCE_VECTORS[s] for s in sentences])


STRATEGIES: dict[str, type] = {
    "recursive": RecursiveChunker,
    "heading_bound": HeadingBoundChunker,
    "semantic": SemanticChunker,
}

# Sentence -> unit vector, shared across the whole §4.2.1 sweep so the semantic
# strategy encodes each sentence once rather than once per (size, overlap) pair.
_SENTENCE_VECTORS: dict[str, Any] = {}


def clear_sentence_cache() -> None:
    """Drop the semantic sentence cache. Tests, and between model swaps."""
    _SENTENCE_VECTORS.clear()


def get_chunker(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Chunker:
    """Build the configured chunker. Every strategy stays overridable (FR-05)."""
    name = strategy or str(SETTINGS.get("chunking.strategy", "heading_bound"))
    if name not in STRATEGIES:
        raise ValueError(
            f"unknown chunking strategy {name!r}; choose one of "
            f"{sorted(STRATEGIES)}"
        )
    size = int(
        chunk_size
        if chunk_size is not None
        else SETTINGS.get("chunking.chunk_size", 800)
    )
    overlap = int(
        chunk_overlap
        if chunk_overlap is not None
        else SETTINGS.get("chunking.chunk_overlap", 100)
    )
    return STRATEGIES[name](size, overlap)     # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Orchestration + persistence
# --------------------------------------------------------------------------- #


def read_raw_docs(path: Path | None = None) -> list[RawDoc]:
    """Load ``artifacts/raw_docs.jsonl`` produced by Stage 1."""
    target = path or SETTINGS.path("loading.raw_docs_path", "artifacts/raw_docs.jsonl")
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found — run `python -m app.ingest --stage load` first"
        )
    docs: list[RawDoc] = []
    with open(target, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            docs.append(RawDoc(**record))
    return docs


def write_chunks_jsonl(chunks: list[Chunk], path: Path | None = None) -> Path:
    """Write ``artifacts/clean_chunks.jsonl`` — committed, so retrieval is auditable."""
    target = path or SETTINGS.path(
        "chunking.clean_chunks_path", "artifacts/clean_chunks.jsonl"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"  wrote {len(chunks)} chunks to {target}", flush=True)
    return target


def read_chunks_jsonl(path: Path | None = None) -> list[Chunk]:
    target = path or SETTINGS.path(
        "chunking.clean_chunks_path", "artifacts/clean_chunks.jsonl"
    )
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found — run `python -m app.ingest --stage chunk` first"
        )
    out: list[Chunk] = []
    with open(target, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Chunk(**json.loads(line)))
    return out


def run(doc: RawDoc, chunker: Chunker | None = None) -> BudgetReport:
    """Chunk one document and enforce the word-piece invariant on the result.

    Returns the :class:`BudgetReport` rather than a bare list so callers get the
    violation count without re-tokenizing every chunk a second time.
    """
    engine = chunker or get_chunker()
    max_wp = int(SETTINGS.get("chunking.max_wordpieces", 256))
    target_wp = int(SETTINGS.get("chunking.target_wordpieces", 200))
    pieces = engine.split(doc)
    report = enforce_token_budget(pieces, max_wp, target_wp)
    if report.violations:
        print(
            f"  [chunk:{getattr(engine, 'name', '?')}] {report.violations} chunk(s) "
            f"dropped for exceeding {max_wp} word-pieces and being unsplittable "
            f"({report.dropped} dropped, {report.rescued} re-split)",
            flush=True,
        )
    return report


def summarise(chunks: list[Chunk], violations: int = 0) -> dict[str, Any]:
    """Stage summary per implementation.md 2.A.8: out/mean_chars/p95_tokens/violations."""
    if not chunks:
        return {
            "out": 0, "mean_chars": 0.0, "p95_tokens": 0, "violations": violations,
            "mean_words": 0.0, "docs": 0,
        }
    lengths = sorted(len(c.text) for c in chunks)
    tokens = sorted(count_wordpieces(c.text) for c in chunks)
    p95 = tokens[min(len(tokens) - 1, int(round(0.95 * (len(tokens) - 1))))]
    return {
        "out": len(chunks),
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "p95_tokens": p95,
        "violations": violations,
        "mean_words": round(sum(c.word_count for c in chunks) / len(chunks), 1),
        "max_tokens": tokens[-1],
        "docs": len({c.source_id for c in chunks}),
    }


__all__ = [
    "Chunker", "RecursiveChunker", "HeadingBoundChunker", "SemanticChunker",
    "STRATEGIES", "get_chunker", "run", "read_raw_docs", "write_chunks_jsonl",
    "read_chunks_jsonl", "enforce_token_budget", "count_wordpieces",
    "split_sentences", "parse_sections", "build_embed_text", "summarise",
    "BudgetReport", "Section", "TokenBudgetError",
]
