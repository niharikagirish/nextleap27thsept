"""Cross-module data types — the only shared vocabulary in the system.

Per architecture.md §3, every other module may import from here, and
``app/pipeline/*`` and ``app/guards/*`` must not import each other. That rule is
what keeps each RAG stage independently testable (constraint C10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(frozen=True)
class SourceSpec:
    """One allowlisted public page (constraint C1)."""
    source_id: str
    scheme_name: str
    category: str
    plan: str
    url: str


@dataclass(frozen=True)
class RawDoc:
    """STAGE 1 output: cleaned plain text for one page, with full provenance."""
    source_id: str
    scheme_name: str
    category: str
    plan: str
    url: str
    page_title: str
    text: str
    fetched_at: str          # ISO-8601 UTC, timezone-aware
    http_status: int
    content_hash: str        # sha256 of the cleaned text

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Chunk:
    """STAGE 2 output: one retrievable passage, carrying full provenance.

    ``text`` and ``embed_text`` differ on purpose (architecture.md §9 A4): the
    heading trail is prefixed onto the *embedded* text only, to sharpen
    retrieval, while the stored/rendered text stays clean for the user.
    """
    chunk_id: str            # sha256(source_id + ":" + str(chunk_index))[:16]
    text: str                # raw passage, no heading prefix
    embed_text: str          # heading trail + "\n\n" + text
    source_id: str
    scheme_name: str
    scheme_slug: str
    category: str
    plan: str
    url: str
    page_title: str
    heading_trail: list[str]
    chunk_index: int
    char_len: int
    word_count: int
    content_hash: str
    fetched_at: str


@dataclass(frozen=True)
class RetrievedChunk:
    """STAGE 8 output: a chunk plus its similarity score and rank."""
    chunk: Chunk
    score: float             # cosine similarity == 1.0 - chroma distance
    rank: int


@dataclass(frozen=True)
class Answer:
    """STAGE 11 output — the only thing the UI and eval script consume.

    Invariants are enforced in ``app/pipeline/validators.py`` (Phase 6), not by
    convention:

    * ``kind == "factual"`` implies ``len(text)`` holds at most 3 sentences (C6)
    * ``kind == "factual"`` implies ``len(sources) >= 1`` (C5)
    * every URL in ``sources`` was present in the retrieved set
    """
    text: str
    sources: list[str]
    last_updated: str
    kind: Literal["factual", "refusal", "abstain", "perf_redirect",
                  "pii_notice", "error"]
    reason: str = ""
    debug: list[RetrievedChunk] = field(default_factory=list)


class Chunker(Protocol):
    """STAGE 2 interface. Three implementations ship; ``config.yaml`` picks one."""

    def split(self, doc: RawDoc) -> list[Chunk]: ...


__all__ = [
    "SourceSpec", "RawDoc", "Chunk", "RetrievedChunk", "Answer", "Chunker",
]
