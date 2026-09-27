"""STAGE 3b — vector store (architecture.md §4.4).

    score = 1.0 - distance for cosine space with normalized embeddings.

That single conversion is used everywhere in the codebase. Vectors are
L2-normalized, so cosine similarity equals the inner product, and ChromaDB's
``cosine`` space reports ``distance = 1 - cos_sim``. Getting this wrong in one
place makes scores silently inverted rather than obviously wrong.

Three things in here are load-bearing:

* **Primitive-only metadata.** ChromaDB rejects ``None``, lists and dicts. The
  ``heading_trail`` list is therefore flattened to a ``" > "``-delimited string
  on the way in and split back into a list on the way out.
* **Deterministic IDs + ``upsert``.** ``chunk_id`` is
  ``sha256(source_id + ":" + str(chunk_index))[:16]``, so re-ingesting the same
  corpus overwrites the same rows instead of duplicating them (FR-04). ``add``
  would raise or double-count; ``upsert`` is what makes idempotency real.
* **A content edit produces a new ID.** Since the ID is derived from
  ``(source_id, chunk_index)`` and not the content, editing a chunk's text does
  *not* change its ID, so an in-place ``upsert`` correctly replaces it. But
  *deleting* a chunk from the corpus leaves its old row behind as garbage, which
  is why ``--reset`` exists.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from ..config import ROOT, SETTINGS
from ..logging_utils import stage_timer
from ..models import Chunk

DB_PATH = ROOT / "chroma_db"
COLLECTION_NAME = str(SETTINGS.get("corpus.collection_name", "hdfc_faq"))
HEADING_DELIM = " > "

# Primitive-only. ChromaDB accepts str/int/float/bool and rejects everything else.
METADATA_FIELDS: tuple[str, ...] = (
    "source_id", "scheme_name", "scheme_slug", "category", "plan", "url",
    "page_title", "heading_trail", "chunk_index", "fetched_at", "char_len",
    "word_count", "content_hash",
)

_client: Any | None = None


class StoreError(RuntimeError):
    """The vector store could not be opened or written, with a remedy attached."""


def _require_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover
        raise StoreError(
            "chromadb is not installed.\n"
            "Remedy: pip install chromadb==0.5.23"
        ) from exc
    return chromadb


def get_client() -> Any:
    """One ``PersistentClient`` per process, pointed at ``./chroma_db``."""
    global _client
    if _client is not None:
        return _client
    chromadb = _require_chromadb()
    try:
        DB_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(DB_PATH))
    except Exception as exc:  # noqa: BLE001
        raise StoreError(
            f"could not open the persistent ChromaDB client at {DB_PATH}: {exc}\n"
            "Remedy: if the directory is corrupt, delete it and re-run "
            "`python -m app.ingest --stage store --reset`."
        ) from exc
    return _client


def _collection_metadata() -> dict[str, Any]:
    return {
        "hnsw:space": "cosine",
        "hnsw:M": 32,
        "hnsw:construction_ef": 200,
    }


def get_collection(reset: bool = False, name: str | None = None) -> Any:
    """Get or create the collection, optionally dropping it first.

    ``reset=True`` is the clean-rebuild path: it removes the collection outright
    rather than trying to reconcile rows, which is the only reliable way to shed
    rows whose chunks no longer exist in the corpus.
    """
    client = get_client()
    target = name or COLLECTION_NAME
    if reset:
        try:
            client.delete_collection(target)
        except Exception:  # noqa: BLE001 - absent collection is not an error
            pass
    try:
        return client.get_or_create_collection(
            target, metadata=_collection_metadata()
        )
    except Exception as exc:  # noqa: BLE001
        raise StoreError(
            f"could not open collection {target!r}: {exc}\n"
            "Remedy: python -m app.ingest --stage store --reset"
        ) from exc


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


def chunk_id_for(source_id: str, chunk_index: int) -> str:
    """``sha256(source_id + ":" + str(chunk_index))[:16]`` — the FR-04 upsert key."""
    return hashlib.sha256(
        f"{source_id}:{chunk_index}".encode("utf-8")
    ).hexdigest()[:16]


def to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Flatten a :class:`Chunk` into ChromaDB-legal primitives."""
    meta: dict[str, Any] = {
        "source_id": chunk.source_id,
        "scheme_name": chunk.scheme_name,
        "scheme_slug": chunk.scheme_slug,
        "category": chunk.category,
        "plan": chunk.plan,
        "url": chunk.url,
        "page_title": chunk.page_title,
        # list -> " > "-delimited str, because ChromaDB rejects lists
        "heading_trail": HEADING_DELIM.join(chunk.heading_trail),
        "chunk_index": int(chunk.chunk_index),
        "fetched_at": chunk.fetched_at,
        "char_len": int(chunk.char_len),
        "word_count": int(chunk.word_count),
        "content_hash": chunk.content_hash,
    }
    # Belt and braces: a None anywhere here raises deep inside ChromaDB with an
    # unhelpful message, so it is caught here instead.
    for key, value in meta.items():
        if value is None:
            meta[key] = ""
        elif not isinstance(value, (str, int, float, bool)):
            raise StoreError(
                f"metadata field {key!r} is {type(value).__name__}; ChromaDB "
                "accepts only str/int/float/bool"
            )
    return meta


def from_metadata(meta: dict[str, Any], document: str) -> dict[str, Any]:
    """Rebuild the chunk fields retrieval needs from a Chroma row.

    ``heading_trail`` comes back as a list again; ``text`` is the RAW stored
    document, which is what keeps citations clean for the user (the heading
    prefix only ever lived in the embedded text).
    """
    trail = meta.get("heading_trail") or ""
    return {
        "text": document,
        "source_id": meta.get("source_id", ""),
        "scheme_name": meta.get("scheme_name", ""),
        "scheme_slug": meta.get("scheme_slug", ""),
        "category": meta.get("category", ""),
        "plan": meta.get("plan", ""),
        "url": meta.get("url", ""),
        "page_title": meta.get("page_title", ""),
        "heading_trail": [p for p in str(trail).split(HEADING_DELIM) if p],
        "chunk_index": int(meta.get("chunk_index", 0) or 0),
        "fetched_at": meta.get("fetched_at", ""),
        "char_len": int(meta.get("char_len", 0) or 0),
        "word_count": int(meta.get("word_count", 0) or 0),
        "content_hash": meta.get("content_hash", ""),
    }


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #


def upsert_chunks(
    collection: Any,
    chunks: Sequence[Chunk],
    embeddings: dict[str, list[float]],
) -> int:
    """Upsert chunks by deterministic ID. Returns the number of rows written.

    Chunks with no embedding are skipped rather than embedded here, so the
    embedding model is never loaded by this module.
    """
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    vectors: list[list[float]] = []

    for chunk in chunks:
        vector = embeddings.get(chunk.chunk_id)
        if vector is None:
            continue
        # The stored ID is recomputed rather than trusted, so a hand-edited or
        # stale chunk_id in the jsonl cannot desynchronise the upsert key.
        ids.append(chunk_id_for(chunk.source_id, chunk.chunk_index))
        # documents get the RAW text: the heading prefix must never reach the user
        documents.append(chunk.text)
        metadatas.append(to_metadata(chunk))
        vectors.append(list(vector))

    if not ids:
        return 0

    with stage_timer("store", **{"in": len(chunks)}, out=len(ids), count=len(ids)) as box:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas,
                          embeddings=vectors)
        box["count"] = collection.count()
    return len(ids)


def reset_and_upsert(chunks: Sequence[Chunk], embeddings: dict[str, list[float]],
                     name: str | None = None) -> tuple[Any, int]:
    """Drop, recreate, and repopulate. The clean-rebuild path."""
    collection = get_collection(reset=True, name=name)
    written = upsert_chunks(collection, chunks, embeddings)
    print(f"  reset + wrote {written} chunks into {name or COLLECTION_NAME}",
          flush=True)
    return collection, written


def count(collection: Any | None = None) -> int:
    return (collection or get_collection()).count()


def peek(collection: Any | None = None, limit: int = 1) -> dict[str, Any]:
    return (collection or get_collection()).peek(limit=limit)


def reset_client_cache() -> None:
    """Drop the cached PersistentClient. Tests only."""
    global _client
    _client = None


__all__ = [
    "get_client", "get_collection", "upsert_chunks", "reset_and_upsert",
    "chunk_id_for", "to_metadata", "from_metadata", "count", "peek",
    "DB_PATH", "COLLECTION_NAME", "HEADING_DELIM", "METADATA_FIELDS",
    "StoreError", "reset_client_cache",
]
