"""Phase 3 gate: the vector store is idempotent and metadata is Chroma-legal.

FR-04 requires that re-running ingest does not grow the index. That is the whole
point of this file, so the tests are about *repetition*: every assertion runs
the same write twice and checks the second run changed nothing.

chromadb is imported lazily and every test skips cleanly without it, so this
file does not block a CPU-only install that has not pulled the store yet.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest

from app.models import Chunk
from app.pipeline import store as store_mod

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def temp_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An isolated PersistentClient in a tmp dir, torn down after the test."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        f"test_{uuid.uuid4().hex[:8]}",
        metadata=store_mod._collection_metadata(),
    )
    yield collection
    try:
        client.delete_collection(collection.name)
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


def make_chunks(n: int = 10) -> list[Chunk]:
    """``n`` synthetic chunks with realistic, distinct content."""
    schemes = ["hdfc-large-cap", "hdfc-equity", "hdfc-elss",
               "hdfc-small-cap", "hdfc-balanced-adv"]
    chunks: list[Chunk] = []
    for i in range(n):
        source_id = schemes[i % len(schemes)]
        text = f"Expense ratio of scheme {i} is {1 + i / 100:.2f}% plus facts."
        trail = [f"Scheme {source_id}", "Scheme facts", f"Row {i}"]
        chunks.append(
            Chunk(
                chunk_id=store_mod.chunk_id_for(source_id, i),
                text=text,
                embed_text=" > ".join(trail) + "\n\n" + text,
                source_id=source_id,
                scheme_name=f"Scheme {source_id}",
                scheme_slug=source_id,
                category="Equity",
                plan="Direct-Growth",
                url=f"https://groww.in/mutual-funds/{source_id}",
                page_title=f"Scheme {source_id} - Groww",
                heading_trail=trail,
                chunk_index=i,
                char_len=len(text),
                word_count=len(text.split()),
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                fetched_at="2026-09-27T00:00:00+00:00",
            )
        )
    return chunks


def fake_vectors(chunks: list[Chunk], dim: int = 384) -> dict[str, list[float]]:
    """Deterministic unit-ish vectors; no model required."""
    out: dict[str, list[float]] = {}
    for index, chunk in enumerate(chunks):
        vector = [0.0] * dim
        vector[index % dim] = 1.0
        out[chunk.chunk_id] = vector
    return out


# --------------------------------------------------------------------------- #
# 3.A.5 / Gate 3 — idempotency
# --------------------------------------------------------------------------- #


def test_upsert_twice_does_not_grow_the_collection(temp_collection):
    chunks = make_chunks(10)
    vectors = fake_vectors(chunks)

    first = store_mod.upsert_chunks(temp_collection, chunks, vectors)
    assert first == 10
    assert temp_collection.count() == 10

    second = store_mod.upsert_chunks(temp_collection, chunks, vectors)
    assert second == 10
    assert temp_collection.count() == 10, "FR-04: re-upsert must not duplicate rows"


def test_ids_are_deterministic_across_runs(temp_collection):
    """The ID is a pure function of (source_id, chunk_index), not of run order."""
    chunks = make_chunks(10)
    vectors = fake_vectors(chunks)
    store_mod.upsert_chunks(temp_collection, chunks, vectors)
    first_ids = set(temp_collection.get(include=[])["ids"])

    # A fresh set of equivalent chunks, built independently, must produce the
    # same IDs — this is what makes the corpus re-ingestable from scratch.
    rebuilt = make_chunks(10)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in rebuilt]

    store_mod.upsert_chunks(temp_collection, rebuilt, fake_vectors(rebuilt))
    assert set(temp_collection.get(include=[])["ids"]) == first_ids
    assert temp_collection.count() == 10


def test_chunk_id_matches_the_documented_formula():
    expected = hashlib.sha256(b"hdfc-elss:7").hexdigest()[:16]
    assert store_mod.chunk_id_for("hdfc-elss", 7) == expected
    assert len(store_mod.chunk_id_for("hdfc-elss", 7)) == 16
    # chunk_index 7 and "7" must not collide across types
    assert store_mod.chunk_id_for("hdfc-elss", 7) != store_mod.chunk_id_for("hdfc-elss:7", 0)


def test_edited_text_upserts_in_place(temp_collection):
    """Same (source_id, chunk_index) with new text must replace, not duplicate."""
    original = make_chunks(3)
    store_mod.upsert_chunks(temp_collection, original, fake_vectors(original))
    assert temp_collection.count() == 3

    edited = []
    for chunk in original:
        text = chunk.text + " Updated fee."
        edited.append(
            Chunk(**{**chunk.__dict__, "text": text, "char_len": len(text),
                     "content_hash": hashlib.sha256(text.encode()).hexdigest()})
        )
    store_mod.upsert_chunks(temp_collection, edited, fake_vectors(edited))
    assert temp_collection.count() == 3
    stored = temp_collection.get(include=["documents"])["documents"]
    assert any("Updated fee." in doc for doc in stored)


# --------------------------------------------------------------------------- #
# Gate 3 — metadata legality and round-trip
# --------------------------------------------------------------------------- #


def test_metadata_is_primitive_only(temp_collection):
    chunks = make_chunks(3)
    store_mod.upsert_chunks(temp_collection, chunks, fake_vectors(chunks))

    sample = temp_collection.peek(limit=1)
    meta = sample["metadatas"][0]
    for key, value in meta.items():
        assert isinstance(value, (str, int, float, bool)), (
            f"{key} is {type(value).__name__}; ChromaDB rejects this"
        )
        assert value is not None
    # Gate 3 requires these two specifically.
    assert meta["url"]
    assert meta["heading_trail"]


def test_heading_trail_round_trips_through_the_delimiter(temp_collection):
    chunks = make_chunks(3)
    store_mod.upsert_chunks(temp_collection, chunks, fake_vectors(chunks))

    stored = temp_collection.get(include=["metadatas", "documents"])
    meta = stored["metadatas"][0]
    rebuilt = store_mod.from_metadata(meta, stored["documents"][0])

    assert isinstance(rebuilt["heading_trail"], list)
    assert rebuilt["heading_trail"] == chunks[0].heading_trail
    assert rebuilt["text"] == chunks[0].text
    assert rebuilt["url"] == chunks[0].url
    assert rebuilt["chunk_index"] == chunks[0].chunk_index


def test_stored_document_is_the_raw_text_not_the_embed_text(temp_collection):
    """Citations must never show the heading prefix."""
    chunks = make_chunks(2)
    store_mod.upsert_chunks(temp_collection, chunks, fake_vectors(chunks))
    stored = temp_collection.get(include=["documents"])["documents"]
    for document in stored:
        assert " > " not in document
        assert not document.startswith("Scheme hdfc-")


def test_to_metadata_rejects_non_primitive_types():
    chunk = make_chunks(1)[0]
    broken = Chunk(**{**chunk.__dict__, "category": {"nested": "dict"}})
    with pytest.raises(store_mod.StoreError, match="ChromaDB accepts only"):
        store_mod.to_metadata(broken)


def test_to_metadata_replaces_none_with_empty_string():
    chunk = make_chunks(1)[0]
    broken = Chunk(**{**chunk.__dict__, "page_title": None})
    meta = store_mod.to_metadata(broken)
    assert meta["page_title"] == ""


def test_chunks_without_embeddings_are_skipped(temp_collection):
    chunks = make_chunks(5)
    partial = fake_vectors(chunks[:2])
    written = store_mod.upsert_chunks(temp_collection, chunks, partial)
    assert written == 2
    assert temp_collection.count() == 2


# --------------------------------------------------------------------------- #
# Cache-key behaviour (3.A.3) — no chromadb needed for the pure part
# --------------------------------------------------------------------------- #


def test_cache_key_is_content_addressed_not_positional():
    from app.pipeline.embedder import cache_key

    a = cache_key("hash-abc")
    b = cache_key("hash-abc")
    c = cache_key("hash-xyz")
    assert a == b, "same content must hit the same cache entry"
    assert a != c


def test_cache_key_includes_the_model_id():
    from app.pipeline.embedder import cache_key

    # Switching models must invalidate the cache, not mix incompatible vectors.
    assert cache_key("same-hash", "model-a") != cache_key("same-hash", "model-b")


def test_cache_key_is_a_filename_safe_hex_digest():
    from app.pipeline.embedder import cache_key

    key = cache_key("hash-abc")
    assert len(key) == 64
    assert all(ch in "0123456789abcdef" for ch in key)
