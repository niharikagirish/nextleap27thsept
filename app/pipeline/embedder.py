"""STAGE 3a — embedding (architecture.md §4.3).

The one job that matters here is the **content-keyed disk cache**. The cache key
is ``sha256(model_id + "::" + content_hash)``, not a chunk index, so an unchanged
chunk is never re-embedded. That is what makes NFR-02 (warm ingest under 20 s)
achievable on CPU: the first run pays ~90 MB of model download plus ~150
MiniLM encodes, and every run after that is a directory of ``.npy`` reads.

Two rules this module exists to enforce:

* **One model instance.** ``app.embedder_singleton`` owns the
  ``SentenceTransformer``. This module imports it and never constructs a second
  one. Two instances with different ``normalize`` settings produce an index
  that loads fine and retrieves nonsense, which is close to undebuggable.
* **The same normalization on both sides.** ``embed_chunks`` and ``embed_query``
  both pass ``normalize_embeddings=True``, so cosine similarity is a plain dot
  product and ``score = 1.0 - distance`` in the store is exact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import SETTINGS
from ..logging_utils import stage_timer
from ..models import Chunk

CACHE_DIR = SETTINGS.path("embedding.cache_dir", "cache/embeddings")
BATCH_SIZE = int(SETTINGS.get("embedding.batch_size", 32))
EXPECTED_DIM = 384          # all-MiniLM-L6-v2, fixed by the class brief


class EmbeddingError(RuntimeError):
    """The model could not produce embeddings, with a usable remedy attached."""


@dataclass
class EmbedStats:
    """Cache accounting for the stage summary (NFR-02)."""

    total: int = 0
    hits: int = 0
    misses: int = 0

    def as_fields(self) -> dict[str, int]:
        return {"in": self.total, "hits": self.hits, "misses": self.misses,
                "cached": self.hits}


def _model_id() -> str:
    return str(SETTINGS.get("embedding.model_id", "sentence-transformers/all-MiniLM-L6-v2"))


def cache_key(content_hash: str, model_id: str | None = None) -> str:
    """``sha256(model_id + "::" + content_hash)`` — content-addressed, not positional.

    Including the model id means switching models invalidates every entry instead
    of silently mixing 384-dim vectors from two different models in one index.
    """
    return hashlib.sha256(
        f"{model_id or _model_id()}::{content_hash}".encode("utf-8")
    ).hexdigest()


def _cache_path(content_hash: str) -> Path:
    return CACHE_DIR / f"{cache_key(content_hash)}.npy"


def _encode(texts: Sequence[str], model: Any) -> np.ndarray:
    """Encode with the shared singleton, L2-normalized, as float32."""
    if not texts:
        return np.zeros((0, EXPECTED_DIM), dtype="float32")
    try:
        vectors = model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > BATCH_SIZE,
            convert_to_numpy=True,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with a remedy
        raise EmbeddingError(
            f"encoding {len(texts)} text(s) failed: {exc}\n"
            "Remedy: check that torch is installed CPU-only with\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2 or array.shape[1] != EXPECTED_DIM:
        raise EmbeddingError(
            f"expected {EXPECTED_DIM}-dimensional embeddings, got shape {array.shape}. "
            f"The configured model_id is {_model_id()!r}; the class brief fixes the "
            "embedding model at all-MiniLM-L6-v2 (384-dim)."
        )
    return array


def embed_chunks(chunks: Sequence[Chunk], use_cache: bool = True) -> dict[str, list[float]]:
    """Embed ``chunk.embed_text``, returning ``{chunk_id: vector}``.

    Note this embeds ``embed_text`` (heading trail + body), not ``text``. The
    heading trail is what lets a question about exit load match the chunk that
    is *about* exit load even when those words are absent from the sentence
    (architecture.md §4.2). The stored document remains the raw ``text``.
    """
    if not chunks:
        return {}

    stats = EmbedStats(total=len(chunks))
    results: dict[str, list[float]] = {}

    with stage_timer("embed", **{"in": len(chunks)}, dim=EXPECTED_DIM) as box:
        pending: list[Chunk] = []
        for chunk in chunks:
            if chunk.chunk_id in results:
                continue
            path = _cache_path(chunk.content_hash)
            if use_cache and path.exists():
                try:
                    results[chunk.chunk_id] = np.load(path).tolist()
                    stats.hits += 1
                    continue
                except (OSError, ValueError):
                    # A truncated or corrupt cache entry is a cache miss, not a
                    # crash: the correct response is to re-embed and overwrite.
                    path.unlink(missing_ok=True)
            pending.append(chunk)

        # Identical content in two chunks shares one cache entry, so encode each
        # distinct body once and fan the vector back out.
        by_hash: dict[str, list[Chunk]] = {}
        for chunk in pending:
            by_hash.setdefault(chunk.content_hash, []).append(chunk)

        if by_hash:
            hashes = list(by_hash)
            vectors = _encode([by_hash[h][0].embed_text for h in hashes], _model())
            for content_hash, vector in zip(hashes, vectors):
                stats.misses += 1
                if use_cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    np.save(_cache_path(content_hash), vector)
                for chunk in by_hash[content_hash]:
                    results[chunk.chunk_id] = vector.tolist()

        box["out"] = len(results)
        box["hits"] = stats.hits
        box["misses"] = stats.misses
        box["cached"] = stats.hits

    return results


def _model() -> Any:
    from ..embedder_singleton import ModelUnavailableError, get_model

    try:
        return get_model()
    except ModelUnavailableError as exc:
        raise EmbeddingError(str(exc)) from exc


def embed_query(text: str) -> list[float]:
    """Embed one query with the SAME model and SAME normalization as Stage 3.

    Never instantiate a model here. A second instance with different
    normalization would make every score meaningless while still looking
    plausible.
    """
    if not text or not text.strip():
        raise EmbeddingError("cannot embed an empty query")
    return _encode([text], _model())[0].tolist()


def cache_size() -> tuple[int, int]:
    """``(file_count, total_bytes)`` for the embedding cache."""
    if not CACHE_DIR.exists():
        return (0, 0)
    files = list(CACHE_DIR.glob("*.npy"))
    return (len(files), sum(f.stat().st_size for f in files))


# --------------------------------------------------------------------------- #
# Handoff to the store stage
# --------------------------------------------------------------------------- #

# The vector handoff between --stage embed and --stage store. The .npy cache is
# the *embedding cache*; this file is the *stage handoff*, so `--stage store` can
# run without loading the model at all. Both are content-addressed, so a stale
# handoff file can never pair a vector with the wrong chunk.
VECTORS_PATH = SETTINGS.path("embedding.vectors_path", "artifacts/embeddings.json")


def save_embeddings(vectors: dict[str, list[float]], path: Path | None = None) -> Path:
    """Write ``{chunk_id: vector}`` to ``artifacts/embeddings.json``."""
    import json

    target = path or VECTORS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({k: [round(float(x), 8) for x in v] for k, v in vectors.items()}),
        encoding="utf-8",
    )
    return target


def load_embeddings(path: Path | None = None) -> dict[str, list[float]]:
    """Read the stage handoff. Missing file is an empty dict, not an error."""
    import json

    target = path or VECTORS_PATH
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): [float(x) for x in v] for k, v in raw.items()}


def clear_cache() -> int:
    """Delete every cached vector. Returns the number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for path in CACHE_DIR.glob("*.npy"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed




__all__ = [
    "embed_chunks", "embed_query", "cache_key", "cache_size", "clear_cache",
    "save_embeddings", "load_embeddings", "VECTORS_PATH",
    "CACHE_DIR", "BATCH_SIZE", "EXPECTED_DIM", "EmbedStats", "EmbeddingError",
]
