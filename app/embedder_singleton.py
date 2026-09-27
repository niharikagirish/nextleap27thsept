"""Shared embedding-model access (architecture.md §4.3).

``all-MiniLM-L6-v2`` is instantiated exactly once per process and imported by
every caller. Two instances with different normalization settings produce an
index that appears to work while retrieving nonsense, so this module exists to
make a second instance impossible rather than merely discouraged.

Phase 2 needs the *tokenizer* (to enforce the 256 word-piece budget) and the
*model* (to run the chunking experiment). Phase 3 adds the content-keyed disk
cache in ``app/pipeline/embedder.py``; the caching and batching logic belongs
there, not here.
"""

from __future__ import annotations

import threading
from typing import Any

from .config import SETTINGS

_LOCK = threading.Lock()
_TOKENIZER: Any | None = None
_MODEL: Any | None = None


def get_model_id() -> str:
    return str(SETTINGS.get(
        "embedding.model_id", "sentence-transformers/all-MiniLM-L6-v2"
    ))


def get_device() -> str:
    return str(SETTINGS.get("embedding.device", "cpu"))


class ModelUnavailableError(RuntimeError):
    """The model or tokenizer could not be loaded, with a usable remedy."""


def get_tokenizer() -> Any:
    """Return the shared tokenizer for the configured model id.

    Deliberately *not* cached to disk by this module: the tokenizer ships inside
    the same model download, so ``get_model`` covers the network case. When only
    the tokenizer is needed (the Phase 2 token-budget check) this loads the
    lightweight tokenizer only, which is much faster than the full model.
    """
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    with _LOCK:
        if _TOKENIZER is None:
            try:
                from transformers import AutoTokenizer

                _TOKENIZER = AutoTokenizer.from_pretrained(get_model_id())
            except Exception as exc:  # noqa: BLE001 - re-raised with a remedy
                raise ModelUnavailableError(
                    f"could not load the tokenizer for {get_model_id()!r}: {exc}\n"
                    "Remedy: pre-download it once with\n"
                    "  python -c \"from transformers import AutoTokenizer; "
                    "AutoTokenizer.from_pretrained("
                    "'sentence-transformers/all-MiniLM-L6-v2')\"\n"
                    "Then re-run. If this machine is offline, set HF_HUB_OFFLINE=1 "
                    "in .env after the cache is warm."
                ) from exc
    return _TOKENIZER


def get_model() -> Any:
    """Return the shared SentenceTransformer, loading it at most once."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is None:
            try:
                from sentence_transformers import SentenceTransformer

                _MODEL = SentenceTransformer(get_model_id(), device=get_device())
            except Exception as exc:  # noqa: BLE001 - re-raised with a remedy
                raise ModelUnavailableError(
                    f"could not load {get_model_id()!r} on device "
                    f"{get_device()!r}: {exc}\n"
                    "Remedy: check that torch is installed CPU-only with\n"
                    "  pip install torch --index-url "
                    "https://download.pytorch.org/whl/cpu"
                ) from exc
    return _MODEL


def reset_for_tests() -> None:
    """Drop the cached singletons. Tests only."""
    global _TOKENIZER, _MODEL
    with _LOCK:
        _TOKENIZER = None
        _MODEL = None


__all__ = [
    "get_model", "get_tokenizer", "get_model_id", "get_device",
    "reset_for_tests", "ModelUnavailableError",
]
