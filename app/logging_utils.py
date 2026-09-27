"""One-line structured stage summaries (NFR-09).

SECURITY (C3 / NFR-08): user input is NEVER logged. A stage summary may contain
counts, durations, and a *salted, truncated hash* of a query — enough to
correlate a slow request, not enough to reconstruct what a user asked about a
mutual fund.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import ROOT

LOG_PATH = ROOT / "logs" / "run.jsonl"
_HASH_SALT = os.getenv("LOG_HASH_SALT", "demo-salt")

# Free-text log keys are rejected rather than silently dropped, so a stage can
# never accidentally emit a message that contains user input.
ALLOWED_LOG_KEYS: frozenset[str] = frozenset({
    "ms",                     # emitted by stage_timer for every stage
    # Stage 1-2: loading, chunking
    "strategy", "in", "out", "mean_chars", "p95_tokens", "violations",
    "mean_words", "max_tokens", "docs", "sections", "excluded", "word_count",
    # Stage 3: embedding, store
    "dim", "hits", "misses", "cached", "count", "reset", "collection",
    # Stage 4: retrieval
    "q_hash", "top_score", "floor", "dropped", "reranked", "n_retrieved",
    "n_kept", "n_deduped", "truncated", "filtered",
    # Stage 5-6: generation, guardrails
    "kind", "intent", "backend", "sentences", "cites", "pii_blocks_total",
    "pii_categories", "redacted", "repaired", "step", "abstained", "refused",
    "perf_redirected", "injection_dropped", "max_age_days", "fresh",
})


def utc_now() -> str:
    """ISO-8601 UTC, second precision, with a trailing Z."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def q_hash(question: str) -> str:
    """Salted, truncated one-way hash of a query. Safe to log; not reversible."""
    return hashlib.sha256(f"{_HASH_SALT}::{question}".encode("utf-8")).hexdigest()[:10]


def emit(stage: str, **fields: Any) -> None:
    """Write one JSON summary line to stdout and logs/run.jsonl."""
    record: dict[str, Any] = {"ts": utc_now(), "stage": stage}
    for key, value in fields.items():
        if key in ALLOWED_LOG_KEYS:
            record[key] = value
    line = json.dumps(record, separators=(",", ":"))
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # logging must never break the pipeline


@contextmanager
def stage_timer(stage: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a stage and emit its summary on exit.

    Usage::

        with stage_timer("load", **{"in": len(ALLOWLIST)}) as s:
            docs = load_all()
            s["out"] = len(docs)
    """
    box: dict[str, Any] = dict(fields)
    started = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = int((time.perf_counter() - started) * 1000)
        emit(stage, **box)
