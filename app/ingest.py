"""Ingestion CLI — runs the offline RAG pipeline stage by stage.

Phases 1-2 implement ``load`` and ``chunk``. Phase 3 adds ``embed`` and ``store``.

    python -m app.ingest                 # everything that is implemented
    python -m app.ingest --stage load    # just fetch/extract/clean
    python -m app.ingest --stage chunk   # chunk the committed corpus
    python -m app.ingest --offline       # replay artifacts/raw_docs.jsonl, no network
    python -m app.ingest --experiment    # FR-05 chunk-strategy sweep

The ``--stage`` flag is what keeps a failure diagnosable: if chunking breaks, you
can still prove the corpus itself loaded (architecture.md §13).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import enable_utf8_console
from .config import SETTINGS
from .logging_utils import emit
from .models import Chunk
from .pipeline import chunker, loader
from .pipeline.embedder import (
    CACHE_DIR,
    cache_size,
    load_embeddings as read_embeddings,
    save_embeddings as write_embeddings,
)
from .pipeline.store import COLLECTION_NAME, DB_PATH, get_collection, upsert_chunks
from .sources import ALLOWLIST

STAGES = ("load", "chunk", "embed", "store")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest",
        description="Build the RAG index from the 5 allowlisted public pages.",
    )
    parser.add_argument(
        "--stage", choices=(*STAGES, "all"), default="all",
        help="Stop after this stage (default: all).",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Replay artifacts/raw_docs.jsonl with zero network calls.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop and recreate the vector collection (Phase 3).",
    )
    parser.add_argument(
        "--strategy", default=None,
        help="Override chunking.strategy (Phase 2).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Override chunking.chunk_size in characters (Phase 2).",
    )
    parser.add_argument(
        "--overlap", type=int, default=None,
        help="Override chunking.chunk_overlap in characters (Phase 2).",
    )
    parser.add_argument(
        "--experiment", action="store_true",
        help="Run the chunk-strategy comparison and write docs/chunking_experiment.md "
             "(Phase 2, FR-05).",
    )
    return parser.parse_args(argv)


def _requested_stages(stage: str) -> list[str]:
    return list(STAGES) if stage == "all" else [stage]


def main(argv: list[str] | None = None) -> int:
    enable_utf8_console()

    args = _parse_args(argv)
    stages = _requested_stages(args.stage)

    print(f"HDFC MF FAQ RAG — ingest ({', '.join(stages)})")
    print(f"Corpus: {len(ALLOWLIST)} allowlisted pages (app/sources.py)")
    if args.offline:
        print("Mode:   offline (replay from artifacts/raw_docs.jsonl)")
    print()

    # ---- STAGE 1: load -----------------------------------------------------
    if "load" in stages:
        try:
            docs = loader.load_all(offline=args.offline)
        except loader.AllowlistViolation as exc:
            print(f"\nALLOWLIST VIOLATION (C1/C2): {exc}", file=sys.stderr)
            return 1
        except loader.FetchError as exc:
            print(f"\nFETCH FAILED: {exc}", file=sys.stderr)
            print(
                "A partial corpus must never ship, so nothing was written. "
                "Re-run once the page is reachable, or use --offline to replay "
                "the committed corpus.",
                file=sys.stderr,
            )
            return 1
        except loader.ExtractionError as exc:
            print(f"\nEXTRACTION FAILED: {exc}", file=sys.stderr)
            print(
                "If this persists the page layout likely changed. trafilatura and "
                "a <main>/<article> fallback will not help on this app: the facts "
                "live in embedded JSON (__NEXT_DATA__ and schema.org FAQPage), so "
                "app/pipeline/loader.py is where a layout change must be handled.",
                file=sys.stderr,
            )
            return 1

        loader.write_jsonl(docs)
        loader.write_sources_csv(docs)

        print()
        print("Corpus summary")
        for doc in docs:
            print(f"  {loader.describe(doc)}")
        total_words = sum(d.word_count for d in docs)
        print(f"  TOTAL: {len(docs)} documents, {total_words} words")
        emit("load_summary", **{"in": len(ALLOWLIST)}, out=len(docs), word_count=total_words)
        print("\nStage 1 complete — corpus ready for chunking.")

    # ---- STAGE 2: chunk ----------------------------------------------------
    if "chunk" in stages:
        try:
            docs = chunker.read_raw_docs()
        except FileNotFoundError as exc:
            print(f"\nCHUNK FAILED: {exc}", file=sys.stderr)
            return 1

        overrides: dict[str, Any] = {}
        if args.strategy:
            overrides["strategy"] = args.strategy
        if args.chunk_size is not None:
            overrides["chunk_size"] = args.chunk_size
        if args.overlap is not None:
            overrides["chunk_overlap"] = args.overlap
        engine = chunker.get_chunker(**overrides)
        if hasattr(engine, "name"):
            print(f"\nSTAGE 2: chunk (strategy={engine.name})")

        total_chunks: list[Chunk] = []
        total_violations = 0
        max_wp = int(SETTINGS.get("chunking.max_wordpieces", 256))
        target_wp = int(SETTINGS.get("chunking.target_wordpieces", 200))
        for doc in docs:
            report = chunker.run(doc, engine)
            total_violations += report.violations
            total_chunks.extend(report.kept)
            print(
                f"  {doc.source_id:<18} {doc.word_count:>5} words -> "
                f"{len(report.kept):>3} chunks"
            )

        chunker.write_chunks_jsonl(total_chunks)
        stats = chunker.summarise(total_chunks, total_violations)
        print()
        print("Chunk summary")
        print(f"  out          : {stats['out']}")
        print(f"  mean_chars   : {stats['mean_chars']}")
        print(f"  p95_tokens   : {stats['p95_tokens']}")
        print(f"  violations   : {stats['violations']}")
        print(f"  max_tokens   : {stats['max_tokens']} (cap {max_wp})")
        emit("chunk_summary", **stats)
        print("\nStage 2 complete — chunks ready for embedding (Phase 3).")

    # ---- STAGE 3a: embed ----------------------------------------------------
    if "embed" in stages:
        try:
            chunks_to_embed = chunker.read_chunks_jsonl()
        except FileNotFoundError as exc:
            print(f"\nEMBED FAILED: {exc}", file=sys.stderr)
            return 1
        if not chunks_to_embed:
            print("\nEMBED FAILED: clean_chunks.jsonl is empty", file=sys.stderr)
            return 1

        print(f"\nSTAGE 3: embed ({len(chunks_to_embed)} chunks)")
        from .pipeline.embedder import EmbeddingError, embed_chunks

        try:
            vectors = embed_chunks(chunks_to_embed)
        except EmbeddingError as exc:
            print(f"\nEMBED FAILED: {exc}", file=sys.stderr)
            return 1

        dim = len(next(iter(vectors.values()))) if vectors else 0
        print(f"  embedded     : {len(vectors)} chunks x {dim} dims")
        print(f"  cache dir    : {CACHE_DIR}")
        files, size = cache_size()
        print(f"  cache files  : {files} ({size / 1024:.0f} KB)")
        # Persist so --stage store can run without re-embedding.
        emb_path = write_embeddings(vectors)
        print(f"  wrote        : {emb_path}")
        emit("embed_summary", **{"in": len(chunks_to_embed)}, out=len(vectors), dim=dim)
        print("\nStage 3a complete — vectors ready for the store.")

    # ---- STAGE 3b: store ----------------------------------------------------
    if "store" in stages:
        try:
            chunks_to_store = chunker.read_chunks_jsonl()
        except FileNotFoundError as exc:
            print(f"\nSTORE FAILED: {exc}", file=sys.stderr)
            return 1

        vectors = read_embeddings()
        if not vectors:
            print(
                "\nSTORE FAILED: no embeddings on disk — run "
                "`python -m app.ingest --stage embed` first.",
                file=sys.stderr,
            )
            return 1

        print(f"\nSTAGE 3b: store (collection={COLLECTION_NAME}"
              f"{', RESET' if args.reset else ''})")
        from .pipeline.store import StoreError

        try:
            collection = get_collection(reset=args.reset)
            written = upsert_chunks(collection, chunks_to_store, vectors)
            total = collection.count()
        except StoreError as exc:
            print(f"\nSTORE FAILED: {exc}", file=sys.stderr)
            return 1

        print(f"  upserted     : {written} rows")
        print(f"  count()      : {total}")
        print(f"  collection   : {COLLECTION_NAME} @ {DB_PATH}")
        emit("store_summary", **{"in": len(chunks_to_store)}, out=total, count=total)
        if written and total > written:
            print(
                f"  note: count {total} > upserted {written} — stale rows from a "
                f"previous corpus are present. Re-run with --reset."
            )
        print("\nStage 3 complete — index is queryable (Phase 4).")

    if args.experiment:
        from .experiment import main as experiment_main

        print()
        return experiment_main([])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
