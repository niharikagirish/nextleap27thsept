"""``python -m app.inspect`` — dump the pipeline's artifacts as readable .txt.

Built because the artifacts themselves are not reviewable in their stored form:
``clean_chunks.jsonl`` is one JSON object per line, and ``embeddings.json`` is a
map of chunk id to a 384-float list. Neither can be skimmed, diffed, or pasted
into a report. This writes a human-readable view of each, plus a straight dump
of what Chroma actually holds.

    python -m app.inspect                    # status of every artifact
    python -m app.inspect --all              # write all .txt dumps
    python -m app.inspect --chunks           # just the chunk dump
    python -m app.inspect --embeddings       # just the vector summary
    python -m app.inspect --chroma           # what the collection actually holds
    python -m app.inspect --chunks --chunk-id abc123def456

Every section reports "MISSING" and says which command produces it rather than
raising. A diagnostic tool that crashes on a fresh clone is useless on the one
occasion you need it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import enable_utf8_console
from .config import ROOT

RAW_DOCS = ROOT / "artifacts" / "raw_docs.jsonl"
CLEAN_CHUNKS = ROOT / "artifacts" / "clean_chunks.jsonl"
VECTORS = ROOT / "artifacts" / "embeddings.json"
OUT_DIR = ROOT / "artifacts" / "inspect"

RULE = "=" * 78
THIN = "-" * 78


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _write(text: str, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    target.write_text(text, encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# raw_docs.jsonl
# --------------------------------------------------------------------------- #


def dump_raw_docs(out_dir: Path) -> str:
    rows = _read_jsonl(RAW_DOCS)
    if not rows:
        return _missing(RAW_DOCS, "python -m app.ingest --stage load")

    parts = [
        RULE,
        f"RAW DOCS  {RAW_DOCS.relative_to(ROOT)}",
        f"{len(rows)} document(s)   generated from the 5 allowlisted Groww pages",
        f"dumped {_now()}",
        RULE,
        "",
    ]
    for doc in sorted(rows, key=lambda r: str(r.get("source_id", ""))):
        text = str(doc.get("text", ""))
        parts += [
            THIN,
            f"source_id    : {doc.get('source_id')}",
            f"scheme_name  : {doc.get('scheme_name')}",
            f"category     : {doc.get('category')}   plan: {doc.get('plan')}",
            f"url          : {doc.get('url')}",
            f"page_title   : {doc.get('page_title')}",
            f"http_status  : {doc.get('http_status')}    fetched_at: {doc.get('fetched_at')}",
            f"content_hash : {doc.get('content_hash')}",
            f"size         : {len(text)} chars, {len(text.split())} words",
            THIN,
            "",
            text if text else "(empty)",
            "",
        ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# clean_chunks.jsonl
# --------------------------------------------------------------------------- #


def dump_chunks(out_dir: Path, only_id: str | None = None) -> str:
    rows = _read_jsonl(CLEAN_CHUNKS)
    if not rows:
        return _missing(CLEAN_CHUNKS, "python -m app.ingest --stage chunk")

    if only_id:
        rows = [r for r in rows if r.get("chunk_id") == only_id]
        if not rows:
            return f"no chunk with chunk_id {only_id!r} in {CLEAN_CHUNKS.name}"

    try:
        from .config import SETTINGS

        cap = int(SETTINGS.get("chunking.max_wordpieces", 256) or 256)
        target = int(SETTINGS.get("chunking.target_wordpieces", 200) or 200)
        strategy = str(SETTINGS.get("chunking.strategy", "?"))
    except Exception:  # noqa: BLE001 - a dump must not depend on config loading
        cap, target, strategy = 256, 200, "?"

    by_source: dict[str, int] = {}
    for row in rows:
        by_source[str(row.get("source_id", "?"))] = by_source.get(
            str(row.get("source_id", "?")), 0) + 1

    parts = [
        RULE,
        f"CLEAN CHUNKS  {CLEAN_CHUNKS.relative_to(ROOT)}",
        f"strategy={strategy}  hard cap={cap} word-pieces  target={target}",
        f"{len(rows)} chunk(s) across {len(by_source)} document(s)",
        f"dumped {_now()}",
        RULE,
        "",
        "SUMMARY",
        THIN,
    ]
    for source_id, count in sorted(by_source.items()):
        parts.append(f"  {source_id:<24} {count:>4} chunk(s)")
    parts += ["", THIN, ""]

    over_cap = 0
    for row in sorted(rows, key=lambda r: (str(r.get("source_id", "")),
                                           int(r.get("chunk_index", 0) or 0))):
        index = int(row.get("chunk_index", 0) or 0)
        trail = [p for p in str(row.get("heading_trail", "")).split(" > ") if p] \
            if isinstance(row.get("heading_trail"), str) else list(row.get("heading_trail", []))
        text = str(row.get("text", ""))
        embed_text = str(row.get("embed_text", ""))
        marked = "  <-- OVER CAP" if len(embed_text.split()) * 1.35 > cap else ""
        if marked:
            over_cap += 1
        parts += [
            THIN,
            f"chunk_id     : {row.get('chunk_id')}",
            f"source_id    : {row.get('source_id')}    chunk_index: {index}",
            f"url          : {row.get('url')}",
            f"heading_trail: {' > '.join(trail) or '(none)'}",
            f"content_hash : {row.get('content_hash')}",
            f"size         : {row.get('char_len')} chars, {row.get('word_count')} words"
            f"{marked}",
            f"embed_text is the heading trail + text, and is what gets embedded.",
            THIN,
            "",
            "TEXT (what is stored, retrieved and shown to the user):",
            "",
            text,
            "",
            "EMBED_TEXT (what MiniLM actually encodes; text plus the trail):",
            "",
            embed_text if embed_text != text else "(identical to TEXT)",
            "",
        ]

    parts += [
        RULE,
        f"chunks over the {cap} word-piece cap (estimated): {over_cap}",
        "The authoritative count is tests/test_chunker.py::"
        "test_every_chunk_fits_the_wordpiece_cap, which tokenizes for real.",
        RULE,
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# embeddings.json
# --------------------------------------------------------------------------- #


def dump_embeddings(out_dir: Path, full: bool = False) -> str:
    if not VECTORS.exists():
        return _missing(VECTORS, "python -m app.ingest --stage embed")

    try:
        vectors: dict[str, list[float]] = json.loads(VECTORS.read_text(encoding="utf-8"))
    except ValueError as exc:
        return f"embeddings.json is not valid JSON: {exc}"

    if not vectors:
        return _missing(VECTORS, "python -m app.ingest --stage embed")

    chunks = {str(r.get("chunk_id")): r for r in _read_jsonl(CLEAN_CHUNKS)}
    dims = {len(v) for v in vectors.values()}

    parts = [
        RULE,
        f"EMBEDDINGS  {VECTORS.relative_to(ROOT)}",
        f"{len(vectors)} vector(s)   dimensions present: {sorted(dims)}",
        f"dumped {_now()}",
        RULE,
        "",
        "A 384-float vector is unreadable in full, so each row below shows the",
        "dimension, the L2 norm (must be ~1.0: vectors are normalized, which is",
        "what makes cosine similarity a plain dot product), and the first 6",
        "components. Use --full-vectors to print all 384 per chunk.",
        "",
        THIN,
    ]

    bad_norm = 0
    for chunk_id, vector in sorted(vectors.items()):
        norm = math.sqrt(sum(float(x) * float(x) for x in vector))
        if abs(norm - 1.0) > 0.01:
            bad_norm += 1
        row = chunks.get(chunk_id, {})
        parts.append(
            f"{chunk_id}  dim={len(vector):<4} norm={norm:.4f}  "
            f"src={row.get('source_id', '?'):<20} "
            f"head=[{', '.join(f'{float(x):+.4f}' for x in vector[:6])}"
            f"{', ...' if len(vector) > 6 else ''}]"
        )
        if full:
            for start in range(0, len(vector), 8):
                window = vector[start:start + 8]
                parts.append(
                    f"    [{start:>3}] " + " ".join(f"{float(x):+.6f}" for x in window)
                )
            parts.append("")

    parts += [
        "",
        THIN,
        f"vectors whose L2 norm is not ~1.0: {bad_norm}",
        "",
        "Disk cache (the .npy files that make a warm ingest fast):",
    ]
    try:
        from .pipeline.embedder import CACHE_DIR, cache_size

        count, total = cache_size()
        parts.append(f"  {CACHE_DIR.relative_to(ROOT)}  {count} file(s), "
                     f"{total / 1024:.1f} KiB")
        parts.append("  key = sha256(model_id + '::' + content_hash) -- content-addressed,")
        parts.append("  so an unchanged chunk is never re-embedded.")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"  (could not read cache stats: {exc})")

    parts += ["", RULE]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Chroma
# --------------------------------------------------------------------------- #


def dump_chroma(out_dir: Path, limit: int = 0) -> str:
    try:
        from .pipeline.store import COLLECTION_NAME, DB_PATH, get_collection
    except Exception as exc:  # noqa: BLE001
        return f"could not import the store module: {exc}"

    if not DB_PATH.exists():
        return _missing(
            DB_PATH,
            f"python -m app.ingest --stage store   (collection {COLLECTION_NAME!r})",
        )

    try:
        collection = get_collection()
        total = collection.count()
    except Exception as exc:  # noqa: BLE001
        return (f"ChromaDB directory exists at {DB_PATH.relative_to(ROOT)} but the\n"
                f"collection could not be opened: {exc}\n"
                f"Remedy: python -m app.ingest --stage store --reset")

    parts = [
        RULE,
        f"VECTOR STORE  {DB_PATH.relative_to(ROOT)}",
        f"collection   : {COLLECTION_NAME}",
        f"rows         : {total}",
        f"space        : cosine  (score = 1.0 - distance)",
        f"dumped {_now()}",
        RULE,
        "",
        "Yes, this is persistent. A PersistentClient writes to the directory",
        "above, so the index survives process exit and reboot. --stage store",
        "re-opens the same directory; nothing is rebuilt in memory.",
        "",
        THIN,
    ]

    if total == 0:
        parts.append("collection exists but is EMPTY -- no chunks have been upserted.")
        return "\n".join(parts)

    got = collection.get(include=["documents", "metadatas"], limit=limit or total)
    ids = got.get("ids") or []
    documents = got.get("documents") or []
    metadatas = got.get("metadatas") or []

    for position, row_id in enumerate(ids):
        meta = metadatas[position] if position < len(metadatas) else {}
        document = documents[position] if position < len(documents) else ""
        parts += [
            THIN,
            f"id           : {row_id}",
            f"source_id    : {meta.get('source_id')}    index: {meta.get('chunk_index')}",
            f"url          : {meta.get('url')}",
            f"heading_trail: {meta.get('heading_trail')}",
            f"fetched_at   : {meta.get('fetched_at')}",
            THIN,
            "",
            document,
            "",
        ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def _missing(path: Path, command: str) -> str:
    return (
        f"MISSING: {path.relative_to(ROOT)} does not exist.\n"
        f"  Produce it with:  {command}\n"
        f"  The pipeline has not been run on this machine yet."
    )


def status() -> int:
    print(RULE)
    print("ARTIFACT STATUS")
    print(f"root: {ROOT}")
    print(RULE)

    rows: list[tuple[str, Path, str]] = [
        ("stage 1  raw docs", RAW_DOCS, "python -m app.ingest --stage load"),
        ("stage 2  chunks", CLEAN_CHUNKS, "python -m app.ingest --stage chunk"),
        ("stage 3  vectors", VECTORS, "python -m app.ingest --stage embed"),
    ]
    for label, path, command in rows:
        if path.exists():
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            if path.suffix == ".jsonl":
                count = len(_read_jsonl(path))
            else:
                count = len(json.loads(path.read_text(encoding="utf-8")) or {})
            print(f"  PRESENT  {label:<18} {count:>5} record(s)  "
                  f"{size / 1024:>8.1f} KiB  {path.relative_to(ROOT)}")
        else:
            print(f"  MISSING  {label:<18} {'':>5}          {'':>8}     "
                  f"{path.relative_to(ROOT)}")
            print(f"           -> {command}")

    from .pipeline.store import COLLECTION_NAME, DB_PATH

    if DB_PATH.exists():
        try:
            from .pipeline.store import get_collection

            total = get_collection().count()
            print(f"  PRESENT  stage 3  store        {total:>5} row(s)      "
                  f"{'':>8}     {DB_PATH.relative_to(ROOT)} ({COLLECTION_NAME})")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR    stage 3  store        {'':>5}          {'':>8}     "
                  f"{DB_PATH.relative_to(ROOT)}: {exc}")
    else:
        print(f"  MISSING  stage 3  store        {'':>5}          {'':>8}     "
              f"{DB_PATH.relative_to(ROOT)} ({COLLECTION_NAME})")
        print("           -> python -m app.ingest --stage store")

    try:
        from .pipeline.embedder import CACHE_DIR, cache_size

        count, total = cache_size()
        state = f"{count:>5} file(s)   {total / 1024:>8.1f} KiB"
    except Exception as exc:  # noqa: BLE001
        state = f"unavailable: {exc}"
    print(f"  CACHE    embedding .npy    {state}      {CACHE_DIR.relative_to(ROOT)}")

    print()
    missing = [
        name for name, path in (
            ("raw docs", RAW_DOCS), ("chunks", CLEAN_CHUNKS),
            ("vectors", VECTORS), ("chroma", DB_PATH),
        ) if not path.exists()
    ]
    if missing:
        print(f"NOTHING RUN YET - missing: {', '.join(missing)}")
        print("Run, in order:")
        print("  python -m app.ingest --stage load")
        print("  python -m app.ingest --stage chunk")
        print("  python -m app.ingest --stage embed")
        print("  python -m app.ingest --stage store")
        print("then:  python -m app.inspect --all")
        return 1
    print("all stages present. To see them:  python -m app.inspect --all")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    enable_utf8_console()

    parser = argparse.ArgumentParser(
        prog="python -m app.inspect",
        description="Dump raw docs, chunks, embeddings and Chroma contents as .txt.",
    )
    parser.add_argument("--all", action="store_true", help="write every dump")
    parser.add_argument("--raw", action="store_true", help="dump raw_docs.jsonl")
    parser.add_argument("--chunks", action="store_true", help="dump clean_chunks.jsonl")
    parser.add_argument("--embeddings", action="store_true", help="dump embeddings.json")
    parser.add_argument("--chroma", action="store_true", help="dump the collection")
    parser.add_argument("--full-vectors", action="store_true",
                        help="print all 384 floats per chunk, not the first 6")
    parser.add_argument("--chunk-id", default=None, help="restrict the chunk dump")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the number of Chroma rows dumped (0 = all)")
    parser.add_argument("--out", default=str(OUT_DIR), help="output directory")
    args = parser.parse_args(argv)

    if not any((args.all, args.raw, args.chunks, args.embeddings, args.chroma)):
        return status()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    written: list[Path] = []

    jobs = [
        (args.raw or args.all, "raw_docs.txt", lambda: dump_raw_docs(out_dir)),
        (args.chunks or args.all, "chunks.txt",
         lambda: dump_chunks(out_dir, args.chunk_id)),
        (args.embeddings or args.all, "embeddings.txt",
         lambda: dump_embeddings(out_dir, args.full_vectors)),
        (args.chroma or args.all, "chroma_store.txt",
         lambda: dump_chroma(out_dir, args.limit)),
    ]
    for wanted, name, produce in jobs:
        if not wanted:
            continue
        text = produce()
        target = _write(text, name, out_dir)
        written.append(target)
        first = text.splitlines()[0] if text.splitlines() else ""
        print(f"wrote {target.relative_to(ROOT)}  ({len(text)} chars)  {first}")

    return 1 if all("MISSING" in t.read_text(encoding="utf-8")
                    for t in written) else 0


if __name__ == "__main__":
    raise SystemExit(main())
