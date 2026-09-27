"""The chunk-strategy experiment required by FR-05 (architecture.md §4.2.1).

Runs the documented grid::

    for strategy in [recursive, heading_bound, semantic]:
        for size     in [400, 600, 800, 1000, 1200]:
            for overlap in [0, 0.10, 0.15]:
                chunks = strategy.split(docs, size, overlap)
                assert all(len(tok(c)) <= 256 for c in chunks)
                recall@5 = mean(gold_chunk in top5 for 20 golden q)
            record(recall@5, n_chunks, mean_chunk_chars, p95_token_len, build_s)

and writes the winner into ``config.yaml`` plus a report in
``docs/chunking_experiment.md``.

Two deliberate departures from the pseudocode above, both to keep the
measurement honest:

* **No ChromaDB.** The index is an exact brute-force cosine in numpy. HNSW is
  approximate, so scoring chunkers through it would mix retrieval error into a
  chunking measurement, and it would drag Stage 4 into Stage 2. The corpus is
  ~150 chunks; exact search is instant and ground-truth.
* **The 256 cap is enforced, then verified.** Rather than asserting and
  crashing, ``enforce_token_budget`` re-splits offenders first and the harness
  records what it had to fix. A config that needed many rescues is a bad config
  and should lose on the tie-break, not abort the sweep.

Recall@5 uses only ``answer_span`` containment — no LLM, no model judging.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import enable_utf8_console
from .config import ROOT, SETTINGS
from .models import Chunk, RawDoc
from .pipeline import chunker as chunker_mod

SIZES: tuple[int, ...] = (400, 600, 800, 1000, 1200)
OVERLAPS: tuple[float, ...] = (0.0, 0.10, 0.15)
STRATEGIES: tuple[str, ...] = ("recursive", "heading_bound", "semantic")
TOP_K = 5


# --------------------------------------------------------------------------- #
# Gold set
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GoldQuestion:
    """One labelled question. Field names follow implementation.md 2.D."""

    id: str
    question: str
    fact_type: str
    expected_url: str
    expected_source_id: str
    answer_span: str
    must_include: tuple[str, ...]


def load_golden(path: Path | None = None) -> list[GoldQuestion]:
    target = path or ROOT / "eval" / "golden.json"
    if not target.exists():
        raise FileNotFoundError(f"gold set not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    out: list[GoldQuestion] = []
    for record in payload["questions"]:
        out.append(
            GoldQuestion(
                id=record["id"],
                question=record["question"],
                fact_type=record["fact_type"],
                expected_url=record["expected_url"],
                expected_source_id=record["expected_source_id"],
                answer_span=record["answer_span"],
                must_include=tuple(record.get("must_include") or ()),
            )
        )
    if not out:
        raise ValueError(f"{target} contains no questions")
    return out


# --------------------------------------------------------------------------- #
# Exact in-memory index
# --------------------------------------------------------------------------- #


class NumpyIndex:
    """Brute-force exact cosine search. Vectors are L2-normalized, so dot == cos."""

    def __init__(self, chunks: Sequence[Chunk], model: Any) -> None:
        self.chunks = list(chunks)
        self.vectors = np.asarray(
            model.encode(
                [c.embed_text for c in self.chunks],
                batch_size=int(SETTINGS.get("embedding.batch_size", 32)),
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )

    def rank(self, query_vector: np.ndarray, k: int = TOP_K) -> list[tuple[int, float]]:
        scores = self.vectors @ query_vector
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def search(self, query: str, model: Any, k: int = TOP_K) -> list[tuple[int, float]]:
        """Convenience path for single ad-hoc queries. The sweep uses ``rank``."""
        return self.rank(encode_queries([query], model)[0], k=k)


def encode_queries(texts: Sequence[str], model: Any) -> np.ndarray:
    """Embed query strings, L2-normalized so a dot product is a cosine."""
    return np.asarray(
        model.encode(
            list(texts),
            batch_size=int(SETTINGS.get("embedding.batch_size", 32)),
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype="float32",
    )


def recall_at_k(
    index: NumpyIndex,
    gold: Sequence[GoldQuestion],
    query_vectors: np.ndarray,
    k: int = TOP_K,
) -> tuple[float, list[dict[str, Any]]]:
    """Fraction of gold questions whose answer chunk appears in the top k.

    A question counts as a hit when *some* top-k chunk from the correct
    ``source_id`` contains the verbatim ``answer_span``. Restricting by
    ``source_id`` stops a chunk from the wrong scheme that happens to repeat the
    same boilerplate ("Minimum SIP amount ... is Rs 100") from scoring a hit.

    ``query_vectors`` are supplied pre-encoded because the 20 questions are
    identical across all 45 grid cells; re-encoding them per cell would add 900
    pointless model calls.
    """
    hits = 0
    detail: list[dict[str, Any]] = []
    for position, item in enumerate(gold):
        ranked = index.rank(query_vectors[position], k=k)
        hit = False
        best_rank = None
        for rank, (chunk_pos, _score) in enumerate(ranked, start=1):
            chunk = index.chunks[chunk_pos]
            if chunk.source_id == item.expected_source_id and item.answer_span in chunk.text:
                hit = True
                best_rank = rank
                break
        hits += int(hit)
        detail.append(
            {
                "id": item.id,
                "fact_type": item.fact_type,
                "hit": hit,
                "rank": best_rank,
            }
        )
    return hits / len(gold) if gold else 0.0, detail


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #


@dataclass
class Result:
    strategy: str
    size: int
    overlap: float
    overlap_chars: int
    n_chunks: int
    recall_at_5: float
    mean_chars: float
    p95_tokens: int
    max_tokens: int
    violations: int
    rescues: int
    build_seconds: float
    skipped: bool = False
    skip_reason: str = ""

    @property
    def passes_cap(self) -> bool:
        return self.max_tokens <= int(SETTINGS.get("chunking.max_wordpieces", 256))


def build_config(
    docs: Sequence[RawDoc], strategy: str, size: int, overlap: float
) -> tuple[list[Chunk], int, int, float]:
    """Chunk every doc under one grid cell and enforce the word-piece cap.

    ``overlap`` is a *fraction* of the size, per the §4.2.1 grid, converted here
    to the character overlap the splitters actually take.
    """
    overlap_chars = int(round(size * overlap))
    engine = chunker_mod.get_chunker(strategy, size, overlap_chars)
    max_wp = int(SETTINGS.get("chunking.max_wordpieces", 256))
    target_wp = int(SETTINGS.get("chunking.target_wordpieces", 200))

    start = time.perf_counter()
    raw: list[Chunk] = []
    for doc in docs:
        raw.extend(engine.split(doc))
    report = chunker_mod.enforce_token_budget(raw, max_wp, target_wp)
    elapsed = time.perf_counter() - start
    return report.kept, report.violations, report.rescued, elapsed


def sweep(
    docs: Sequence[RawDoc],
    gold: Sequence[GoldQuestion],
    model: Any,
    sizes: Sequence[int] = SIZES,
    overlaps: Sequence[float] = OVERLAPS,
    strategies: Sequence[str] = STRATEGIES,
) -> list[Result]:
    results: list[Result] = []
    total = len(strategies) * len(sizes) * len(overlaps)
    # The gold questions are the same for every cell; encode them exactly once.
    query_vectors = encode_queries([g.question for g in gold], model)
    step = 0
    for strategy in strategies:
        for size in sizes:
            for overlap in overlaps:
                step += 1
                label = f"[{step}/{total}] {strategy} size={size} overlap={overlap}"
                print(f"  {label} ...", flush=True)
                try:
                    chunks, violations, rescues, build_s = build_config(
                        docs, strategy, size, overlap
                    )
                except chunker_mod.TokenBudgetError as exc:
                    print(f"      skipped: {exc}", flush=True)
                    continue
                if not chunks:
                    print("      skipped: produced no chunks", flush=True)
                    results.append(
                        Result(
                            strategy=strategy, size=size, overlap=overlap,
                            overlap_chars=int(round(size * overlap)), n_chunks=0,
                            recall_at_5=0.0, mean_chars=0.0, p95_tokens=0,
                            max_tokens=0, violations=0, rescues=0,
                            build_seconds=0.0, skipped=True,
                            skip_reason="produced no chunks",
                        )
                    )
                    continue

                # implementation.md §2.C: "configs that violate the word-piece cap
                # are skipped and recorded as skipped". Enforcement re-splits what
                # it can, so a non-zero `violations` here means fragments survived
                # that no split could rescue — the config cannot be trusted to
                # represent its own text, so it is recorded but never scored.
                if violations > 0:
                    stats = chunker_mod.summarise(chunks, violations)
                    reason = (
                        f"{violations} fragment(s) exceeded the word-piece cap and "
                        f"could not be re-split"
                    )
                    print(f"      SKIPPED (cap): {reason}", flush=True)
                    results.append(
                        Result(
                            strategy=strategy, size=size, overlap=overlap,
                            overlap_chars=int(round(size * overlap)),
                            n_chunks=stats["out"], recall_at_5=0.0,
                            mean_chars=stats["mean_chars"],
                            p95_tokens=stats["p95_tokens"],
                            max_tokens=stats["max_tokens"],
                            violations=violations, rescues=rescues,
                            build_seconds=round(build_s, 3), skipped=True,
                            skip_reason=reason,
                        )
                    )
                    continue

                index = NumpyIndex(chunks, model)
                recall, _detail = recall_at_k(index, gold, query_vectors)
                stats = chunker_mod.summarise(chunks, violations)
                results.append(
                    Result(
                        strategy=strategy,
                        size=size,
                        overlap=overlap,
                        overlap_chars=int(round(size * overlap)),
                        n_chunks=stats["out"],
                        recall_at_5=round(recall, 4),
                        mean_chars=stats["mean_chars"],
                        p95_tokens=stats["p95_tokens"],
                        max_tokens=stats["max_tokens"],
                        violations=violations,
                        rescues=rescues,
                        build_seconds=round(build_s, 3),
                    )
                )
                print(
                    f"      recall@5={recall:.3f}  chunks={stats['out']}  "
                    f"mean_chars={stats['mean_chars']}  "
                    f"p95_tokens={stats['p95_tokens']}  "
                    f"violations={violations}",
                    flush=True,
                )
    return results


def pick_winner(results: Sequence[Result]) -> Result | None:
    """Tie-break exactly as documented: recall@5, then fewer chunks, then tighter.

    Skipped configs are excluded outright — a config that could not represent its
    own text within the model limit is not a candidate, whatever it scored before
    it was dropped.
    """
    scored = [r for r in results if not r.skipped]
    if not scored:
        return None
    return sorted(
        scored,
        key=lambda r: (
            0 if r.passes_cap else 1,
            -r.recall_at_5,
            r.n_chunks,
            r.mean_chars,
        ),
    )[0]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def render_report(
    results: Sequence[Result],
    winner: Result | None,
    gold: Sequence[GoldQuestion],
    docs: Sequence[RawDoc],
) -> str:
    max_wp = int(SETTINGS.get("chunking.max_wordpieces", 256))
    lines: list[str] = []
    add = lines.append

    add("# Chunking Strategy Experiment (FR-05)")
    add("")
    add("Generated by `python -m app.experiment`. Do not edit by hand — re-run the")
    add("harness to regenerate. The winning cell below is written back into")
    add("`config.yaml` automatically.")
    add("")
    add("## Method")
    add("")
    add(f"- Corpus: {len(docs)} documents "
        f"({sum(d.word_count for d in docs):,} words total)")
    add(f"- Gold set: {len(gold)} hand-labelled questions in `eval/golden.json`")
    add(f"- Grid: {len(STRATEGIES)} strategies x {len(SIZES)} sizes "
        f"{list(SIZES)} x {len(OVERLAPS)} overlaps {list(OVERLAPS)} "
        f"= {len(STRATEGIES) * len(SIZES) * len(OVERLAPS)} configs")
    add(f"- Metric: Recall@{TOP_K}, retrieval only. No LLM, no model-as-judge.")
    add("- Hit condition: a top-5 chunk from the correct `source_id` contains the")
    add("  verbatim `answer_span`.")
    add(f"- Cap: every chunk re-split until its MiniLM token length is <= {max_wp}")
    add("  word-pieces; `violations` counts fragments still unsplittable after that.")
    add("- Index: exact brute-force cosine in numpy — no ChromaDB, no HNSW")
    add("  approximation, so recall measures chunking rather than the index.")
    add("")

    add("## Winner")
    add("")
    if winner is None:
        add("No config completed. See the table below for errors.")
    else:
        add(f"**`{winner.strategy}` at `chunk_size={winner.size}`, "
            f"`chunk_overlap={winner.overlap_chars}` "
            f"(={winner.overlap:.0%} of size)**")
        add("")
        add("| metric | value |")
        add("|--------|-------|")
        add(f"| Recall@{TOP_K} | {winner.recall_at_5:.3f} |")
        add(f"| chunks | {winner.n_chunks} |")
        add(f"| mean chars/chunk | {winner.mean_chars} |")
        add(f"| p95 word-pieces | {winner.p95_tokens} |")
        add(f"| max word-pieces | {winner.max_tokens} (cap {max_wp}) |")
        add(f"| violations | {winner.violations} |")
        add(f"| build seconds | {winner.build_seconds} |")
    add("")

    add("## Full grid")
    add("")
    scored = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]
    add(f"Scored configs: {len(scored)}. Skipped: {len(skipped)}.")
    add("")
    add("| strategy | size | overlap | chunks | recall@5 | mean_chars | "
        "p95_tok | max_tok | viol | build_s |")
    add("|----------|-----:|-------:|-------:|---------:|----------:|"
        "-------:|--------:|-----:|-------:|")
    for r in sorted(
        scored, key=lambda x: (-x.recall_at_5, x.n_chunks, x.mean_chars)
    ):
        flag = "" if r.passes_cap else " **!**"
        add(
            f"| `{r.strategy}` | {r.size} | {r.overlap_chars} | {r.n_chunks} | "
            f"{r.recall_at_5:.3f} | {r.mean_chars} | {r.p95_tokens} | "
            f"{r.max_tokens}{flag} | {r.violations} | {r.build_seconds} |"
        )
    if skipped:
        add("")
        add("### Skipped configurations")
        add("")
        add("| strategy | size | overlap | reason |")
        add("|----------|-----:|-------:|--------|")
        for r in skipped:
            add(
                f"| `{r.strategy}` | {r.size} | {r.overlap_chars} | "
                f"{r.skip_reason} |"
            )
    add("")
    add("Rows marked **!** exceeded the word-piece cap even after re-splitting.")
    add("")

    add("## Per-strategy best")
    add("")
    add("| strategy | best recall@5 | at size/overlap | chunks | verdict |")
    add("|----------|--------------:|----------------|-------:|---------|")
    for strategy in STRATEGIES:
        subset = [r for r in scored if r.strategy == strategy]
        if not subset:
            reason = next(
                (r.skip_reason for r in skipped if r.strategy == strategy),
                "failed to run",
            )
            add(f"| `{strategy}` | — | — | — | skipped: {reason} |")
            continue
        best = sorted(
            subset,
            key=lambda r: (0 if r.passes_cap else 1, -r.recall_at_5, r.n_chunks),
        )[0]
        verdict = (
            "winner" if winner is not None and best is winner else "runner-up"
        )
        add(
            f"| `{strategy}` | {best.recall_at_5:.3f} | "
            f"{best.size}/{best.overlap_chars} | {best.n_chunks} | {verdict} |"
        )
    add("")

    add("## Reading these numbers")
    add("")
    add("- **Recall@5 may saturate near 1.0 for every strategy.** This corpus is")
    add("  small and the questions are single-fact lookups, so the gold chunk is")
    add("  usually the obvious neighbour. A flat grid is a real result, and it is")
    add("  *not* evidence that chunking does not matter — it is evidence that this")
    add("  corpus is too easy to discriminate between chunkers. The tie-break then")
    add("  does the real work, and \"any of these work, take the cheapest\" is a")
    add("  legitimate outcome.")
    add("- **The cap is the filter that actually matters.** A config whose `max_tok`")
    add(f"  exceeds {max_wp} would truncate silently at embedding time and answer from")
    add("  the wrong half of a passage. That failure mode is invisible in Recall@5")
    add("  and catastrophic in production, which is why it is asserted in a test and")
    add("  why cap-violating configs are recorded as skipped rather than scored.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# config.yaml write-back (comment-preserving, no new dependency)
# --------------------------------------------------------------------------- #

_CFG_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<key>strategy|chunk_size|chunk_overlap)\s*:\s*.*$"
)


def write_back_config(winner: Result, path: Path | None = None) -> bool:
    """Rewrite the three chunking scalars in place, leaving every comment intact.

    ruamel.yaml would do this properly but is not a dependency, and a full
    ``yaml.safe_load`` + ``safe_dump`` round-trip would strip the explanatory
    comments that make config.yaml readable during the demo. A line-targeted
    substitution inside the ``chunking:`` block is both safer and smaller.
    """
    target = path or ROOT / "config.yaml"
    if not target.exists():
        print(f"  {target} not found; skipping write-back", flush=True)
        return False

    text = target.read_text(encoding="utf-8")
    lines = text.split("\n")
    in_block = False
    replaced: set[str] = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^chunking\s*:\s*$", stripped):
            in_block = True
            continue
        if in_block and stripped and not line.startswith((" ", "\t")):
            break                                   # left the chunking block
        if not in_block:
            continue
        match = _CFG_LINE.match(line)
        if not match:
            continue
        key = match.group("key")
        if key == "strategy":
            new = f"{match.group('indent')}strategy: {winner.strategy}"
        elif key == "chunk_size":
            new = f"{match.group('indent')}chunk_size: {winner.size}"
        else:
            new = (
                f"{match.group('indent')}chunk_overlap: {winner.overlap_chars}  "
                f"# {winner.overlap:.0%} of chunk_size, chosen by app/experiment.py"
            )
        lines[index] = new
        replaced.add(key)

    missing = {"strategy", "chunk_size", "chunk_overlap"} - replaced
    if missing:
        print(
            f"  write-back skipped, keys not found in chunking block: {sorted(missing)}",
            flush=True,
        )
        return False

    target.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"  wrote winner to {target}: strategy={winner.strategy} "
        f"chunk_size={winner.size} chunk_overlap={winner.overlap_chars}",
        flush=True,
    )
    return True


def results_to_json(results: Sequence[Result], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    enable_utf8_console()
    parser = argparse.ArgumentParser(
        prog="python -m app.experiment",

        description="Run the FR-05 chunk-strategy sweep and pick the winner.",
    )
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES),
                        choices=list(STRATEGIES))
    parser.add_argument("--sizes", nargs="*", type=int, default=list(SIZES))
    parser.add_argument("--overlaps", nargs="*", type=float, default=list(OVERLAPS))
    parser.add_argument("--no-write-config", action="store_true",
                        help="report the winner but leave config.yaml alone")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs" / "chunking_experiment.md")
    args = parser.parse_args(argv)

    print("FR-05 chunking experiment", flush=True)
    try:
        docs = chunker_mod.read_raw_docs()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    gold = load_golden()
    print(
        f"  {len(docs)} documents, {sum(d.word_count for d in docs):,} words, "
        f"{len(gold)} gold questions",
        flush=True,
    )

    from .embedder_singleton import ModelUnavailableError, get_model

    try:
        model = get_model()
    except ModelUnavailableError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1

    print(
        f"  model={SETTINGS.get('embedding.model_id')} "
        f"device={SETTINGS.get('embedding.device')}",
        flush=True,
    )

    results = sweep(docs, gold, model, args.sizes, args.overlaps, args.strategies)
    if not results:
        print("ERROR: no configuration completed; nothing to write", flush=True)
        return 1

    winner = pick_winner(results)
    print("", flush=True)
    if winner is not None:
        print(
            f"WINNER: {winner.strategy} size={winner.size} "
            f"overlap={winner.overlap_chars} recall@5={winner.recall_at_5:.3f} "
            f"chunks={winner.n_chunks}",
            flush=True,
        )

    report = render_report(results, winner, gold, docs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"  wrote {args.out}", flush=True)
    results_to_json(results, ROOT / "artifacts" / "chunking_experiment.json")
    print(f"  wrote {ROOT / 'artifacts' / 'chunking_experiment.json'}", flush=True)

    if winner is not None and not args.no_write_config:
        write_back_config(winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
