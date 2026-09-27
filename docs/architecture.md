# Architecture — Mutual Fund FAQ RAG Chatbot (HDFC AMC)

**Document type:** Technical / Software Architecture
**Version:** 1.0
**Status:** Draft for review
**Date:** 2026-09-27
**Companion to:** [`PRD.md`](./PRD.md) — requirement IDs (`C*`, `FR-*`, `NFR-*`) are traced throughout
**Source brief:** [`problemstatement.txt`](./problemstatement.txt)

---

## 1. Purpose of This Document

The PRD defines *what* we build and *why*. This document defines *how* the system is decomposed, what
data moves between components, and which decisions are load-bearing. It is the reference an engineer
implements from and a reviewer reads to judge whether the demo is genuinely a RAG pipeline.

### 1.1 Design Principles

| # | Principle | Consequence in the design |
|---|-----------|---------------------------|
| P1 | **Every RAG stage is a separate, independently runnable, separately logged module** | Required by C10. No framework orchestration; a plain Python call chain in `app/pipeline.py` so each stage is visible |
| P2 | **Grounding is enforced after generation, not just requested in the prompt** | Prompts are advisory; validators in Stage 11 are authoritative |
| P3 | **The corpus is the security boundary** | 5 allowlisted URLs, hardcoded. A small corpus makes hallucination detectable by inspection |
| P4 | **Deterministic and reproducible** | `temperature=0`, content-hashed chunk IDs, idempotent upsert, committed artifacts |
| P5 | **Fail loudly at build time, abstain softly at query time** | A broken fetch fails `ingest`; an unanswerable question returns an abstention, never a guess |
| P6 | **The demo must be inspectable** | Every stage prints a one-line summary (NFR-09); `--debug` exposes retrieved chunks and scores |
| P7 | **Zero-friction run** | CPU-only, no GPU, no paid key required to demo the full pipeline (NFR-05) |

---

## 2. System Context

```
                 ┌───────────────────────────────────────────┐
   developer ───▶ │  app.ingest   (offline / build time)     │
   (one-off)     │  allowlist → fetch → clean → chunk →      │
                 │  embed → chroma upsert → artifacts       │
                 └───────────────┬───────────────────────────┘
                                 │  chroma_db/  artifacts/*.jsonl  data/sources.csv
                                 ▼
   end user  ───▶ ┌───────────────────────────────────────────┐
   (UI or CLI)    │  app.ui / app.chat                        │
                  │  guard → embed → retrieve → prompt →      │
                  │  generate → validate → render             │
                  └───────────────┬───────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
      ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
      │  LLM provider │   │  local model  │   │ 5 HDFC pages  │
      │  (optional)   │   │ MiniLM-L6-v2  │   │  (fetched     │
      │               │   │  ~90 MB CPU   │   │   once only)  │
      └───────────────┘   └───────────────┘   └───────────────┘
```

**Trust boundaries:** the only network egress at build time is the 5 allowlisted hosts. At query time the
only egress is the LLM provider (optional) — never a live web search (PRD Q6, closed).

---

## 3. Component & Module Layout

```
project/
├── app/
│   ├── __init__.py
│   ├── config.py            # loads config.yaml + .env, typed settings object
│   ├── sources.py           # ALLOWLIST: the 5 URLs + scheme metadata (C1/C2)  <-- single source of truth
│   ├── ingest.py            # CLI entry: full offline pipeline            (FR-01..FR-05)
│   ├── chat.py              # CLI entry: one-shot question                (FR-06)
│   ├── ui.py                # Streamlit app                              (FR-07)
│   ├── evaluate.py          # golden-set scorer                          (FR-15)
│   ├── disclaimers.py       # all user-facing legal strings              (FR-18, C9)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── loader.py        # STAGE 1  fetch + extract + clean
│   │   ├── chunker.py       # STAGE 2  recursive | heading | semantic
│   │   ├── embedder.py      # STAGE 3  MiniLM sentence-transformers
│   │   ├── store.py         # STAGE 4  ChromaDB persistent client
│   │   ├── retriever.py     # STAGE 5/7 embed query + vector search + filter
│   │   ├── prompt.py        # STAGE 9  system prompt + context assembly
│   │   ├── generator.py     # STAGE 10 LLM call (LLMClient interface)
│   │   ├── validators.py    # STAGE 11 citation / length / C4 / PII checks
│   │   └── orchestrator.py  # wires the query-time call chain
│   │
│   ├── guards/
│   │   ├── __init__.py
│   │   ├── pii.py           # C3  input PII screen (pre-log, pre-LLM)
│   │   ├── intent.py        # C8/C4  FACTUAL | ADVICE | PERFORMANCE | OUT_OF_CORPUS
│   │   └── injection.py     # imperative-phrasing filter on retrieved text
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py          # abstract LLMClient
│   │   ├── openai_compat.py # any OpenAI-compatible endpoint
│   │   ├── free_tier.py     # default no-key backend
│   │   └── echo.py          # retrieval-only mode: proves pipeline w/o LLM
│   │
│   ├── models.py            # dataclasses: RawDoc, Chunk, RetrievedChunk, Answer, ChatTurn
│   ├── logging_utils.py     # stage timers, structured one-line summaries
│   └── experiment.py        # chunk-strategy comparison harness        (FR-05)
│
├── config.yaml              # all tunables (§10)
├── requirements.txt         # pinned
├── .env.example             # LLM_API_KEY etc., never committed
│
├── artifacts/               # committed audit trail
│   ├── raw_docs.jsonl       # STAGE 1 output
│   └── clean_chunks.jsonl   # STAGE 2 output
│
├── chroma_db/               # persistent vector store (gitignored)
├── cache/embeddings/        # sha256-keyed embedding cache (gitignored)
│
├── data/
│   ├── sources.csv          # FR-17 deliverable
│   ├── sources.md           # FR-17 deliverable
│   └── sample_qa.md         # D5 deliverable
│
├── eval/
│   ├── golden.json          # 20 labelled factual questions
│   ├── opinion_probes.json  # 5 must-refuse
│   ├── pii_probes.json      # 6 must-block
│   └── ooc_probes.json      # must-abstain
│
├── logs/run.jsonl            # stage summaries only — NEVER user input (NFR-08)
├── notebooks/demo.ipynb      # D1 deliverable
├── docs/
│   ├── problemstatement.txt
│   ├── PRD.md
│   ├── architecture.md      # this document
│   └── chunking_experiment.md  # FR-05 output
└── README.md                # D4 deliverable
```

**Import rule (enforced by convention):** `pipeline/*` and `guards/*` may import `models.py` and
`config.py` only. They must not import each other's internals. This is what keeps each stage testable
in isolation and keeps C10 honest.

---

## 4. Offline Pipeline (Build Time)

Entry point: `python -m app.ingest` (FR-01 → FR-05).

```
app.sources.ALLOWLIST
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STAGE 1  loader.py                                                 │
│  fetch()   httpx GET, 10s timeout, retry x2, browser UA           │
│  allow()   url must be in ALLOWLIST else raise AllowlistViolation  │
│  extract() trafilatura.extract() -> main text, fallback BS4        │
│  clean()   strip nav/footer/cookie/CTA, normalize whitespace       │
│  emit      RawDoc -> artifacts/raw_docs.jsonl                      │
└───────────────────────────────────────────────────────────────────┘
        │  List[RawDoc]  (5 expected; len != 5 => hard fail)
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STAGE 2  chunker.py  (strategy from config.yaml)                   │
│  recursive      size 800 / overlap 100, separators [\n\n,\n,. ]    │
│  heading_bound  split on H2/H3, recurse to size cap                │
│  semantic       MiniLM similarity breakpoint                       │
│  annotate       heading_trail, chunk_index, char_len, content_hash│
│  emit           Chunk -> artifacts/clean_chunks.jsonl              │
└───────────────────────────────────────────────────────────────────┘
        │  List[Chunk]  (expected 80-250)
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STAGE 3  embedder.py                                               │
│  cache_key = sha256(MODEL_ID + "::" + chunk.content_hash)          │
│  model     = SentenceTransformer("all-MiniLM-L6-v2", device=cpu)   │
│  guard     assert wordpiece_len(chunk) <= 256  (else split)       │
│  batch=32, normalize_embeddings=True, 384-dim                      │
│  emit      Dict[cache_key, List[float]]                            │
└───────────────────────────────────────────────────────────────────┘
        │  embeddings
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ STAGE 4  store.py                                                  │
│  client    chromadb.PersistentClient(path="./chroma_db")            │
│  collection "hdfc_faq", metadata={"hnsw:space":"cosine"}          │
│  ids       sha256(source_id + ":" + str(chunk_index))[:16]         │
│  op        collection.upsert(...)   # idempotent (FR-04)           │
│  flags     --reset drops+recreates; default upserts               │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
  data/sources.csv   (FR-17: url, status, word_count, fetched_at)
```

### 4.1 Stage 1 — Loading

**Component:** `app/pipeline/loader.py`

| Aspect | Design |
|--------|--------|
| **Input** | `app.sources.ALLOWLIST` — a frozen tuple of 5 `SourceSpec` objects. Hardcoded, not env-driven, so the corpus cannot be widened at runtime (C1) |
| **Fetch** | `httpx.Client(timeout=10, follow_redirects=True)` + browser-like `User-Agent`. A redirect is validated against the allowlist again — a redirect off-allowlist is a violation, not a silent follow |
| **Extract** | `trafilatura.extract(html, include_comments=False, include_tables=True)`. Tables matter: fee tables are `<table>` markup and are exactly where expense ratio / exit load live. Falls back to `BeautifulSoup4` main-content extraction if trafilatura returns < 200 chars |
| **Clean** | Drop `<nav>`, `<footer>`, `<header>`, cookie/consent blocks, sticky elements, "app download" CTAs. Collapse 3+ blank lines. Collapse intra-line runs of 2+ spaces. Strip zero-width chars |
| **Output** | `RawDoc` per URL → `artifacts/raw_docs.jsonl` |
| **Failure** | Non-200, empty extraction, or an off-allowlist redirect raises and **aborts the run** (P5). Never ship a partial corpus — the bot would answer "not in sources" for facts that exist on the dead page |
| **Offline mode** | `ingest --offline` reads `raw_docs.jsonl` instead of the network (PRD Q2). Demo-day insurance |

**`RawDoc` shape** (see `app/models.py`):

```python
@dataclass(frozen=True)
class RawDoc:
    source_id: str        # "hdfc-large-cap"
    scheme_name: str
    category: str
    plan: str             # "Direct-Growth"
    url: str
    page_title: str
    text: str
    fetched_at: str       # ISO-8601 UTC
    http_status: int
    content_hash: str     # sha256 of cleaned text
```

### 4.2 Stage 2 — Chunking

**Component:** `app/pipeline/chunker.py` — a `Chunker` interface with three swappable implementations,
selected by `config.yaml: chunking.strategy`.

```python
class Chunker(Protocol):
    def split(self, doc: RawDoc) -> list[Chunk]: ...
```

| Strategy | Mechanism | Expected outcome |
|----------|-----------|------------------|
| `recursive` | `RecursiveCharacterTextSplitter(size=800, overlap=100, separators=["\n\n","\n",". "," "])` | Baseline; fast |
| `heading_bound` | Split on H2/H3, then recurse each section to the size cap | Expected winner |
| `semantic` | Sliding window + MiniLM cosine distance; break at the largest gap above a threshold | Slowest; may not pay off on short, heading-structured pages |

**Why the size cap is 800 characters and not 1,500 (the load-bearing decision):**
`all-MiniLM-L6-v2` has a hard input limit of **256 word-pieces** and truncates silently. A 1,500-char
chunk is roughly 350–400 word-pieces, so its tail — often the exit-load condition or the lock-in clause —
is discarded at embedding time while still present in the stored text. The chunk then *retrieves* on its
first half and the answer comes from the wrong passage. Therefore:

> **Invariant (asserted in a unit test, not just a comment): every chunk's tokenized length ≤ 256
> word-pieces, targeted at ≤ 200 (~800 chars) to leave headroom for title/preamble tokenization.**

This is a model property, not a tuning preference, and it is why the experiment in §4.2.1 should show the
larger chunk sizes underperforming rather than merely "being worse".

**Chunk enrichment — what makes retrieval precise:**

- `heading_trail` is prefixed to the embedded text as `"{page_title} > {h2} > {h3}\n\n{chunk_text}"`.
  The heading is high-signal context: a question about "exit load" should match a chunk that is
  *about* exit load even if the words do not appear in that sentence.
- `scheme_name` is likewise folded into the embedded prefix, so a scheme-specific question
  ("HDFC Small Cap expense ratio") gets a similarity boost for the right page without needing a metadata filter.

The stored `text` field keeps the **raw** chunk; only the **embedded** text carries the prefix. This
keeps citations clean for the user.

#### 4.2.1 The chunk-strategy experiment (FR-05)

`app/experiment.py` executes the protocol the PRD requires:

```
for strategy in [recursive, heading_bound, semantic]:
    for size in [400, 600, 800, 1000, 1200]:
        for overlap in [0, 0.10, 0.15]:
            chunks = strategy.split(docs, size, overlap)
            assert all(len(tok(c)) <= 256 for c in chunks)       # skip configs that violate the cap
            index = build_temp_index(chunks)                      # isolated temp Chroma collection
            recall@5 = mean(gold_chunk in top5 for 20 golden q)   # retrieval only, no LLM
    record(recall@5, n_chunks, mean_chunk_chars, p95_token_len, build_seconds)
```

Tie-break order: higher `Recall@5` → fewer chunks → shorter mean chunk length (tighter context, cheaper
prompts). Output: `docs/chunking_experiment.md`, with the winning config written back into
`config.yaml` and every alternative still selectable via `--strategy/--chunk-size/--overlap`.

### 4.3 Stage 3 — Embedding

**Component:** `app/pipeline/embedder.py`

| Aspect | Design |
|--------|--------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` — 384-dim, fixed by the brief |
| Device | `cpu` by default; `auto` opt-in via config |
| Call | `model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)` |
| Similarity | Cosine. Because vectors are L2-normalized, `dot(a,b) == cos(a,b)`, and Chroma `cosine` space reports `distance = 1 - cos_sim`. Hence **`score = 1.0 - distance`** — the single conversion used everywhere in the codebase |
| Cache | `cache/embeddings/{sha256(MODEL_ID + "::" + content_hash)}.npy`, keyed on **content**, not position, so an unchanged chunk is never re-embedded |
| Warm start | First run downloads ~90 MB from HuggingFace. README documents a one-line pre-warm; `HF_HUB_OFFLINE=1` for a fully cached demo |
| Failure | A model/tokenizer load error aborts with an actionable message rather than a stack trace |

The **same model instance** must serve Stage 3 and query embedding (Stage 7). Two instances with
different normalization settings silently produce a garbage index that "works" but retrieves nonsense —
so the model is a module-level singleton created once in `app/embedder_singleton.py`.

### 4.4 Stage 4 — Vector Store

**Component:** `app/pipeline/store.py`

```python
client      = chromadb.PersistentClient(path="./chroma_db")
collection  = client.get_or_create_collection(
    name="hdfc_faq",
    metadata={"hnsw:space": "cosine", "hnsw:M": 32, "hnsw:construction_ef": 200},
)
```

| Decision | Rationale |
|----------|-----------|
| **Single collection** for all 5 schemes | At ~150 chunks there is no search-quality benefit to partitioning; Chroma handles millions of vectors in one collection. A per-scheme collection is the *documented migration path* (§11), not a v1 cost |
| **Cosine space** | Vectors are normalized, so cosine ≡ inner product; consistent with `normalize_embeddings=True` |
| **Deterministic IDs** — `sha256(source_id + ":" + str(chunk_index))[:16]` | Re-running ingest **upserts** instead of duplicating (FR-04). Content change ⇒ new ID; deleted content ⇒ stale row, which is why `--reset` exists |
| **`where` filter available but optional** | Applied only when the intent gate extracts a scheme name — reduces cross-scheme noise at zero cost |
| **Primitive-only metadata** | Chroma metadata values must be `str/int/float/bool`. `heading_trail` is stored as a `" > "`-delimited string |

**Idempotency contract (FR-04):** `ingest` run twice with unchanged pages ⇒ identical collection count.
`ingest --reset` ⇒ count rebuilt from scratch. This is asserted in a test, since a silent duplicate
inflates the corpus and quietly degrades retrieval.

---

## 5. Query-Time Pipeline

Entry points: `app.chat` (FR-06, CLI) and `app.ui` (FR-07, Streamlit). Both call the **same**
`QueryPipeline.answer(question, debug=False)`, so behaviour cannot drift between the demo interface and
the terminal.

```
User input (CLI arg or UI textbox)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 6a  guards/pii.py            C3                          │
│  scan(PAN, Aadhaar, account, OTP, email, phone)                │
│  HIT  -> return pii_notice() ; DO NOT log, DO NOT call LLM     │
└──────────────────────────────────────────────────────────────────┘
        │ clean
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 6b  guards/intent.py         C8 / C4                     │
│  keyword pre-filter (free) -> LLM classifier (only if needed)  │
│  -> FACTUAL | ADVICE | PERFORMANCE | OUT_OF_CORPUS              │
│  ADVICE       -> return refusal() + SEBI link                  │
│  PERFORMANCE  -> return perf_redirect() + factsheet link        │
└──────────────────────────────────────────────────────────────────┘
        │ FACTUAL
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 7  retriever.embed_query                                    │
│  same MiniLM singleton, normalize_embeddings=True              │
└──────────────────────────────────────────────────────────────────┘
        │ vector(384)
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 8  retriever.search          C5 groundwork                │
│  n_results = 12 (fetch wide)                                    │
│  score = 1 - distance                                           │
│  drop score < 0.35                                              │
│  optional cross-encoder rerank -> keep 6   (stretch, Q5)        │
│  dedupe near-identical chunks                                   │
│  trim to ~2000 token context budget                             │
│  injection guard: drop chunks with imperative phrasing          │
│  ZERO survivors -> return abstain() + closest scheme URL        │
└──────────────────────────────────────────────────────────────────┘
        │ List[RetrievedChunk]
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 9   prompt.build            C4, C5, C6, injection        │
│  system prompt + <<<SOURCE url=...>>> blocks + question         │
│  asks for strict JSON: {answer, sources[], refused, reason}     │
└──────────────────────────────────────────────────────────────────┘
        │ messages
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 10  generator.generate       temperature=0, max_tokens~250 │
│  LLMClient interface -> provider | free_tier | echo            │
└──────────────────────────────────────────────────────────────────┘
        │ JSON
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 11  validators.apply         C4, C5, C6, C7  (AUTHORITATIVE)│
│  1 parse-safe JSON extraction                                    │
│  2 PII echo scan on output        -> redact, drop                │
│  3 C4 performance scan           -> perf_redirect (overrides all)│
│  4 citation present?             -> 1 retry, else chunk URL only │
│  5 cited URL in retrieved set?   -> drop hallucinated citation   │
│  6 sentence count <= 3           -> 1 retry, else truncate to 3  │
│  7 append "Last updated from sources: <max fetched_at>"          │
└──────────────────────────────────────────────────────────────────┘
        │ Answer (validated)
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 12  render  -> UI (FR-07..FR-10) or CLI (FR-06)            │
└──────────────────────────────────────────────────────────────────┘
```

**Key architectural point (P2):** Stage 11 is the authority. Stages 9 and 10 *request* compliance;
Stage 11 *enforces* it. The C4 performance guard in particular runs after generation and can replace the
answer entirely — because a prompt instruction alone is not a control, and C4 is a graded constraint.

### 5.1 Stage 6a — PII guard (C3, FR-11)

**Component:** `app/guards/pii.py`. Runs **before any logging and before any LLM call** — the ordering is
the entire point, since the moment text reaches a log file it is stored.

```python
PATTERNS = [
    ("email",     r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ("pan",       r"\b[A-Z]{5}\s?[0-9]{4}\s?[A-Z]\b"),
    ("aadhaar",   r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b"),
    ("account",   r"\b[0-9]{10,}\b"),
    ("phone",     r"(?:\+?91[\s\-.]?)?\b[6-9][0-9]{9}\b"),
    ("otp",       r"(?i)\b(?:otp|one[\s\-]?time|verification|passcode|pin)\b\D{0,20}\b[0-9]{4,6}\b"),
    ("ifsc",      r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),   # defence in depth, not in the brief
]
```

**Ordering matters:** `aadhaar` before `account` (a 12-digit Aadhaar partially matches the 10+ digit
account pattern), and `pan` before `account`. `phone` is constrained to a leading 6–9 to avoid
colliding with year-like and amount-like numbers.

**On hit:**
- Do **not** call the LLM, do **not** write the text anywhere, do **not** echo the matched value back
  (an echo is a second leak).
- Return the fixed notice in `app/disclaimers.py`, naming the *categories* blocked.
- Increment a **count-only** metric so the demo can show the guard firing (`pii_blocks_total`). Counts,
  never values.

**Known false-positive risk:** a legitimate question could contain a 10-digit number (a folio-like value,
a large amount). Accepted for v1 — a false block is safe; a false pass is a breach. Documented in
README known limits.

### 5.2 Stage 6b — Intent gate (C8, C4, FR-12)

Two-tier, so the common case costs nothing:

**Tier 1 — deterministic keyword pre-filter.** Zero-latency, catches the obvious cases:
`should i, should you, which is better, recommend, suggest, best fund, better than, buy, sell, hold,
switch, allocate, portfolio, is now a good time, worth investing, lump sum or sip, suitable for me`.

**Tier 2 — LLM classifier** (`classify_intent`), invoked only when tier 1 is inconclusive. Returns a
single token from `{FACTUAL, ADVICE, PERFORMANCE, OUT_OF_CORPUS}` with `temperature=0`.

| Class | Trigger examples | Response |
|-------|------------------|----------|
| `FACTUAL` | "Expense ratio of HDFC Large Cap?" · "Minimum SIP for small cap?" · "How do I download a capital gains statement?" | Full RAG pipeline |
| `ADVICE` | "Should I buy HDFC Small Cap?" · "How should I split 10L across these 5?" | Refusal template + SEBI link (PRD §11.3) |
| `PERFORMANCE` | "Best performing HDFC scheme?" · "What did Large Cap return in 2019?" | No numbers; redirect to factsheet (PRD §11.5) |
| `OUT_OF_CORPUS` | "Expense ratio of Parag Parag Flexi Cap?" · "Today's NAV?" | Abstention (PRD §11.4) |

**A deliberate boundary case (documented in PRD §20):** *"Expense ratio of Large Cap vs Flexi Cap?"*
is `FACTUAL`, not `ADVICE` or `PERFORMANCE` — it retrieves two expense-ratio facts and cites both. It
makes no recommendation and asserts no return. The gate is scoped to *advice and performance*, not to
*all comparison*.

### 5.3 Stage 8 — Retrieval

**Component:** `app/pipeline/retriever.py`

```python
res = collection.query(
    query_embeddings=[qvec],
    n_results=12,                                 # fetch wide, then filter
    include=["documents", "metadatas", "distances"],
)
scored = [(1.0 - d, m, doc) for d, m, doc in zip(res["distances"][0],
                                                  res["metadatas"][0],
                                                  res["documents"][0])]
kept = [c for c in scored if c[0] >= SCORE_FLOOR]  # 0.35
```

| Decision | Value | Rationale |
|----------|-------|-----------|
| Fetch width | `n_results=12` | Filtering and dedupe happen *after* the vector search. Fetching only 6 and then dropping low scorers can leave 3 chunks; fetching wide and trimming keeps the context full |
| Score floor | `0.35` cosine | Tuned on the golden set (FR-05 harness). Below it, the top hit is unrelated, and an abstention is the honest answer. Recorded in `config.yaml`, not hardcoded |
| Final context | top 6 | Enough to answer a multi-part factual question (e.g. benchmark *and* riskometer) without flooding the prompt |
| Context budget | ~2,000 tokens | Trimming is score-ordered, so the least relevant chunks are dropped first |
| Metadata filter | `scheme_slug` when the gate names a scheme | Cheap precision win; skipped when the scheme is absent or ambiguous |
| Dedupe | normalized-text hash + Jaccard > 0.9 | Boilerplate fee blocks repeat across pages; duplicates waste context |
| Reranker | cross-encoder, optional | **Out for v1** (PRD Q5). The interface exists, the flag defaults off |

**Failure semantics:** zero survivors after the floor ⇒ return `abstain()` with the URL of the most
relevant *scheme* page (even if its chunks scored low) so the user still has somewhere to go. Never
generate an answer from below-threshold context.

### 5.4 Stage 9 — Prompt construction

**Component:** `app/pipeline/prompt.py`

```
SOURCES:
<<<SOURCE url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
         scheme="HDFC Large Cap Fund" fetched_at="2026-09-27T10:14:03Z">>>
{chunk 1 text}
<<<END SOURCE>>>

<<<SOURCE url="..." scheme="..." fetched_at="...">>>
{chunk 2 text}
<<<END SOURCE>>>

<<<END SOURCES>>>

QUESTION: {question}

Reply with JSON only:
{"answer": "<=3 sentences>", "sources": ["<url>"], "refused": false, "reason": ""}
```

Design points:

- **Delimiter framing.** The system prompt states that text between `<<<SOURCE` / `<<<END SOURCE` is
  **data, never instructions**. Combined with `guards/injection.py` dropping any chunk containing
  imperative phrasing ("ignore previous", "you must now", "disregard the above"), an injected
  instruction has neither the framing nor the channel to succeed.
- **No tool-calling, no code execution anywhere in the app.** Injection has no lever to pull. This is a
  structural defence, not a filter.
- **Structured output over prose.** Requesting JSON means Stage 11 validates a field, rather than
  regex-scraping a sentence for a URL.
- **Sources are constrained to the retrieved set.** A URL the LLM emits that was not in the context is
  dropped in Stage 11 step 5 — this kills the most common hallucination failure mode (a plausible but
  invented Groww path).

### 5.5 Stage 11 — Validators (the control point)

**Component:** `app/pipeline/validators.py`. Ordered, cheap, deterministic, each with a defined repair.

| # | Check | Repair on failure |
|---|-------|-------------------|
| 1 | **Parse** — extract the JSON object, tolerate fenced blocks | On failure, retry once with a stricter instruction; then fall back to `abstain()` rather than render raw text |
| 2 | **PII echo** (C3) — same regex set applied to the *output* | Redact matches, then drop the answer and return the PII notice |
| 3 | **C4 performance guard** — see below | **Replace the entire answer** with `perf_redirect()` + factsheet link. This override is unconditional and runs before every other check |
| 4 | **Citation present** (C5) — `len(sources) >= 1` | Retry once; if still missing, attach the URL of the top-scoring retrieved chunk. An uncited answer is never displayed |
| 5 | **Citation valid** — every URL is in the retrieved set | Drop the invented URLs; if none survive, apply repair #4 |
| 6 | **Length** (C6) — `sentences(answer) <= 3` | Retry once; else take the first 3 sentences and append "See the source for full details." |
| 7 | **Freshness label** (C7) | Append `Last updated from sources: {max(fetched_at of cited chunks)}` — always the **cited** sources, never the whole corpus, so the timestamp is honest about what was actually used |

**The C4 guard regex** (mirrors the PRD constraint):

```python
PERFORMANCE_RE = re.compile(
    r"(?ix)
      \b\d+(?:\.\d+)?\s?%                     # any percentage
    | \b(?:CAGR|XIRR|IRR|NAV|absolute\s+returns?|returns?|outperform\w*
        | beat(?:s)?\s+the | best[\s-]?perform\w*
        | nifty|sensex|top[\s-]?perform\w*)\b
    "
)
ADVICE_RE = re.compile(
    r"(?ix)\b(?:you\s+should|should\s+(?:buy|sell|invest|start|switch)
                | we\s+recommend|i\s+recommend|is\s+a\s+good\s+(?:buy|time)
                | worth\s+investing|ideal\s+for\s+you)\b"
)
```

A hit on either ⇒ `perf_redirect()` (C4) or `refusal()` (C8) respectively.

**Honest limitation:** these are blunt instruments. The word `returns` also matches "exit load returns
after 12 months" — a legitimate fee fact — and would trigger a false redirect. For a facts-only demo
this trade is correct (a wrongly blocked answer is embarrassing; a stated return is a violation), and
percentages are universally banned because a genuine fee table line like "1.5%" must not be confused
with a return figure. The trade-off is documented in README known limits rather than hidden, and the
refusal message names the reason so a user is never left guessing.

**Sentence counting that survives decimals and abbreviations:**

```python
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z\u20b9"\'(\u201c])')
```

A naive `text.split(".")` destroys `"1.5%"` and `"Rs.500"`. Requiring the next token to start with an
uppercase letter, currency symbol, or quote keeps decimals and abbreviations intact, so the C6 check
counts *actual* sentences.

### 5.6 The `Answer` contract

```python
@dataclass(frozen=True)
class Answer:
    text: str                    # <= 3 sentences, already validated
    sources: list[str]           # >= 1, all verified present in the retrieved set
    last_updated: str            # max fetched_at across cited chunks
    kind: Literal["factual", "refusal", "abstain", "perf_redirect", "pii_notice", "error"]
    debug: list["RetrievedChunk"] = field(default_factory=list)  # FR-14, only when debug=True
```

`kind` is what the UI and the eval script switch on. Rendering lives in exactly one place, which is why
C5/C6/C7 cannot be satisfied in the UI and forgotten in the CLI (FR-18).

---

## 6. Guardrails Summary (traceability)

| Constraint | Where enforced | Layer | Bypassable? |
|-----------|-----------------|-------|-------------|
| **C1** public sources only | `app/sources.py` allowlist, checked pre-fetch and post-redirect | Structural | No — no runtime corpus config |
| **C2** no third-party blogs | Same allowlist; domain denylist in fetcher | Structural | No |
| **C3** no PII | `guards/pii.py` pre-logging; re-checked on output in Stage 11 | Code + order | No — runs before persistence |
| **C4** no performance claims | System prompt (request) **+** Stage 11 regex (enforce) | Code | No — enforcement overrides the model |
| **C5** citation in every answer | Stage 11 steps 4–5 + renderer assertion | Code | No — uncited answers cannot render |
| **C6** ≤ 3 sentences | Stage 11 step 6 | Code | No — truncation is the floor |
| **C7** "Last updated from sources:" | Stage 11 step 7 | Code | No |
| **C8** refuse advice | `guards/intent.py` + prompt + Stage 11 `ADVICE_RE` | Code (two layers) | No — regex backstop |
| **C9** disclaimer in UI | `app/disclaimers.py` constants, rendered in `app/ui.py` | Code | Only by editing the app |
| **C10** all stages present | Separate modules in `app/pipeline/`, one-line log each | Structural | No |

---

## 7. Observability

### 7.1 Stage summary format (NFR-09)

One JSON object per stage to stdout and `logs/run.jsonl`:

```json
{"ts":"2026-09-27T10:14:22Z","stage":"chunk","strategy":"heading_bound","in":5,"out":147,
 "mean_chars":742,"p95_tokens":211,"violations":0,"ms":88}
{"ts":"2026-09-27T10:14:23Z","stage":"retrieve","q_hash":"9f2a…","in":1,"out":6,
 "top_score":0.712,"floor":0.35,"dropped":6,"reranked":false,"ms":41}
```

Allowed keys only. **`q_hash` is a salted hash of the question, never the question itself** (NFR-08):
enough to correlate a slow query, not enough to reconstruct a user's financial question.

### 7.2 The debug drawer (FR-14) — the highest-value demo asset

```
QUESTION  What is the exit load on the HDFC Small Cap Fund?
RETRIEVED 6 chunks  (floor 0.35, top 0.712)
  0.71  hdfc-small-cap  > "Exit load"            "…no exit load if held more than 12 months…"
  0.68  hdfc-small-cap  > "Expenses"             "…direct plan expense ratio 0.94%…"
  0.44  hdfc-elss        > "Lock-in period"       "…3 years from the date of allotment…"
  0.41  hdfc-large-cap   > "Exit load"           "…1% if redeemed within 12 months…"
  0.38  hdfc-balanced    > "Fees"                "…exit load 0.5% for < 12 months…"
  0.36  hdfc-equity      > "Expenses"            "…"
ANSWER   <3 sentences>
SOURCE   https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth
```

This is what makes the demo persuasive: the reviewer sees *which passages* produced the answer and how
strongly they matched, and can check the answer against the cited page. Showing the low-score cross-scheme
chunks (0.38, 0.36) is honest and instructive — it demonstrates the score floor doing its job.

### 7.3 Evaluation harness (FR-15)

`python -m app.evaluate` runs four suites and prints one table:

| Suite | File | Pass criterion |
|-------|------|----------------|
| Golden factual (20) | `eval/golden.json` | `must_include` keywords present **and** cited URL == expected |
| Opinion probes (5) | `eval/opinion_probes.json` | `kind == "refusal"` |
| PII probes (6) | `eval/pii_probes.json` | `kind == "pii_notice"` **and** value not in output |
| Out-of-corpus (5) | `eval/ooc_probes.json` | `kind in {"abstain", "perf_redirect"}` |

Plus a `retrieval_only` mode that skips the LLM entirely and reports **Recall@5** — the number that
isolates retrieval quality from generation quality, and the one the chunking experiment optimizes.

---

## 8. Configuration (`config.yaml`)

```yaml
corpus:
  collection_name: hdfc_faq
  # The 5 URLs live in app/sources.py, NOT here — the corpus is code (C1).

loading:
  timeout_s: 10
  retries: 2
  user_agent: "Mozilla/5.0 (compatible; mf-faq-rag/1.0; educational demo)"

chunking:
  strategy: heading_bound        # winning config from docs/chunking_experiment.md
  chunk_size: 800                # chars
  chunk_overlap: 100             # chars
  max_wordpieces: 256            # hard model cap — asserted, not assumed
  target_wordpieces: 200         # headroom budget
  prepend_heading_trail: true

embedding:
  model_id: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  batch_size: 32
  normalize: true
  cache_dir: cache/embeddings

retrieval:
  n_results: 12                  # fetch wide
  top_k: 6                       # after filtering
  score_floor: 0.35              # tuned on golden set
  dedupe_jaccard: 0.9
  max_context_tokens: 2000
  rerank:
    enabled: false               # stretch (PRD Q5)
    model_id: cross-encoder/ms-marco-MiniLM-L-6-v2
    top_n: 20

generation:
  temperature: 0.0
  max_tokens: 250
  max_sentences: 3               # C6
  provider: env                  # env | openai_compat | free_tier | echo
  model: null                    # provider default
  api_key_env: LLM_API_KEY

guards:
  pii_enabled: true              # C3
  intent_enabled: true           # C8
  injection_enabled: true
  performance_guard: true        # C4

ui:
  debug_default: false
  example_questions:
    - "What is the expense ratio of the HDFC Large Cap Fund?"
    - "Does the HDFC ELSS Tax Saver Fund have a lock-in period?"
    - "How do I download a capital gains statement?"
```

Env vars (`.env`, never committed): `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `HF_HUB_OFFLINE`,
`LOG_LEVEL`. **No secret is read from `config.yaml`**, so the config is safe to commit and show on screen
during the demo.

---

## 9. Key Design Decisions (ADR summary)

| # | Decision | Alternatives rejected | Rationale |
|---|----------|----------------------|-----------|
| A1 | **Plain Python, no LangChain/LlamaIndex** for orchestration | LangChain LCEL, LlamaIndex | C10 requires every stage to be visible. A framework hides exactly the thing being graded. A reviewer must be able to trace query → answer through ~6 files |
| A2 | **Deterministic validators after generation** (Stage 11) | Rely on prompt instructions alone | Prompts are advisory. A graded constraint needs a deterministic control. C4/C5/C6 are enforced in code |
| A3 | **800-char chunk cap driven by the 256-word-piece model limit** | 1,000–1,500 chars for "richer context" | Larger chunks truncate silently in the embedding while the text stays intact — the worst failure mode, because the chunk retrieves on its head and the answer comes from the wrong passage |
| A4 | **Heading trail prefixed into the embedded text only** | Prefix into stored text too | Improves retrieval match; keeps the user-facing citation text clean. Stored and embedded text being different is deliberate and documented |
| A5 | **Single Chroma collection** | One collection per scheme | At ~150 chunks, partitioning adds operational cost with no quality gain. Documented as the scaling path (§11) |
| A6 | **Fetch 12, filter, keep 6** | Fetch 6, use 6 | Filtering and dedupe after a narrow fetch can starve the context of relevant passages |
| A7 | **Structured JSON output** | Free-form prose + regex for URLs | Turns validation from scraping into field access; makes `kind` explicit for the UI and eval script |
| A8 | **Two-tier intent gate** (keywords, then LLM) | LLM-only | The common case (clearly factual) costs zero extra latency and zero extra tokens; the classifier is spent only on genuinely ambiguous queries |
| A9 | **`echo` LLM backend** | Require a key for every path | Lets the pipeline (retrieval, guards, rendering) be demoed and tested with no credentials at all — directly serves NFR-05 and demo-day risk mitigation |
| A10 | **Committed `raw_docs.jsonl` + `--offline` mode** | Live fetch only | Groww can block, throttle, or redesign. Demo day must not depend on a third party's uptime |
| A11 | **No web search at query time** | Retrieval-augmented web search | Would break the fixed 5-URL source-list deliverable and the public-sources constraint (PRD Q6) |
| A12 | **Commit `clean_chunks.jsonl`** | Gitignore artifacts | A reviewer can inspect exactly what the bot was told, and re-verify any answer without running anything |

---

## 10. Error Handling & Failure Modes

| Failure | Stage | Behaviour | Rationale |
|---------|-------|-----------|-----------|
| Fetch returns non-200 / empty extraction | 1 | **Hard fail**, abort ingest, print the URL and status | A partial corpus makes the bot confidently say "not in sources" about facts that exist |
| Redirect off the allowlist | 1 | **Hard fail** | C1 — the allowlist must not be laundered through a redirect |
| Chunk exceeds 256 word-pieces | 2 | Re-split, then **drop the oversized chunk with a logged violation** | Never embed a silently-truncated chunk |
| `n_chunks == 0` after chunking | 2 | **Hard fail** | The extractor found nothing usable; downstream stages would produce an empty bot |
| Model download fails | 3 | Abort with the offline-cache remedy in the message | Actionable over a stack trace |
| ChromaDB dir corrupt | 4 | Detect on open, instruct `ingest --reset` | Rebuildable from committed artifacts |
| Query embedding fails | 7 | Return a friendly error, log the stack server-side | The user is not a developer |
| Zero chunks above the score floor | 8 | `abstain()` + most relevant scheme URL | An honest abstention beats a guess (P5) |
| LLM returns malformed JSON | 10→11 | One strict retry, then `abstain()` | Never render unparsed text as an answer |
| LLM cites a URL not retrieved | 11 | Drop it; if none remain, attach the top chunk's URL | The most likely hallucination path, closed deterministically |
| C4 guard trips | 11 | Replace the answer wholesale with the redirect | C4 override is unconditional |
| No API key set | 10 | Auto-fall back to the `echo`/free-tier backend, and say so in the UI | Pipeline still demonstrable; no silent degradation (NFR-05) |
| Streamlit reruns on every keystroke | 12 | Debounce + `st.chat_input`; cache the model in `@st.cache_resource` | Without the singleton, a ~90 MB model reloads per interaction |

---

## 11. Extension Points (explicitly out of v1)

Each is a designed seam, not an accident — so scaling does not require a rewrite.

| Extension | Seam | Migration |
|-----------|------|-----------|
| **More AMCs / schemes** | `app/sources.py` allowlist + re-ingest | Beyond a few hundred schemes, switch to one collection per AMC with a router that picks the collection from the detected AMC name |
| **Live factsheet PDFs** | Loader already isolates extraction; add a `pypdf` branch | Factsheet data is tabular and more reliable than HTML scraping — actually the *better* source for fees |
| **Cross-encoder reranking** | `retrieval.rerank` config flag + existing interface | Enable the flag; cost is one extra model (~80 MB) and ~150 ms |
| **Better chunking at scale** | `chunker.py` `Chunker` protocol | Semantic or proposition-level chunking becomes worthwhile once documents exceed a few pages |
| **Multi-turn context** | `QueryPipeline.answer` takes a single question | Add a history-aware rephrase step *before* Stage 7, keeping guardrails in front of the LLM |
| **More languages** | `embedder.py` is the only model-bound module | A multilingual MiniLM or `paraphrase-multilingual-MiniLM-L12-v2` swap, plus translated corpus |
| **Feedback loop** | `Answer` + `eval/` harness | Thumbs-up/down into a review queue; wrong answers become new golden-set cases |
| **Hosting** | `app/ui.py` is a standard Streamlit app | Deploy to Streamlit Community Cloud; the 1.5 GB footprint (NFR-06) is the only real limit |

---

## 12. Testing Strategy

| Layer | Tool | What it guarantees |
|-------|------|--------------------|
| Unit — `pii.py` | `pytest` | Every `eval/pii_probes.json` pattern is caught; benign finance questions are not blocked |
| Unit — `validators.py` | `pytest` | Sentence counter handles `"1.5%"` and `"Rs.500"` correctly; C4 regex fires on planted violations; citation repair paths work |
| Unit — `chunker.py` | `pytest` | **Invariant: every chunk ≤ 256 word-pieces**; no empty chunks; order preserved; heading trail populated |
| Unit — `store.py` | `pytest` | Ingest twice ⇒ identical collection count (idempotency, FR-04) |
| Integration — `ingest` | `pytest`, offline fixture | 5 docs in → N chunks out → N embeddings → collection populated |
| Integration — query | `pytest` | Each probe file yields the expected `Answer.kind` with no network or LLM |
| Eval | `app.evaluate` | The four suites in §7.3 and the §15 PRD targets |
| Manual | Demo script | 3-minute walkthrough, ending on the debug drawer |

**Fixture discipline:** unit and integration tests run against committed `raw_docs.jsonl` with the
`echo` backend. No test touches the network or needs a key — so the suite is green on a fresh clone,
which is what a reviewer will do first.

---

## 13. Build Order → Milestones

Mirrors PRD §17. Each milestone ends with a *runnable artifact*, so there is always a demo.

| Milestone | Modules | Runs | Satisfies |
|-----------|---------|------|-----------|
| M0 | `config.py`, `sources.py`, `models.py`, `disclaimers.py` | `python -c "import app"` | — |
| M1 | `loader.py` | `python -m app.ingest --stage load` | FR-01 |
| M2 | `chunker.py`, `experiment.py` | `--stage chunk`, `ingest --experiment` | FR-02, FR-05 |
| M3 | `embedder.py`, `store.py` | `--stage embed,store` | FR-03, FR-04 |
| M4 | `retriever.py`, `prompt.py`, `generator.py`, `llm/*` | `python -m app.chat "…"` | FR-06 |
| M5 | `guards/*`, `validators.py` | probe suites | FR-11, FR-12, FR-13, FR-16 |
| M6 | `ui.py`, `orchestrator.py` | `python -m app.ui` | FR-07 – FR-10, FR-14 |
| M7 | `evaluate.py`, `data/`, `README.md` | `python -m app.evaluate` | FR-15, FR-17, FR-18, D1–D6 |

The `--stage` flag is worth the small extra surface area: it lets M1–M3 be verified independently, which
is what keeps a failure diagnosable to one stage instead of the whole pipeline.

---

## 14. Architecture Decision Log — What We Deliberately Did *Not* Do

For the reviewer's benefit, the plausible-but-rejected directions:

1. **A real agentic loop** (plan → retrieve → re-retrieve → verify → answer). Rejected: latency blows
   past NFR-03, and with a 5-page corpus a second retrieval pass adds nothing. The loop would be theatre.
2. **Fine-tuning or a domain model.** Rejected: 5 pages is not training data, and a fine-tuned model
   cannot cite a URL.
3. **Knowledge graph for scheme metadata.** Rejected: expense ratio, exit load, and minimum SIP are
   *prose* facts on these pages, not triples. A graph would need an extraction step that is strictly more
   fragile than the chunker we already need.
4. **A general web-crawl corpus.** Rejected: it violates C1/C2 and destroys the "every answer cites one
   of 5 known pages" property that makes the demo verifiable.
5. **Caching generated answers per question.** Rejected: it masks retrieval regressions during eval.
   The embedding cache is fine (content-keyed, deterministic); an answer cache is not.
6. **Storing chat history to disk for "context".** Rejected: it directly conflicts with C3/NFR-08 and
   buys nothing for a demo.

---

## 15. Requirement Traceability

| Req | Architecture element |
|-----|----------------------|
| C1, C2 | §3 `sources.py`, §4 Stage 1 allowlist check pre- and post-redirect, A11 |
| C3 | §5.1 `guards/pii.py`, ordering before logging, §5.5 step 2 output re-check, §12 unit tests |
| C4 | §5.4 prompt request + §5.5 step 3 deterministic override, §6 |
| C5 | §5.5 steps 4–5, `Answer.sources` non-empty by construction |
| C6 | §5.5 step 6 + decimal-safe sentence splitter, `max_sentences` config |
| C7 | §5.5 step 7, timestamp scoped to cited chunks |
| C8 | §5.2 two-tier intent gate + `ADVICE_RE` backstop, refusal template |
| C9 | `disclaimers.py` single source (FR-18), rendered in `ui.py` |
| C10 | §3 one module per stage, §7.1 per-stage logs, A1 |
| FR-01…05 | §4, §13 M1–M3, `experiment.py` §4.2.1 |
| FR-06, FR-07 | §5 shared `QueryPipeline`, `chat.py`, `ui.py` |
| FR-08, FR-09, FR-10 | §5.5 steps 4/6/7 + renderer |
| FR-11…13 | §5.1, §5.2, §5.3 zero-survivor branch |
| FR-14 | §7.2 debug drawer |
| FR-15 | §7.3 eval harness |
| FR-16 | §5.5 step 3 |
| FR-17 | §3 `data/sources.csv` emitted by ingest |
| FR-18 | `disclaimers.py` imported by `ui.py`, `chat.py`, README |
| NFR-01, NFR-02 | §4 stage timings, embedding cache |
| NFR-03, NFR-04 | §5.3 fetch width, §10 no-key fallback |
| NFR-05 | A9 `echo` backend, CPU default, §10 |
| NFR-06 | §11 hosting note (1.5 GB) |
| NFR-07 | `temperature=0`, content-hashed IDs, deterministic upsert |
| NFR-08 | §7.1 `q_hash` never the query, no history persistence |
| NFR-09 | §7.1 stage summary schema |
| NFR-10 | §3 module layout, §13 one command per milestone |

---

*End of architecture document — v1.0. Derived from `docs/PRD.md`.*
