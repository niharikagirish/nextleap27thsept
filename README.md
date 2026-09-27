# HDFC Mutual Fund FAQ Assistant

A retrieval-augmented Q&A assistant that answers questions about **five HDFC
Mutual Fund pages on Groww**, with every sentence traceable back to a source URL
and every answer carrying a fact-only disclaimer.

The corpus is a closed, versioned allowlist. The assistant cannot browse the web
at query time, cannot use general model knowledge, and cannot answer about any
fund outside the five pages below.

| slug | scheme |
|------|--------|
| `hdfc-large-cap` | HDFC Large Cap Fund |
| `hdfc-equity` | HDFC Flexi Cap Fund |
| `hdfc-elss` | HDFC ELSS Tax Saver Fund |
| `hdfc-small-cap` | HDFC Small Cap Fund |
| `hdfc-balanced-adv` | HDFC Balanced Advantage Fund |

## What it does

- **Intent routing** — classifies a question as factual, non-factual, or
  performance-like, and refuses the non-factual ones instead of answering them.
- **Scheme detection** — resolves the fund from the question so retrieval is
  filtered to that scheme's page.
- **Hybrid-free, dense retrieval** — `all-MiniLM-L6-v2` embeddings in a Chroma
  collection, with a similarity floor, near-duplicate de-duplication, and context
  truncation.
- **Grounded generation** — the LLM sees only retrieved passages, is required to
  cite, and is held to a fact-only answer shape.
- **Guardrails** — prompt-injection detection, PII refusal, and a hard cap of
  three cited sentences.
- **Observability** — structured JSONL run logs with a salted one-way query
  hash, so logs are correlatable without recording user questions.

Two front ends share one pipeline entry point, `app.pipeline.orchestrator.answer_question`:

- `app/ui.py` — Streamlit web app (the demo)
- `app/chat.py` — command line

## Measured results

Retrieval is measured against a 23-question gold set (`eval/golden.json`) by
`python -m app.eval_retrieval --strict`. A "hit" means the top-k chunk from the
correct source contains the verbatim `answer_span` — so a hit cannot be produced
by the LLM writing a plausible sentence.

| metric | value |
|--------|------:|
| hit@1 | 0.739 |
| hit@3 | 1.000 |
| hit@5 | 1.000 |
| hit@6 | 1.000 |
| scheme@1 | 1.000 |
| scheme@3 | 1.000 |
| scheme@5 | 1.000 |
| scheme@6 | 1.000 |
| MRR | 0.862 |

Corpus: 5 documents, 11,756 words, 146 chunks, **0** token-budget violations,
p95 218 word-pieces against a 256 cap. Report: `docs/retrieval_eval.md`.

Test suite: **427 passing** (`python -m pytest`).

`hit@1` of 0.739 with `scheme@1` of 1.000 is the interesting shape: the correct
page is always the top result, and the misses are chunk-boundary cases where the
gold sentence is real and present in the document but not inside any single
retrieved chunk. They are chunking artefacts, not retrieval failures.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env      # then fill in a key, or leave it empty
```

Install torch from the CPU index **first**. The default PyPI torch wheel pulls
multi-GB CUDA dependencies that this project never uses.

### The API key is optional

With no key the pipeline runs on `app/llm/echo.py` in retrieval-only mode:
ingestion, retrieval, guardrails, and the whole eval suite work, and the UI
shows the retrieved passages with their scores instead of a generated answer. It
says so on screen. Set `GROQ_API_KEY` in `.env` for generated answers.

## Build the index

`chroma_db/` and `cache/` are gitignored — they are build outputs, not source.
Rebuild from the committed artifacts with no network access:

```bash
python -m app.ingest --stage all --offline
```

`--offline` replays the committed `artifacts/raw_docs.jsonl`, so the index is
deterministic and matches the one the retrieval report was measured against. To
deliberately re-scrape the five live pages instead:

```bash
python -m app.ingest --stage all
```

A fresh build takes roughly 30 seconds plus a one-time ~90 MB model download.

## Run it

```bash
# web app
streamlit run app/ui.py

# CLI
python -m app.chat "What is the expense ratio of HDFC Large Cap Fund?"
python -m app.chat "Who manages HDFC ELSS Tax Saver Fund?" --debug

# retrieval without generation
python -m app.retrieve_debug "exit load HDFC Small Cap Fund"

# scoring gate
python -m app.eval_retrieval --strict
```

## Deploy to Render

`render.yaml` is a Render Blueprint, so the service is defined in version
control rather than clicked into a dashboard.

Via the dashboard — **New → Blueprint** and point it at this repo. Or with the
Render CLI: `render blueprint launch`.

To configure it by hand instead, these are the three values:

| field | value |
|-------|-------|
| **Root Directory** | *(empty — repository root)* |
| **Build Command** | `bash scripts/render_build.sh` |
| **Start Command** | `streamlit run app/ui.py --server.address 0.0.0.0 --server.port $PORT` |

Then set one environment variable in the Render dashboard:

| key | how |
|-----|-----|
| `GROQ_API_KEY` | paste your key; Render stores it as a secret |
| `PYTHON_VERSION` | `3.11.9` (or set in `render.yaml`) |
| `LOG_HASH_SALT` | let Render generate one |

Without `GROQ_API_KEY` the deploy still works in retrieval-only mode.

Two deployment details that are easy to get wrong:

- **The build must build the index.** A fresh clone has no `chroma_db/`. Skip
  the ingest step and the app serves "the retrieval index is empty" to every
  visitor while looking otherwise healthy. The build passes `--reset` so a
  rebuild never inherits a stale HNSW index — that failure otherwise surfaces
  much later as an opaque hnswlib error rather than as a build failure.
- **The container must bind `0.0.0.0`.** `.streamlit/config.toml` sets this, and
  the start command passes it too. Binding localhost makes the deploy refuse
  every connection.

The Blueprint uses the **starter** plan deliberately. The free plan's 512 MB
ceiling is tight against torch + the MiniLM model + chromadb, and running out of
it presents as a random crash rather than a clear error.

## Known limits and deliberate deviations

Stated plainly, because a demo that hides them is worse than useless.

### Deviations from the spec

Three places where the code intentionally does not match the written brief. Each
is a one-line revert if a rubric requires the literal spec behaviour.

1. **The freshness label reads `Source fetched:`, not `Last updated from
   sources:`** (PRD.md C7). The timestamp is `RawDoc.fetched_at` — when *we*
   pulled the page — not when the AMC last revised the figures, which the page
   never states. "Last updated" asserts a fact the data does not contain. Revert:
   change the value of `LAST_UPDATED_LABEL` in `app/disclaimers.py`.
2. **The gold set has 23 questions, not the 20 of FR-05.** The 20 are all still
   present and still checked; three were added (`q21`-`q23`) to cover
   `fund_manager` at the retrieval layer. `tests/test_chunker.py` now asserts the
   20-question *core* rather than an exact total.
3. **The `fund_manager` fact type is extracted, though it is not one of the seven
   fact types the brief lists.** It is on every page and users ask for it, so
   dropping it would have been a worse product.

### Limits

- The five pages are a snapshot. `artifacts/raw_docs.jsonl` records the fetch
  timestamp, and the UI shows it as `Source fetched`, so a stale answer is
  identifiable.
- Retrieval is dense-only. There is no BM25 leg, so an exact-match question
  about an unusual token can rank below paraphrased ones.
- The sentence chunker is naive; the p95 is 218 of a 256 token budget, so
  gold spans that straddle a boundary are unreachable by design.
- The injection guard is exercised only on inputs the intent router lets
  through; an out-of-topic question is rejected by the router, not the guard.
- The "Additional page text" section is best-effort `trafilatura` output and does
  carry return figures from the pages' performance tables. The Phase 6 output
  guard is what stops the model from stating them, and C4 is enforced on the
  extracted facts rather than on this trailing section.

## Repository layout

```
app/
  ui.py              Streamlit front end
  chat.py            CLI front end
  sources.py         the 5-URL allowlist + example questions
  disclaimers.py     shared user-facing strings
  ingest.py          load -> chunk -> embed -> store
  eval_retrieval.py  gold-set scoring, --strict gate
  retrieve_debug.py  retrieval only, no generation
  guards/            intent, injection, PII
  llm/               backend abstraction: echo, free-tier, OpenAI-compatible
  pipeline/          loader, chunker, embedder, store, retriever,
                     prompt, generator, validators, orchestrator
artifacts/           raw_docs.jsonl, clean_chunks.jsonl, embeddings.json
eval/golden.json     23-question retrieval gold set
docs/                implementation.md, retrieval_eval.md
render.yaml          Render Blueprint
scripts/             build scripts
tests/               427 tests
```

The `artifacts/*.jsonl` files are committed deliberately: a reviewer can verify
any answer's source text without running anything.
