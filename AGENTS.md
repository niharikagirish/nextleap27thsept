# AGENTS.md

Standing instructions for this repository. Read before changing anything here.

## What this is

A retrieval-augmented Q&A assistant over exactly five HDFC Mutual Fund pages on
Groww, with a Streamlit front end (`app/ui.py`), a CLI (`app/chat.py`), and a
Render deployment. The project is graded on traceability: every sentence must be
traceable to a source URL, and no claim may exist that the corpus cannot support.

## Hard constraints

Breaking any of these produces a wrong demo, not a slightly worse one.

- **The corpus is exactly the five URLs in `app/sources.py:ALLOWLIST`.** No new
  sources, no query-time web search, no general model knowledge.
- **Never state or compare returns.** Performance content is excluded at
  ingestion (`loader.EXCLUDED_FIELDS`) *and* guarded at output. Do not weaken
  either side.
- **Never commit `.env`, `GROQ_API_KEY`, or any credential.** `.env.example` is
  the only env file that is tracked. The key is optional: with none, the pipeline
  falls back to `app/llm/echo.py` and everything still works.
- **No hardcoded fund facts in code.** Facts come from the fetched pages. A
  literal fund figure in a source file is a bug, even in a test that "just needs
  a value" — use a clearly synthetic fixture instead.
- **`chroma_db/` and `cache/` are build outputs and stay gitignored.** Rebuild
  with `python -m app.ingest --stage all --offline --reset`. Never omit
  `--reset`: upserting into an existing HNSW index corrupts it, and the failure
  appears later as an opaque hnswlib error at query time.

## Verify before claiming done

A change is not finished until these have actually been run, in this order:

```bash
python -m pytest -q --no-header -p no:warnings     # must be fully green
python -m app.eval_retrieval --strict              # must exit 0
```

If the corpus changes, the eval rewrites `docs/retrieval_eval.md`. Commit that
file in the same change, and update any numbers quoted in `README.md`. CI fails
the build if the report drifts from the code.

## Verifying the UI

`tests/test_ui.py` drives the real Streamlit app through
`streamlit.testing.v1.AppTest` and must stay green. For a live check:

```bash
streamlit run app/ui.py
```

Prefer asserting against shared constants (e.g. `disclaimers.LAST_UPDATED_LABEL`)
over string literals, so a relabel cannot silently stop being tested.

## Git

- Branch is `main`. Remote is `git@github.com:niharikagirish/nextleap27thsept.git`.
- Before committing, confirm nothing ignored slipped in:
  `git status --porcelain --ignored`.
- `core.hooksPath` is `scripts/git-hooks`. The pre-push hook runs the test suite
  and the retrieval gate; it cannot be bypassed by accident. Only use
  `SKIP_PREPUSH=1` when a failure is already understood and the push is
  deliberate.
- Commit messages explain *why*, not *what*. The diff already says what changed.
- Push when a coherent unit of work is green. Do not push a red suite.

## Layout

| Path | Role |
|------|------|
| `app/sources.py` | the five-URL allowlist, example questions |
| `app/pipeline/loader.py` | fetch, extract, render facts into labelled prose |
| `app/pipeline/chunker.py` | sentence chunking with a word-piece budget |
| `app/pipeline/retriever.py` | dense search, score floor, dedup, truncation |
| `app/pipeline/orchestrator.py` | `answer_question()`, the one entry point both front ends use |
| `app/guards/` | intent routing, prompt injection, PII |
| `app/disclaimers.py` | user-facing strings, shared by UI and CLI |
| `artifacts/*.jsonl` | committed corpus, deliberately, so answers are auditable without running anything |
| `eval/golden.json` | retrieval gold set; `answer_span` must be verbatim corpus text |

## Known deviations from the spec

Present in the code, documented in the README, each a one-line revert. Do not
"fix" them back without the user asking:

1. Freshness label is `Source fetched:`, not `Last updated from sources:`
   (PRD.md C7) — the timestamp is `fetched_at`, not an AMC revision date.
2. The gold set has 23 questions, not FR-05's 20. The 20-question core is still
   asserted intact by `tests/test_chunker.py`.
3. `fund_manager` is extracted and evaluated although it is not one of the seven
   fact types the brief lists.
