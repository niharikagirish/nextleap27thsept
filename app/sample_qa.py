"""Regenerate `docs/sample_qa.md` from the live pipeline.

    python -m app.sample_qa

Every line in the sample file is produced by `answer_question()`; nothing is
hand-authored. That is the point of the artefact — a sample Q&A written by hand
proves nothing about whether the assistant behaves that way. Regenerate it after
any change to the loader, the guards or the generator, and commit the diff.
"""
from __future__ import annotations

import pathlib
import sys
import time

from app.disclaimers import SHORT
from app.pipeline.orchestrator import answer_question
from app.sources import ALLOWLIST

CASES: tuple[tuple[str, str], ...] = (
    ("What is the expense ratio of the HDFC Large Cap Fund?", "cited answer"),
    ("Does the HDFC ELSS Tax Saver Fund have a lock-in period?", "cited answer"),
    ("Who manages the HDFC Balanced Advantage Fund?", "cited answer"),
    ("What is the minimum SIP amount for the HDFC Small Cap Fund?", "cited answer"),
    ("What benchmark does the HDFC Balanced Advantage Fund track?", "cited answer"),
    ("What are the 1 year returns of the HDFC Large Cap Fund?", "refused: performance"),
    ("Should I invest in the HDFC Small Cap Fund?", "refused: advice"),
    ("What is the expense ratio of HDFC Parag Parag Fund?", "abstained: not in corpus"),
    ("What is the fund size of the HDFC Large Cap Fund?", "abstained: AUM excluded"),
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "sample_qa.md"

# A shared free-tier endpoint rate-limits bursts. The sample file is a
# deliverable, so a transient 429 must not be baked into it as the answer to a
# question the corpus can answer. Retry instead of recording the error.
_ATTEMPTS = 4
_BACKOFF_SECONDS = (0, 8, 20, 45)


def _ask(question: str):
    """Answer `question`, retrying a provider rate limit.

    Only `kind == "error"` is retried. An abstain or a refusal is a real answer
    and must be recorded as-is.
    """
    for attempt in range(_ATTEMPTS):
        if _BACKOFF_SECONDS[attempt]:
            time.sleep(_BACKOFF_SECONDS[attempt])
        answer = answer_question(question)
        if answer.kind != "error":
            return answer
        print(
            f"    rate limited, retry {attempt + 1}/{_ATTEMPTS}: {question}",
            file=sys.stderr,
            flush=True,
        )
    return answer


def _scheme_for(url: str) -> str:
    return next((s.scheme_name for s in ALLOWLIST if s.url == url), "scheme page")


def _render(index: int, question: str, intent: str, answer) -> str:
    out = [f"### Q{index}. {question}", ""]
    out.append(f"*Intended behaviour: {intent} - actual: `{answer.kind}`*")
    out.append("")
    out.append("> " + answer.text.strip().replace("\n", "\n> "))
    out.append("")
    if answer.sources:
        out.append("**Sources**")
        out.append("")
        for url in answer.sources:
            out.append(f"- [{_scheme_for(url)}]({url})")
        out.append("")
    else:
        out.append("**Sources:** none - no factual claim was generated.")
        out.append("")
    if answer.last_updated:
        out.append(f"Source fetched: {answer.last_updated}")
    return "\n".join(out)


_HEADER = """# Sample Q&A

Verbatim assistant output from the live pipeline. Regenerate with

```
python -m app.sample_qa
```

Answers are produced, not authored - do not hand-edit this file.

Disclaimer shown with every answer: **{short}**

Corpus: the five HDFC Direct-Growth scheme pages listed in `data/sources.csv`.

---
"""

_FOOTER = """---

## Why the last four behave this way

- **Returns and NAV** are dropped at ingestion (`loader.EXCLUDED_FIELDS`) and
  refused again at output. The assistant never states a return figure.
- **Buy/sell/hold** is non-factual and is declined before retrieval runs.
- **Parag Parag Fund** is not one of the five allowlisted schemes, so the
  assistant must not answer it from a similar-sounding fund in the corpus.
- **Fund size (AUM)** is present on the live page but deliberately excluded from
  the corpus, so the assistant abstains rather than quoting it. See "Known
  limits" in the README.
"""


def main() -> int:
    blocks: list[str] = []
    failed = False
    for i, (question, intent) in enumerate(CASES, start=1):
        print(f"[{i}/{len(CASES)}] {question}", file=sys.stderr, flush=True)
        answer = _ask(question)
        if answer.kind == "error":
            failed = True
            print(
                f"    !! still erroring after {_ATTEMPTS} attempts: {answer.text}",
                file=sys.stderr,
                flush=True,
            )
        blocks.append(_render(i, question, intent, answer))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        _HEADER.format(short=SHORT) + "\n\n" + "\n\n".join(blocks) + "\n\n" + _FOOTER,
        encoding="utf-8",
    )
    print(f"wrote {OUT}", file=sys.stderr)
    if failed:
        print(
            "at least one question still returned an error; do not commit this file",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())