"""STAGE 1 — Loading: fetch, extract, clean, persist (architecture.md §4.1).

FINDINGS THAT SHAPED THIS MODULE
--------------------------------
Probed 2026-09-27 against the live pages: all five are server-rendered Next.js
apps whose root container is ``id="__next"``. There is **no** ``<main>``,
``<article>``, ``<nav>`` or ``<footer>`` anywhere in the markup, and the fact
prose (expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark)
does **not** appear as visible body text. It lives in two embedded JSON blobs:

1. ``<script type="application/ld+json">`` — a schema.org ``FAQPage`` with ~8
   natural-language Q&A pairs per page.
2. ``window.__NEXT_DATA__`` — the app state, at
   ``props.pageProps.mfServerSideData``, a *flat* object carrying the fee and
   mandate fields as structured values.

Consequences, both of which contradict the naive plan in implementation.md:

* ``trafilatura`` alone extracts very little from a client-rendered shell, and
  the planned BeautifulSoup4 ``<main>``/``<article>`` fallback would match
  nothing. So JSON extraction is the *primary* path here, not the fallback.
* Rendering the structured fields into explicit, labelled prose is materially
  better for retrieval than dumping raw JSON: each fact becomes a self-contained
  sentence under a heading, which is what the ``heading_bound`` chunker and the
  cosine similarity search both want.

PERFORMANCE-CONTENT EXCLUSION
-----------------------------
The corpus genuinely contains return figures (``return_stats``, the JSON-LD
"what kind of returns" FAQ, NAV, AUM). Constraint C4 forbids us from *stating*
performance, so this loader drops those fields and FAQ entries at ingestion time
rather than relying on the Phase 6 output guard to catch them. The output guard
stays in place regardless, because the model can volunteer a return figure from
parametric memory even when it is absent from the context.

This is defence in depth, and it is a deliberate, auditable decision: the
exclusion list is a named module constant, not a scattered set of ``if``s.
"""

from __future__ import annotations

import csv
import hashlib
import html as html_lib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import httpx
from bs4 import BeautifulSoup

from ..config import SETTINGS
from ..logging_utils import stage_timer, utc_now
from ..models import RawDoc, SourceSpec
from ..sources import (
    ALLOWLIST,
    EXPECTED_SOURCE_COUNT,
    is_allowed_url,
    is_denied_host,
)

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class AllowlistViolation(RuntimeError):
    """A URL is not on the allowlist, or a redirect left it (C1, C2)."""


class FetchError(RuntimeError):
    """Non-200 response after the configured retries."""


class ExtractionError(RuntimeError):
    """The page yielded too little text to be a usable source."""


# --------------------------------------------------------------------------- #
# Corpus minimisation (C4)
# --------------------------------------------------------------------------- #

#: Structured fields deliberately NOT rendered into the corpus. All of these
#: describe performance, which constraint C4 forbids us from stating.
#:
#: ``_render_facts`` is a positive allowlist — it only ever touches the fields it
#: names — so this set is defence in depth and documentation rather than the
#: mechanism. ``tests/test_loader.py`` asserts that no field in this set has its
#: value appear in the rendered corpus text, so the two cannot drift apart.
EXCLUDED_FIELDS: frozenset[str] = frozenset({
    "return_stats", "sip_return", "simple_return", "nav", "nav_date",
    "aum", "fund_news", "fund_events", "holdings", "analysis",
    "peerComparison", "historic_expense_ratio_note",
})

#: FAQ entries whose *question* matches this are dropped, for the same reason.
#: The page's own wording varies, so this matches loosely and case-insensitively.
#:
#: Two patterns here are deliberately narrow, because every question on these pages
#: ends with the scheme name and therefore contains "Growth":
#:   * "how much has ... grow" must require BOTH phrases. A looser
#:     ``how (?:much|has).*grow`` also matched "How much expense ratio is charged
#:     by HDFC Small Cap Fund Direct Growth?", silently dropping the single most
#:     important FAQ on the page.
#:   * PE/PB is a static valuation ratio, not a performance figure, so it is kept.
EXCLUDED_FAQ_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\breturns?\b",
        r"\bnav\b",
        r"\baum\b",
        r"\bperform\w*",
        r"\bhow much has\b[^?]*\bgrow",
        r"\bmoney multiplier\b",
        r"\branking\b",
    )
)

#: The seven fact types the brief requires, mapped to the fields that carry them.
#: Kept as data so the README, the eval harness and this module agree on what
#: "covered" means.
REQUIRED_FACT_FIELDS: dict[str, tuple[str, ...]] = {
    "expense_ratio": ("expense_ratio", "base_expense_ratio"),
    "exit_load": ("exit_load",),
    "minimum_sip": ("min_sip_investment", "min_investment_amount"),
    "lock_in": ("lock_in",),
    "riskometer": ("nfo_risk",),
    "benchmark": ("benchmark", "benchmark_name"),
    "documents": ("sid_url", "brochure_link", "scheme_info_link"),
}

_MIN_EXTRACTED_CHARS = int(SETTINGS.get("loading.min_extracted_chars", 200))
_EXCLUDE_PERFORMANCE = bool(
    SETTINGS.get("loading.exclude_performance_content", True)
)

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_INLINE_SPACE = re.compile(r"[\t\u00a0]")
_INLINE_HTML = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------- #
# STAGE 1a — fetch
# --------------------------------------------------------------------------- #


def fetch(spec: SourceSpec) -> tuple[str, int]:
    """Fetch one allowlisted page, returning ``(html, status)``.

    Enforces C1 twice: the requested URL must be on the allowlist *before* any
    network call, and the final URL after redirects must still be on it. A
    redirect off the allowlist is a violation, never a silent follow.
    """
    if not is_allowed_url(spec.url):
        raise AllowlistViolation(f"refusing to fetch non-allowlisted URL: {spec.url}")
    if is_denied_host(spec.url):
        raise AllowlistViolation(f"refusing to fetch denied host: {spec.url}")

    timeout = float(SETTINGS.get("loading.timeout_s", 10))
    retries = int(SETTINGS.get("loading.retries", 2))
    headers = {
        "User-Agent": SETTINGS.get(
            "loading.user_agent",
            "Mozilla/5.0 (compatible; mf-faq-rag/1.0; educational demo)",
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=timeout, follow_redirects=True,
                      headers=headers) as client:
        for attempt in range(retries + 1):
            try:
                response = client.get(spec.url)
                if response.status_code != 200:
                    raise FetchError(
                        f"HTTP {response.status_code} for {spec.url} "
                        f"(attempt {attempt + 1}/{retries + 1})"
                    )
                final_url = str(response.url)
                if not is_allowed_url(final_url):
                    raise AllowlistViolation(
                        f"redirect left the allowlist: {spec.url} -> {final_url}"
                    )
                if is_denied_host(final_url):
                    raise AllowlistViolation(
                        f"redirect landed on a denied host: {final_url}"
                    )
                return response.text, response.status_code
            except (httpx.HTTPError, FetchError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))

    raise FetchError(f"failed to fetch {spec.url}: {last_error}")


# --------------------------------------------------------------------------- #
# STAGE 1b — extract
# --------------------------------------------------------------------------- #


def _strip_tags(fragment: str) -> str:
    """Flatten an HTML fragment to plain text."""
    text = _INLINE_HTML.sub(" ", fragment)
    return re.sub(r"\s+", " ", html_lib.unescape(text)).strip()


def _jsonld_blocks(soup: BeautifulSoup) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            blocks.extend(b for b in parsed if isinstance(b, dict))
        elif isinstance(parsed, dict):
            blocks.append(parsed)
    return blocks


def _extract_jsonld_faq(soup: BeautifulSoup) -> tuple[list[tuple[str, str]], int]:
    """Pull the schema.org FAQPage Q&A pairs. Returns ``(pairs, excluded_count)``."""
    pairs: list[tuple[str, str]] = []
    excluded = 0
    for block in _jsonld_blocks(soup):
        if str(block.get("@type", "")).lower() != "faqpage":
            continue
        entities = block.get("mainEntity") or []
        if isinstance(entities, dict):
            entities = [entities]
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            question = _strip_tags(str(entity.get("name", "")))
            answer_obj = entity.get("acceptedAnswer")
            if isinstance(answer_obj, list):
                answer_obj = answer_obj[0] if answer_obj else {}
            answer = _strip_tags(
                answer_obj.get("text", "") if isinstance(answer_obj, dict) else ""
            )
            if not question or not answer:
                continue
            if _EXCLUDE_PERFORMANCE and any(
                p.search(question) for p in EXCLUDED_FAQ_PATTERNS
            ):
                excluded += 1
                continue
            pairs.append((question, answer))
    return pairs, excluded


def _next_data(soup: BeautifulSoup) -> dict[str, Any] | None:
    """Return ``props.pageProps.mfServerSideData`` from ``__NEXT_DATA__``."""
    tag = soup.find("script", id="__NEXT_DATA__")
    if tag is None:
        return None
    raw = tag.string or tag.get_text() or ""
    try:
        payload = json.loads(raw)
        return payload["props"]["pageProps"]["mfServerSideData"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _endstop(text: str) -> str:
    """Append a full stop unless the text already ends with sentence punctuation.

    Some pages supply an ``exit_load`` string that already ends in a period (the
    Balanced Advantage scheme does), so naively appending one produced
    ".within 1 year.." in the corpus. Whitespace is stripped on *both* ends
    first: a trailing space would otherwise hide the existing punctuation from
    ``endswith``, and a leading one would survive into the corpus and shift every
    downstream sentence split.
    """
    stripped = text.strip()
    return stripped if stripped.endswith((".", "!", "?")) else stripped + "."



def _fmt_money(value: Any) -> str | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return f"Rs {int(number):,}"
    return f"Rs {number:,.2f}"


def _fmt_yes_no(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    text = str(value).strip()
    if text.lower() in {"true", "yes", "1"}:
        return "Yes"
    if text.lower() in {"false", "no", "0"}:
        return "No"
    return text or None


def _lock_in_phrase(lock_in: Any) -> str:
    """Render the lock-in object as prose, including the explicit "none" case.

    The absent case matters: for a non-ELSS scheme the honest, useful answer is
    "there is no lock-in", so the field is always emitted rather than skipped.
    """
    if not isinstance(lock_in, dict):
        return "not stated"
    parts = []
    for unit, label in (("years", "year"), ("months", "month"), ("days", "day")):
        raw = lock_in.get(unit)
        if raw in (None, "", 0, "0"):
            continue
        try:
            count = int(raw)
        except (TypeError, ValueError):
            continue
        parts.append(f"{count} {label}{'s' if count != 1 else ''}")
    if not parts:
        return "none"
    return " ".join(parts)


def _document_links(mf: dict[str, Any]) -> list[tuple[str, str]]:
    labels = (
        ("sid_url", "Scheme Information Document (SID)"),
        ("brochure_link", "Fund brochure"),
        ("scheme_info_link", "Scheme information link"),
    )
    links: list[tuple[str, str]] = []
    for key, label in labels:
        url = mf.get(key)
        if isinstance(url, str) and url.startswith("http"):
            links.append((label, url))
    return links


def _render_facts(mf: dict[str, Any]) -> list[str]:
    """Render the flat mfServerSideData object as labelled, self-contained facts.

    Each emitted line is written so that it reads correctly on its own once the
    chunker isolates it — e.g. not "Expense ratio: 0.78" but a full sentence.
    """
    name = str(mf.get("scheme_name") or "this scheme").strip()
    lines: list[str] = []

    category = " / ".join(
        str(mf.get(k)).strip()
        for k in ("category", "sub_category")
        if mf.get(k) not in (None, "", "None")
    )
    if category:
        lines.append(f"Category: {category}.")
    if mf.get("amc"):
        lines.append(f"Asset Management Company (AMC): {mf.get('amc')}.")
    plan_type = mf.get("plan_type")
    scheme_type = mf.get("scheme_type")
    if plan_type or scheme_type:
        lines.append(f"Plan: {' '.join(str(x) for x in (plan_type, scheme_type) if x)}.")

    expense = mf.get("expense_ratio")
    if expense not in (None, "", "0"):
        base = mf.get("base_expense_ratio")
        extra = f" (base expense ratio {base}%)" if base not in (None, "", "0") else ""
        lines.append(
            f"Expense ratio of {name} is {expense}%{extra}, charged annually on the "
            "assets of the scheme."
        )

    exit_load = str(mf.get("exit_load") or "").strip()
    if exit_load:
        lines.append(_endstop(f"Exit load of {name}: {exit_load}"))
    else:
        lines.append(f"Exit load of {name}: Nil.")

    min_sip = mf.get("min_sip_investment")
    if min_sip not in (None, "", 0, "0"):
        money = _fmt_money(min_sip)
        lines.append(
            f"Minimum SIP amount for {name} is {money} per month, with a minimum "
            f"additional SIP of {_fmt_money(mf.get('mini_additional_investment'))}."
        )
    min_lump = mf.get("min_investment_amount")
    if min_lump not in (None, "", 0, "0"):
        lines.append(f"Minimum lump sum investment for {name} is {_fmt_money(min_lump)}.")

    for key, label in (("sip_allowed", "SIP investment is allowed"),
                       ("lumpsum_allowed", "Lump sum investment is allowed")):
        rendered = _fmt_yes_no(mf.get(key))
        if rendered:
            lines.append(f"{label} for {name}: {rendered}.")

    lines.append(f"Lock-in period for {name}: {_lock_in_phrase(mf.get('lock_in'))}.")

    risk = str(mf.get("nfo_risk") or "").strip()
    if risk:
        # Pages are inconsistent: some carry the word "Riskometer" in the value
        # ("Moderately High Riskometer"), some do not ("Moderately High"). The
        # line already has the label, so drop a redundant suffix.
        if risk.lower().endswith("riskometer"):
            risk = risk[: -len("riskometer")].strip()
        lines.append(f"Riskometer for {name}: {risk}.")

    benchmark = str(mf.get("benchmark") or "").strip()
    if benchmark:
        full_name = str(mf.get("benchmark_name") or "").strip()
        suffix = f" ({full_name})" if full_name and full_name.lower() != benchmark.lower() else ""
        lines.append(f"Benchmark for {name}: {benchmark}{suffix}.")

    for label, url in _document_links(mf):
        lines.append(f"Official documents for {name} - {label}: {url}")

    for key, label in (
        ("launch_date", "Launch date"),
        ("fund_manager", "Fund manager"),
        ("registrar_agent", "Registrar and transfer agent"),
        ("portfolio_turnover", "Portfolio turnover ratio"),
        ("face_value", "Face value"),
    ):
        value = mf.get(key)
        if value not in (None, "", "None"):
            lines.append(f"{label} of {name}: {value}.")

    return lines


def _render_visible_text(soup: BeautifulSoup) -> str:
    """Best-effort visible prose, for narrative sections the JSON does not carry.

    trafilatura is tried first (it implements a readability-style extraction);
    BeautifulSoup4 over the Next.js root is the fallback. On these pages both are
    thin, which is exactly why the JSON path above is the primary one.
    """
    try:
        import trafilatura

        extracted = trafilatura.extract(
            str(soup), include_comments=False, include_tables=True,
        )
        if extracted and len(extracted.strip()) >= _MIN_EXTRACTED_CHARS:
            return extracted.strip()
    except Exception:
        pass

    root = soup.find(id="__next") or soup.body or soup
    for tag in root(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return _strip_tags(str(root))


def extract(html: str) -> dict[str, Any]:
    """Extract a document from page HTML.

    Returns a dict with ``title``, ``meta_description``, ``facts``, ``faqs``,
    ``visible`` and ``excluded_faq_count``. Raises :class:`ExtractionError` when
    the page yields too little to be a usable source.
    """
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = _strip_tags(soup.title.string)
    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = _strip_tags(meta_tag["content"])

    facts: list[str] = []
    mf = _next_data(soup)
    if isinstance(mf, dict):
        facts = _render_facts(mf)

    faqs, excluded_faq_count = _extract_jsonld_faq(soup)
    visible = _render_visible_text(soup)

    usable = len("\n".join(facts)) + sum(len(a) + len(b) for a, b in faqs)
    if usable < _MIN_EXTRACTED_CHARS and len(visible) < _MIN_EXTRACTED_CHARS:
        raise ExtractionError(
            f"extraction yielded {usable + len(visible)} usable chars "
            f"(minimum {_MIN_EXTRACTED_CHARS}); treating the page as unusable"
        )

    return {
        "title": title,
        "meta_description": meta_description,
        "facts": facts,
        "faqs": faqs,
        "visible": visible,
        "excluded_faq_count": excluded_faq_count,
    }


# --------------------------------------------------------------------------- #
# STAGE 1c — clean
# --------------------------------------------------------------------------- #

_BOILERPLATE = (
    "download the groww app", "open the groww app", "log in to groww",
    "cookie", "all rights reserved", "terms of use", "privacy policy",
    "disclosure:", "risk disclosures", "know your customer",
)


def clean(text: str) -> str:
    """Normalise whitespace and drop boilerplate lines.

    These pages carry almost no semantic nav/footer markup, so this is a
    light touch: strip control characters, collapse runs of spaces and blank
    lines, and drop obvious app-promotional lines.
    """
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_SPACE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip() and not _is_boilerplate(line)
    ]
    return "\n".join(lines).strip()


def _is_boilerplate(line: str) -> bool:
    lowered = line.lower()
    if len(lowered) > 200:
        return False
    return any(marker in lowered for marker in _BOILERPLATE)


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #

# Groww serves one identical SEO meta description on all five corpus pages,
# verified 2026-09-27. It reads:
#
#   "<Name> - Get latest NAV, SIP Returns & Rankings, Ratings, Fund Performance,
#    Portfolio, Expense Ratio, Holding Analysis, and Peers. Invest in <Name>
#    Online with Groww."
#
# It is pure marketing template: the only scheme-specific token in it is the
# name, which is already the document's ``#`` heading. Every other clause is
# either banned performance vocabulary (constraint C4 / C6) or a promise of
# content that does not exist on the page. Emitting it would have put "NAV",
# "SIP Returns", "Rankings" and "Fund Performance" into every committed chunk
# and failed the C4 corpus test. So it is dropped unless it is genuinely
# informative *and* performance-free.
_PERFORMANCE_NOISE = re.compile(
    r"\b(returns?|nav|navs|aum|aums|performance|rankings?|money multiplier|"
    r"since inception|ytd)\b",
    re.IGNORECASE,
)
_META_TEMPLATE = re.compile(r"get latest|sip returns\s*&|invest in .* online with", re.IGNORECASE)


def _usable_meta_description(text: str) -> str:
    """Return the meta description only if it is informative and performance-free."""
    cleaned = _strip_tags(text or "")
    if not cleaned:
        return ""
    if _META_TEMPLATE.search(cleaned):
        return ""
    if _PERFORMANCE_NOISE.search(cleaned):
        return ""
    return cleaned


def _assemble(spec: SourceSpec, page_title: str, extracted: dict[str, Any]) -> str:
    """Compose the corpus document for one scheme.

    The output is markdown-flavoured on purpose: the ``##``/``###`` headings give
    the ``heading_bound`` chunker (Phase 2) real structure to split on, and become
    each chunk's ``heading_trail`` in retrieval.
    """
    name = spec.scheme_name
    parts: list[str] = [f"# {name}", f"Source page: {spec.url}"]

    if page_title:
        parts.append(f"Page title: {page_title}")

    facts = extracted.get("facts") or []
    if facts:
        parts.append("## Scheme facts")
        parts.extend(facts)

    meta = _usable_meta_description(extracted.get("meta_description") or "")
    if meta:
        parts.append("## About this scheme")
        parts.append(meta)

    faqs = extracted.get("faqs") or []
    if faqs:
        parts.append("## Frequently asked questions")
        for question, answer in faqs:
            parts.append(f"### {question}")
            parts.append(answer)

    visible = extracted.get("visible") or ""
    if visible:
        parts.append("## Additional page text")
        parts.append(visible)

    return "\n\n".join(part for part in parts if part.strip())


def load_one(spec: SourceSpec) -> RawDoc:
    """Fetch, extract and clean one page into a :class:`RawDoc`."""
    html, status = fetch(spec)
    extracted = extract(html)

    body = _assemble(spec, extracted["title"], extracted)
    text = clean(body)
    if len(text) < _MIN_EXTRACTED_CHARS:
        raise ExtractionError(
            f"{spec.url}: cleaned text is only {len(text)} chars "
            f"(minimum {_MIN_EXTRACTED_CHARS})"
        )

    return RawDoc(
        source_id=spec.source_id,
        scheme_name=spec.scheme_name,
        category=spec.category,
        plan=spec.plan,
        url=spec.url,
        page_title=extracted["title"] or spec.scheme_name,
        text=text,
        fetched_at=utc_now(),
        http_status=status,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def load_all(
    specs: Iterable[SourceSpec] = ALLOWLIST,
    offline: bool = False,
) -> list[RawDoc]:
    """Load every allowlisted page.

    ``offline=True`` replays ``artifacts/raw_docs.jsonl`` with zero network
    calls, so a demo or a test never depends on a third party's uptime
    (architecture.md §9 A10).

    Fails loudly: if any page errors, the whole run aborts and nothing is
    written. A partial corpus must never ship, because the bot would then
    answer "not in sources" about facts that exist on the dead page.
    """
    specs = list(specs)
    expect_full_corpus = len(specs) == EXPECTED_SOURCE_COUNT

    if offline:
        return _load_offline(specs)

    docs: list[RawDoc] = []
    with stage_timer("load", **{"in": len(specs)}, out=0) as summary:
        for spec in specs:
            doc = load_one(spec)
            docs.append(doc)
            print(
                f"  {doc.source_id:<18} {doc.http_status}  "
                f"{len(doc.text):>7} chars  {doc.word_count:>5} words",
                flush=True,
            )
        summary["out"] = len(docs)
        summary["word_count"] = sum(d.word_count for d in docs)
        summary["sections"] = len(docs)

    if expect_full_corpus and len(docs) != EXPECTED_SOURCE_COUNT:
        raise ExtractionError(
            f"expected {EXPECTED_SOURCE_COUNT} documents, got {len(docs)}"
        )
    return docs


def _load_offline(specs: list[SourceSpec]) -> list[RawDoc]:
    """Replay previously ingested documents, validating they match the allowlist."""
    path: Path = SETTINGS.path("loading.raw_docs_path", "artifacts/raw_docs.jsonl")
    if not path.exists():
        raise ExtractionError(f"--offline requested but {path} does not exist")

    by_id: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            spec = next((s for s in specs if s.source_id == record["source_id"]), None)
            if spec is None or record["url"] != spec.url:
                continue
            by_id[record["source_id"]] = record

    missing = [s.source_id for s in specs if s.source_id not in by_id]
    if missing:
        raise ExtractionError(f"--offline corpus is missing sources: {missing}")

    docs = [
        RawDoc(
            source_id=record["source_id"],
            scheme_name=record["scheme_name"],
            category=record["category"],
            plan=record["plan"],
            url=record["url"],
            page_title=record["page_title"],
            text=record["text"],
            fetched_at=record["fetched_at"],
            http_status=record["http_status"],
            content_hash=record["content_hash"],
        )
        for record in (by_id[s.source_id] for s in specs)
    ]
    print(f"  replayed {len(docs)} documents from {path} (no network)", flush=True)
    return docs


# --------------------------------------------------------------------------- #
# STAGE 1d — persist
# --------------------------------------------------------------------------- #


def write_jsonl(docs: list[RawDoc]) -> Path:
    """Write ``artifacts/raw_docs.jsonl`` — committed, so any answer stays auditable."""
    path: Path = SETTINGS.path("loading.raw_docs_path", "artifacts/raw_docs.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    print(f"  wrote {len(docs)} documents to {path}", flush=True)
    return path


def write_sources_csv(docs: list[RawDoc]) -> Path:
    """Write the source list deliverable (FR-17, D3)."""
    path: Path = SETTINGS.path("loading.sources_csv_path", "data/sources.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "source_id", "scheme_name", "category", "plan", "url",
            "fetched_at", "http_status", "word_count", "notes",
        ])
        for doc in docs:
            writer.writerow([
                doc.source_id, doc.scheme_name, doc.category, doc.plan, doc.url,
                doc.fetched_at, doc.http_status, doc.word_count,
                "official public scheme page",
            ])
    print(f"  wrote source list to {path}", flush=True)
    return path


def describe(doc: RawDoc) -> str:
    """Short human summary used by the CLI and the eval harness."""
    return (
        f"{doc.source_id}: {len(doc.text)} chars, {doc.word_count} words, "
        f"fetched {doc.fetched_at}"
    )


__all__ = [
    "AllowlistViolation", "FetchError", "ExtractionError",
    "fetch", "extract", "clean", "load_one", "load_all",
    "write_jsonl", "write_sources_csv", "describe",
    "EXCLUDED_FIELDS", "EXCLUDED_FAQ_PATTERNS", "REQUIRED_FACT_FIELDS",
]
