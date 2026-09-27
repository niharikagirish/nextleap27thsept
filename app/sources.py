"""The corpus — exactly five public pages, per the class brief (C1, C2).

This is deliberately CODE rather than configuration: the corpus cannot be
widened at runtime by editing a YAML file or setting an environment variable.
``app/pipeline/loader.py`` validates every URL against ``ALLOWLIST`` *before*
fetching, and re-validates the final URL *after* redirects, so the allowlist
cannot be laundered through a redirect chain.

Verified 2026-09-27: all five URLs return HTTP 200 with no redirects and are
server-rendered Next.js pages. See ``docs/implementation.md`` Phase 1 for the
extraction findings that drove the loader design.
"""

from __future__ import annotations

from urllib.parse import urlparse

from .models import SourceSpec

ALLOWLIST: tuple[SourceSpec, ...] = (
    SourceSpec(
        source_id="hdfc-large-cap",
        scheme_name="HDFC Large Cap Fund",
        category="Large Cap",
        plan="Direct-Growth",
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    ),
    SourceSpec(
        source_id="hdfc-equity",
        scheme_name="HDFC Equity Fund (Flexi Cap)",
        category="Flexi Cap",
        plan="Direct-Growth",
        url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    ),
    SourceSpec(
        source_id="hdfc-elss",
        scheme_name="HDFC ELSS Tax Saver Fund",
        category="ELSS",
        plan="Direct-Growth",
        url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    ),
    SourceSpec(
        source_id="hdfc-small-cap",
        scheme_name="HDFC Small Cap Fund",
        category="Small Cap",
        plan="Direct-Growth",
        url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    ),
    SourceSpec(
        source_id="hdfc-balanced-adv",
        scheme_name="HDFC Balanced Advantage Fund",
        category="Balanced Advantage",
        plan="Direct-Growth",
        url="https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    ),
)

ALLOWED_HOSTS: frozenset[str] = frozenset({"groww.in", "www.groww.in"})

# Names people actually use, mapped to ``source_id``. The allowlist name for the
# Flexi Cap scheme is "HDFC Equity Fund (Flexi Cap)", which is the one a Groww
# page uses but which almost nobody writes - they say "HDFC Flexi Cap Fund". The
# intent gate needs to recognise that as a *listed* scheme; without these aliases
# it treats the corpus's own fifth scheme as an unknown fund and abstains on
# questions the corpus answers perfectly well.
#
# These are recognition aliases for matching only. They do not widen the corpus:
# the URLs in ALLOWLIST are still the only five pages that are ever fetched.
SCHEME_ALIASES: dict[str, tuple[str, ...]] = {
    "hdfc-large-cap": ("hdfc large cap",),
    "hdfc-equity": ("hdfc flexi cap fund", "hdfc flexi cap", "flexi cap fund"),
    "hdfc-elss": ("hdfc elss", "hdfc elss tax saver", "elss tax saver fund"),
    "hdfc-small-cap": ("hdfc small cap",),
    "hdfc-balanced-adv": ("hdfc balanced advantage",),
}

# Blog / platform / UGC domains. Defence in depth behind the allowlist (C2).
DENIED_HOST_SUFFIXES: tuple[str, ...] = (
    "medium.com", "blogspot.com", "wordpress.com", "reddit.com", "quora.com",
    "youtube.com", "youtu.be", "wikipedia.org", "wikimedia.org", "linkedin.com",
    "twitter.com", "x.com", "facebook.com", "instagram.com", "substack.com",
    "tradingview.com", "moneycontrol.com", "etmoney.com", "zerodha.com",
)

# Rendered as clickable chips in the UI (FR-07). Kept next to the allowlist so
# the demo questions and the corpus stay visibly in sync.
EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "What is the expense ratio of the HDFC Large Cap Fund?",
    "Does the HDFC ELSS Tax Saver Fund have a lock-in period?",
    "How do I download a capital gains statement?",
)

EXPECTED_SOURCE_COUNT: int = len(ALLOWLIST)


def spec_for(source_id: str) -> SourceSpec:
    """Look up an allowlist entry, or raise. Never constructs a spec on the fly."""
    for spec in ALLOWLIST:
        if spec.source_id == source_id:
            return spec
    raise KeyError(f"unknown source_id: {source_id!r} (not in ALLOWLIST)")


def is_allowed_url(url: str) -> bool:
    """True only for an exact ALLOWLIST match. Used pre-fetch AND post-redirect."""
    return any(url.rstrip("/") == spec.url.rstrip("/") for spec in ALLOWLIST)


def is_denied_host(url: str) -> bool:
    """True for blog / UGC / third-party hosts. Defence in depth behind the allowlist."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == suffix or host.endswith("." + suffix)
               for suffix in DENIED_HOST_SUFFIXES)
