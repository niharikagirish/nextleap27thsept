"""Two-tier intent gate (FR-03, FR-04, C4).

Placed **before** retrieval and before any LLM call, because the cheapest and
safest way to handle a question you will not answer is to never fetch and never
prompt for it.

The two tiers are not interchangeable, and keeping them apart is the point:

* **Tier 1 — out of scope by policy.** Returns, comparisons, and buy/sell/hold
  language are refused for *any* scheme, including ones in the corpus, and
  regardless of how confidently the corpus could have answered. This is a
  product constraint (C4), not a knowledge gap, so it does not depend on what
  happens to be retrievable. Refuse with an explanation and an offer of
  objective facts.

* **Tier 2 — in scope, but not in the corpus.** Statement downloads, folio
  numbers, and "is my portfolio diversified" are legitimate HDFC questions that
  a five-page public corpus cannot answer. These **abstain** rather than refuse:
  abstaining admits the gap rather than implying the question was forbidden.
  Saying "I can't help with that" to someone asking how to download their own
  statement is a worse answer than "that isn't in my sources".

Tier 2 also covers two scope questions that are easy to miss and expensive to
get wrong:

* **An unlisted scheme.** "What is the expense ratio of the HDFC Mid-Cap
  Opportunity Fund?" has no page in this corpus, and a retrieval system will
  cheerfully return the *Large Cap* ratio and attach a real groww.in link to it.
  A genuine citation is what makes that error convincing, so it is caught here,
  before retrieval, rather than left to the score floor.
* **Off topic.** Under C1 there is no query-time web search, so a question
  naming no allowlisted scheme and no corpus topic can only be answered from
  outside the corpus. That is a *whitelist* test, not a blocklist: see
  :func:`is_off_topic`.

The distinction is a UX and an honesty property, not a technical one: a refusal
says "I won't", an abstention says "I don't know". Getting them backwards makes
the bot look either hostile or unreliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Intent = Literal["factual", "refusal", "abstain", "perf_redirect"]

# A bounded "rest of the clause" window that steps *over* decimal numbers
# instead of stopping at the dot. Without this, "Is the expense ratio of 0.87%
# high?" breaks: the "." in 0.87 ends the window, so the evaluation adjective is
# out of reach and a request for an opinion is answered as a plain fact.
#
# The decimal alternative is listed FIRST on purpose. If the generic single
# character were tried first it would consume the digit before the dot and then
# have no way to match the dot itself, and the whole clause would still fail.
_GAP = r"(?:(?:\d\.\d)|[^.!?\n])"

# --- Tier 1: policy -------------------------------------------------------- #
# Performance / return language. Ordered longest-first so "yearly return" beats
# "return" and the label is the most specific one available.
_PERFORMANCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("returns_and_cagr",
     re.compile(r"(?i)\b(?:cagr|annualized?\s+return|average\s+return|"
                r"rolling\s+return|yearly\s+return|returns?\s+(?:of|since|over|for)|"
                r"has\s+it\s+(?:out)?performed)\b")),
    ("performance_words",
     re.compile(r"(?i)\b(?:performance|perform(?:ed|s|ing)?|how\s+did\s+it\s+do|"
                r"worst\s+drawdown|drawdown|volatility|sharpe|sortino|alpha|beta|"
                r"benchmark\s+return|beat\s+the\s+benchmark|outperform(?:ed|ing)?)\b")),
    # NAV is a performance figure, not a fund fact: stating it means stating a
    # price at a point in time, which the answer policy forbids outright.
    ("nav",
     re.compile(r"(?i)\bnav\b")),
    # AUM plus growth is a performance signal. The two can sit far apart - "the
    # AUM of HDFC Balanced Advantage Fund and has it grown" puts a whole scheme
    # name between them - so this is a lookahead over the rest of the question
    # rather than a bounded window. A bare "what is the AUM" stays answerable:
    # a fund's size is a fact, its trajectory is performance.
    ("fund_aum_growth",
     re.compile(r"(?i)\b(?:aum|assets?\s+under\s+management)\b(?=[\s\S]*"
                r"\b(?:grown|grew|growing|risen|increased|increased\s+over|"
                r"growth\s+of|has\s+it\s+(?:grown|increased|risen))\b)")),
    # "If I invest 5000 a month for 10 years what will I get?" names no scheme
    # and no corpus topic, so without this it would fall through to an
    # off-topic abstention and tell the user the question was unrecognised
    # rather than that the bot will not forecast.
    ("growth_projection",
     re.compile(r"(?i)\b(?:what\s+will\s+i\s+(?:get|earn|receive|make)|"
                r"how\s+much\s+(?:will|would)\s+i\s+(?:get|earn|have)|"
                r"i\s+will\s+(?:get|earn)\b|final\s+value|"
                r"maturity\s+(?:value|amount|benefit)|worth\s+after)\b")),
    # A bare "return" is ambiguous ("what is the return period?"). Only count it
    # when paired with a return-ish noun so we do not refuse bookkeeping questions.
    ("bare_return_noun",
     re.compile(r"(?i)\b(?:returns?|cagr)\b(?![\s\S]{0,15}?\b(?:period|load|"
                r"ratio|amount|fee|charge|tax|window)\b)")),
)

# Recommendation / advice language. This is the FR-05 refusal trigger.
_ADVICE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("buy_sell_hold",
     re.compile(r"(?i)\b(?:should\s+i|would\s+you|which\s+one\s+should|is\s+it\s+a\s+good"
                r"|good\s+time\s+to)\b" + _GAP + r"{0,40}?"
                r"\b(?:buy|sell|invest|hold|redeem|switch|exit|allocate|start)\b")),
    ("recommendation",
     re.compile(r"(?i)\b(?:recommend(?:ation|ed)?|suggest(?:ion|ed)?|advise|advice|"
                r"best\s+(?:fund|scheme|option|choice)|top\s+pick|worth\s+investing|"
                r"good\s+investment)\b")),
    # Both "for me" and "for my ..." are how people actually ask, and "does it
    # suit me" is the most common phrasing of all. Matching only the first-person
    # singular object ("safe for me") misses "safe for my retirement" entirely.
    ("risk_suitability",
     re.compile(r"(?i)\b(?:fits?\s+my|suit(?:s)?\s+me|"
                r"suit(?:able)?\s+for\s+(?:me|my)|"
                r"safe\s+for\s+(?:me|my)|"
                r"should\s+i\s+be\s+(?:in|invested|allocated)|"
                r"right\s+for\s+me|risk\s+profile)\b")),
    ("comparison",
     re.compile(r"(?i)\b(?:better|best|versus|vs\.?|compare[sd]?|comparison|"
                r"outperform\w*)\b" + _GAP + r"{0,50}?"
                r"\b(?:fund|scheme|elss|equity|small\s+cap|large\s+cap|balanced)\b")),
    # Judging a fee is a recommendation. Fee term FIRST, evaluation second, so
    # "What is the expense ratio?" stays answerable while "Is the expense ratio
    # of 0.87% high or low?" does not - the difference is the adjective, and
    # only one of the two is asking for an opinion.
    ("evaluative",
     re.compile(r"(?i)\b(?:expense\s+ratio|exit\s+load|entry\s+load|ter\b|fees?|"
                r"charges?)\b" + _GAP + r"{0,30}?"
                r"\b(?:high|low|expensive|cheap|too\s+high|too\s+low|good|bad|"
                r"worth|fair|unfair|competitive|overpriced|underpriced)\b")),
)

# --- Tier 2: in scope, absent from a 5-page public corpus ------------------- #
_ABSENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("statements_documents",
     re.compile(r"(?i)\b(?:capital\s+gains?|consolidated|account)\s+statement\b|"
                r"\bstatement(?:s)?\s+(?:download|generate|get|fetch)\b|"
                r"\bdownload\s+(?:my\s+|the\s+)?(?:statement|statement[s]?|report|"
                r"tax\s+document|documents?)\b|"
                r"\b(?:tax\s+)?statement\s+(?:for\s+)?(?:fy\s*)?\d{4}\b|"
                r"\bhow\s+do\s+i\s+(?:get|download|generate)\b[^.!?\n]{0,30}?"
                r"\bstatement\b")),
    ("personal_account_data",
     re.compile(r"(?i)\b(?:my\s+(?:folio|holdings?|portfolio|sip|account|nav|"
                r"investments?)|folio\s+number|switch\s+from\s+folio|"
                r"update\s+my\s+(?:bank|mandate|details?)|"
                r"change\s+my\s+(?:bank|address|email|phone))\b")),
    ("account_operations",
     re.compile(r"(?i)\b(?:register|registration|open\s+(?:an?\s+)?account|"
                r"nominee|add\s+nominee|terminate|closure|grievance|"
                r"complaint|toll\s+free|customer\s+care)\b")),
    ("process_timing",
     re.compile(r"(?i)\b(?:when\s+(?:will|does|is|are)\b" + _GAP + r"{0,40}?"
                r"\b(?:credit|process|refund|settle|allot|confirm)\b|"
                r"\bhow\s+long\s+does\b" + _GAP + r"{0,30}?"
                r"\b(?:take|settlement)\b)")),
    # The 3-year lock-in is a corpus fact; "how much tax will I save" is a
    # computation that is not, and answering it needs an input we do not have
    # (the investor's own contribution and regime).
    ("tax_computation",
     re.compile(r"(?i)\bhow\s+much\s+tax\b|\btax\s+savings?\b|\btax\s+benefit\b|"
                r"\bsection\s+80[cd]\b|\bhow\s+much\s+can\s+i\s+save\b")),
)

# A mutual-fund name in running text: "HDFC Mid-Cap Opportunity Fund",
# "SBI Bluechip Fund". Used to spot a scheme that is NOT on the allowlist.
_FUND_MENTION = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9'\-]*\s+){0,5}[A-Z][A-Za-z0-9'\-]*\s+Fund)\b"
)

# Vocabulary that marks a question as a fund-fact question even with no scheme
# named. Sourced from the allowlist scheme names and the golden question set, so
# it covers what the corpus can actually answer and nothing more.
_IN_SCOPE_TOPICS = re.compile(
    r"(?i)\b(?:expense\s+ratio|exit\s+load|entry\s+load|loads?\b|"
    r"minimum\s+(?:sip|investment|amount)|sip\b|lock[\s\-]?in|"
    r"riskometer|risk\s+ometer|risk\s+category|benchmark|"
    r"fund\s+manager|who\s+manages|sebi\b|direct[\s\-]?growth|"
    r"charges?\b|fees?\b|ter\b|nav\b|aum\b|tax\b|"
    r"contact\s+detail|toll[\s\-]?free|switch\b|units?\b|"
    r"minimum\b|maximum\b)\b"
)


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    label: str = ""
    matched: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.intent == "factual"


def _first_hit(
    question: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]
) -> tuple[str, str] | None:
    for name, pattern in patterns:
        if pattern.search(question):
            return name, name
    return None


def _normalise(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", name.lower()).split())


def _contains(haystack: str, needle: str) -> bool:
    """Whole-token containment on space-joined normalised text.

    Padding with spaces stops "hdfc large cap fund" from matching inside
    "hdfc large cap fund house", and stops "cap fund" from matching inside
    "capital fund".
    """
    return f" {needle} " in f" {haystack} "


def _allowlisted_names() -> list[str]:
    """Allowlist scheme names plus recognition aliases, all normalised."""
    from ..sources import ALLOWLIST, SCHEME_ALIASES

    names = [_normalise(spec.scheme_name) for spec in ALLOWLIST]
    for aliases in SCHEME_ALIASES.values():
        names.extend(_normalise(alias) for alias in aliases)
    return names


def _matches_allowlist(candidate: str, allowed: list[str]) -> bool:
    """True when ``candidate`` refers to an allowlisted scheme.

    **Token-subset** matching, because the two failure directions are not
    symmetric:

    * Equality would abstain on the corpus's own fifth scheme - the allowlist
      name is "HDFC Equity Fund (Flexi Cap)" while everyone writes "HDFC Flexi
      Cap Fund" (handled by :data:`SCHEME_ALIASES`).
    * Plain substring containment would *accept* an invented fund, because
      "hdfc large cap" sits inside "hdfc large cap mid cap fund". A bot that
      accepts a fund it does not have answers from the nearest page it does have,
      with a real citation. Requiring the candidate's tokens to be a subset of a
      known name's tokens rejects that, and still accepts every genuine
      abbreviation.
    """
    tokens = set(candidate.split())
    if not tokens:
        return False
    return any(tokens <= set(name.split()) for name in allowed)


def mentions_unlisted_scheme(question: str) -> bool:
    """True when the question names a mutual fund that is NOT on the allowlist.

    This closes a failure mode that is far worse than an abstention: asked
    "What is the expense ratio of the HDFC Mid-Cap Opportunity Fund?", a
    retrieval system with no such page will happily return the *Large Cap*
    expense ratio, attach a real groww.in citation, and state it confidently. A
    real-looking link is exactly what makes that error convincing to a user, so
    the check happens before retrieval rather than being left to the score floor.
    """
    allowed = _allowlisted_names()
    for candidate in _FUND_MENTION.findall(question):
        if not _matches_allowlist(_normalise(candidate), allowed):
            return True
    return False


def mentions_allowlisted_scheme(question: str) -> bool:
    normalised = _normalise(question)
    return any(_contains(normalised, name) for name in _allowlisted_names())


def is_off_topic(question: str) -> bool:
    """True when nothing in the question points at the 5 schemes or a fund fact.

    A whitelist test rather than a blocklist: under C1 there is no web search at
    query time, so a question naming no scheme and no corpus topic can only be
    answered from outside the corpus. Catching it here means an honest
    abstention instead of a retrieval that returns the least-irrelevant chunk in
    the index.
    """
    if mentions_allowlisted_scheme(question):
        return False
    return _IN_SCOPE_TOPICS.search(question) is None


def classify(question: str) -> IntentResult:
    """Tier-1 gate, then Tier-2. First match wins, most specific tier first."""
    text = (question or "").strip()
    if not text:
        return IntentResult("abstain", "empty", "empty")

    advice = _first_hit(text, _ADVICE_PATTERNS)
    if advice:
        return IntentResult("refusal", "advice_request", advice[0])

    performance = _first_hit(text, _PERFORMANCE_PATTERNS)
    if performance:
        return IntentResult("perf_redirect", "performance_request", performance[0])

    absent = _first_hit(text, _ABSENT_PATTERNS)
    if absent:
        return IntentResult("abstain", "out_of_corpus", absent[0])

    if mentions_unlisted_scheme(text):
        return IntentResult("abstain", "unlisted_scheme", "unlisted_scheme")

    if is_off_topic(text):
        return IntentResult("abstain", "off_topic", "off_topic")

    return IntentResult("factual")


__all__ = [
    "classify", "IntentResult", "Intent", "mentions_unlisted_scheme",
    "mentions_allowlisted_scheme", "is_off_topic",
    "_PERFORMANCE_PATTERNS", "_ADVICE_PATTERNS", "_ABSENT_PATTERNS",
]
