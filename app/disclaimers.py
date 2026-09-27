"""Every user-facing legal string, defined exactly once (FR-18, C9).

The UI, the CLI and the README all import from here, so the disclaimer cannot
drift between surfaces. Do not inline disclaimer text anywhere else — that is
precisely the failure mode FR-18 exists to prevent.
"""

SHORT = "Facts-only. No investment advice."

WELCOME = (
    "Hi! I answer facts about 5 HDFC mutual fund schemes using official public pages. "
    "Facts only, no investment advice."
)

FULL = """Facts-only. No investment advice.

This assistant answers objective questions about 5 HDFC mutual fund schemes using only
official public pages. It does not recommend, compare, or time any investment, and it does
not state or calculate returns. Figures are point-in-time snapshots - always verify against
the linked source page. For investment suitability, consult a SEBI-registered investment
advisor. Do not share PAN, Aadhaar, account numbers, OTPs, email addresses, or phone
numbers."""

PII_NOTICE = (
    "Please don't share personal identifiers such as PAN, Aadhaar, account numbers, OTP, "
    "email addresses, or phone numbers. I only answer fund facts from public pages."
)

REFUSAL = (
    "I'm a facts-only assistant, so I can't give investment advice or tell you whether to buy, "
    "sell, or hold a scheme. I can share objective facts from official public pages - expense "
    "ratio, exit load, minimum SIP, lock-in, riskometer, benchmark - each with a source link. "
    "For suitability, please consult a SEBI-registered investment advisor."
)
SEBI_EDU_URL = "https://www.sebi.gov.in/"

ABSTAIN_PREFIX = "I could not find that in the official sources for these 5 HDFC schemes."
ABSTAIN_SUGGESTIONS = (
    "Try asking about expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, "
    "or how to download statements."
)

PERF_REDIRECT = (
    "I don't state or compare returns. For audited performance figures, please see the "
    "official factsheet or scheme page."
)

LAST_UPDATED_LABEL = "Last updated from sources:"

FRESHNESS_CAVEAT = (
    "This is a point-in-time snapshot from the linked page, so verify the details there."
)

ERROR_GENERIC = (
    "I wasn't able to build a reliable answer just now. Please rephrase your question "
    "and try again."
)

NO_LLM_KEY = (
    "No LLM API key found - running in retrieval-only mode, so you can see the retrieved "
    "passages and scores without generated answers. Set LLM_API_KEY in .env for full answers."
)

SCOPE_SUMMARY = (
    "HDFC Asset Management - 5 schemes, all Direct-Growth: Large Cap, Flexi Cap, ELSS Tax "
    "Saver, Small Cap, and Balanced Advantage. Sourced from public Groww scheme pages only."
)
