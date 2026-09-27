# Known Limits of the Corpus

Everything here was found by reading the five allowlisted pages on 2026-09-27,
not by assuming. Each entry states what was found, why it matters, and what the
bot should do about it.

## 1. "How do I download a capital gains statement?" cannot be answered

**Found:** the class brief requires the bot to answer questions about downloading
capital-gains statements, and `app/sources.py: EXAMPLE_QUESTIONS` lists it as a
demo question. None of the five pages contains a statements guide. The JSON-LD
`FAQPage` has 8 entries per page and none of them is about statements; the only
document link field, `sid_url`, resolves to `https://www.hdfcfund.com` — the AMC
homepage — for **all five** schemes, and `brochure_link` is `null` everywhere.

**Consequence:** the only honest answer is a refusal that says the corpus does
not cover it and points at the AMC. `eval/golden.json` deliberately does **not**
contain a statement-download question, because labelling one would force a gold
chunk that does not exist and would quietly teach the experiment to reward
retrieving the wrong thing.

**If this must be demoed:** the fix is a corpus change (add the HDFC MF
statement pages to `ALLOWLIST`), not a prompt change. It needs the class brief's
sign-off because C1 fixes the corpus at five pages.

## 2. The SEO meta description was dropped entirely

**Found:** all five pages serve a byte-identical template meta description:

> `<Name> - Get latest NAV, SIP Returns & Rankings, Ratings, Fund Performance,
> Portfolio, Expense Ratio, Holding Analysis, and Peers. Invest in <Name> Online
> with Groww.`

It is marketing boilerplate whose only scheme-specific token is the name — which
is already the document's `#` heading. Emitting it would have put "NAV", "SIP
Returns", "Rankings" and "Fund Performance" into every committed chunk and
failed the C4 corpus test in `tests/test_loader.py`.

**Handled:** `app/pipeline/loader.py::_usable_meta_description` drops it, and
keeps a meta description only if it is neither template-shaped nor
performance-vocabulary-bearing.

## 3. `hdfc-equity` no longer says "Equity Fund"

**Found:** the page at `.../hdfc-equity-fund-direct-growth` reports its own name
as **"HDFC Flexi Cap Direct Plan Growth"**, not "HDFC Equity Fund (Flexi Cap)".
The URL and `SourceSpec.scheme_name` still say Equity/Flexi Cap.

**Why it matters:** a user typing "HDFC Equity Fund" must still match this page.
It does, because the URL, the `source_id`, and the page's own `HDFC Equity Fund`
references all keep the string in the corpus — but the two names differ, so an
answer that quotes the page name will look inconsistent with the question.

**Not "fixed":** silently rewriting the scheme name would mean the answer and the
citation disagree with each other. Both names are kept, and the answer uses
whichever the source page supplies.

## 4. Performance content is removed at ingestion, not just blocked at output

**Found:** the pages carry `return_stats` (1/3/5-year returns), NAV, AUM, and an
FAQ titled "What kind of returns does <fund> provide?".

**Handled:** `EXCLUDED_FAQ_PATTERNS` and `EXCLUDED_FIELDS` drop them in
`_render_facts`, so the committed corpus never contains them. C4 still runs at
output time, because the model can volunteer a return figure from parametric
memory regardless of what was retrieved — retrieval-time filtering cannot prevent
that.

**Care needed in Phase 6:** C4 must not ban *every* `%`. A blanket percentage
ban would also block the expense ratio, which is a required, in-corpus fact. The
guard has to target return-attached percentages and performance vocabulary, not
the character.

## 5. Benchmark and riskometer fields are inconsistently formatted upstream

**Found:** `nfo_risk` is `"Moderately High Riskometer"` on three pages and
`"Moderately High"` on two. `benchmark` and `benchmark_name` are distinct on
four pages and identical on the fifth.

**Handled:** the redundant `" Riskometer"` suffix is stripped and the redundant
benchmark alias is suppressed, so the corpus never says "Riskometer: Moderately
High Riskometer".

## 6. Exit-load strings are free text and not normalised

**Found:** the Balanced Advantage page supplies a conditional string —
`Exit Load for units in excess of 15% of the investment,1% will be charged for
redemption within 1 year.` — with no space after the comma, and already ending in
a full stop.

**Handled:** the trailing-period case is fixed by `_endstop`, which was a real
double-punctuation bug found while building `eval/golden.json`. The missing space
after the comma is left exactly as published: altering quoted source text to look
tidier is the wrong trade when the number must be traceable to the page.
