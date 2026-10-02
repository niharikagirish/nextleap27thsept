# Sample Q&A

Verbatim assistant output from the live pipeline. Regenerate with

```
python -m app.sample_qa
```

Answers are produced, not authored - do not hand-edit this file.

Disclaimer shown with every answer: **Facts-only. No investment advice.**

Corpus: the five HDFC Direct-Growth scheme pages listed in `data/sources.csv`.

---


### Q1. What is the expense ratio of the HDFC Large Cap Fund?

*Intended behaviour: cited answer - actual: `factual`*

> The expense ratio of HDFC Large Cap Fund Direct Growth is 1.03%, with a base expense ratio of 0.84%.

**Sources**

- [HDFC Large Cap Fund](https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth)

Source fetched: 2026-09-27T10:59:40Z

### Q2. Does the HDFC ELSS Tax Saver Fund have a lock-in period?

*Intended behaviour: cited answer - actual: `factual`*

> Yes, the HDFC ELSS Tax Saver Fund Direct Plan Growth has a lock-in period of 3 years.

**Sources**

- [HDFC ELSS Tax Saver Fund](https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth)

Source fetched: 2026-09-27T10:59:43Z

### Q3. Who manages the HDFC Balanced Advantage Fund?

*Intended behaviour: cited answer - actual: `factual`*

> The fund managers of HDFC Balanced Advantage Fund Direct Growth are Anil Bamboli, Arun Agarwal, Dhruv Muchhal, Nandita Menezes, Gopal Agrawal, and Ihab Dalwai.

**Sources**

- [HDFC Balanced Advantage Fund](https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth)

Source fetched: 2026-09-27T10:59:45Z

### Q4. What is the minimum SIP amount for the HDFC Small Cap Fund?

*Intended behaviour: cited answer - actual: `factual`*

> The minimum SIP amount for HDFC Small Cap Fund Direct Growth is Rs 100 per month, with a minimum additional SIP of Rs 100.

**Sources**

- [HDFC Small Cap Fund](https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth)

Source fetched: 2026-09-27T10:59:44Z

### Q5. What benchmark does the HDFC Balanced Advantage Fund track?

*Intended behaviour: cited answer - actual: `factual`*

> The benchmark for HDFC Balanced Advantage Fund Direct Growth is the NIFTY 50 Hybrid Composite Debt 50:50 Index.

**Sources**

- [HDFC Balanced Advantage Fund](https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth)

Source fetched: 2026-09-27T10:59:45Z

### Q6. What are the 1 year returns of the HDFC Large Cap Fund?

*Intended behaviour: refused: performance - actual: `perf_redirect`*

> I don't state or compare returns. For audited performance figures, please see the official factsheet or scheme page.

**Sources:** none - no factual claim was generated.


### Q7. Should I invest in the HDFC Small Cap Fund?

*Intended behaviour: refused: advice - actual: `refusal`*

> I'm a facts-only assistant, so I can't give investment advice or tell you whether to buy, sell, or hold a scheme. I can share objective facts from official public pages - expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark - each with a source link. For suitability, please consult a SEBI-registered investment advisor.

**Sources:** none - no factual claim was generated.


### Q8. What is the expense ratio of HDFC Parag Parag Fund?

*Intended behaviour: abstained: not in corpus - actual: `abstain`*

> I could not find that in the official sources for these 5 HDFC schemes. Try asking about expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, or how to download statements.

**Sources:** none - no factual claim was generated.


### Q9. What is the fund size of the HDFC Large Cap Fund?

*Intended behaviour: abstained: AUM excluded - actual: `abstain`*

> I could not find that in the official sources for these 5 HDFC schemes. Try asking about expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, or how to download statements.

**Sources:** none - no factual claim was generated.

Source fetched: 2026-09-27T10:59:40Z

---

## Why the last four behave this way

- **Returns and NAV** are dropped at ingestion (`loader.EXCLUDED_FIELDS`) and
  refused again at output. The assistant never states a return figure.
- **Buy/sell/hold** is non-factual and is declined before retrieval runs.
- **Parag Parag Fund** is not one of the five allowlisted schemes, so the
  assistant must not answer it from a similar-sounding fund in the corpus.
- **Fund size (AUM)** is present on the live page but deliberately excluded from
  the corpus, so the assistant abstains rather than quoting it. See "Known
  limits" in the README.
