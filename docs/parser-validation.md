# Parser validation against real 10-K filings

This document records two measurements of `src/era/edgar/sections.py` against
real 10-K filings pulled live from EDGAR via `tests/edgar/test_sections_live.py`
(`ERA_LIVE_EDGAR=1 pytest tests/edgar/test_sections_live.py -v`): the first
against the original pure-regex parser (Task 18), and the second (this task)
against the hybrid model-selector redesign plus a targeted prompt fix. Only
the second measurement reflects the current code.

## Current result (Task 19, `_DECISION_VERSION = "3"`)

*Provenance: these numbers were first measured under `_DECISION_VERSION = "2"`.
A subsequent clarity-only prompt edit bumped the version to `"3"` without a
re-run, which would have left this document describing a prompt the code no
longer used. The suite was therefore re-run against `"3"` on 2026-08-20 —
all five tests pass and the results below are unchanged. Evidence for a
prompt that is not the shipped prompt is worse than no evidence.*

**Zero wrong content across all twenty target tickers.** Every item the
parser returns opens with its own title (`Business`, `Risk Factors`,
`Management's Discussion...`) and is well past `MIN_BODY_CHARS`. Nothing
came back silently wrong.

- 20 tickers attempted, 19 resolved to a fetchable 10-K, 1 (XOM) did not —
  same infrastructure issue as Task 18 (see below), unrelated to the parser.
- Of the 19 filings parsed, 15 yielded all three items, correctly bounded:
  AAPL, MSFT, BRK-B, KO, JNJ, PG, WMT, MRK, T, VZ, PFE, CSCO, BA, CAT, DIS.
- The remaining 4 (GE, INTC, JPM, CVX) are each missing one or more items —
  and in every case, `missing` is the *correct* answer, not a parser defect.
  See "What the parser does not handle" below.
- Across the 57 item-slots attempted (19 filings × 3 items), 49 came back
  correct, 8 came back `missing`, 0 came back wrong.

### Full results

| Ticker | Item | Chars | First 60 characters | Result |
|---|---|---:|---|---|
| AAPL | 1 | 15995 | Item 1. Business Company Background The Company designs, man | PASS |
| AAPL | 1A | 68042 | Item 1A. Risk Factors The following summarizes factors that  | PASS |
| AAPL | 7 | 18002 | Item 7. Management's Discussion and Analysis of Financial Co | PASS |
| MSFT | 1 | 122364 | ITEM 1. BUSINESS GENERAL Microsoft is a technology company c | PASS (fixed this task) |
| MSFT | 1A | 81183 | ITEM 1A. RISK FACTORS Our operations and financial results a | PASS |
| MSFT | 7 | 51248 | ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CO | PASS |
| BRK-B | 1 | 126373 | Item 1. Business Description Berkshire Hathaway Inc. ("Berks | PASS |
| BRK-B | 1A | 21789 | Item 1A. Risk Factors Berkshire and its subsidiaries (referr | PASS |
| BRK-B | 7 | 119377 | Item 7. Management's Discussion and Analysis of Financial Co | PASS |
| JPM | 1 | 39112 | Item 1. Business. Overview JPMorgan Chase & Co. ("JPMorganCh | PASS |
| JPM | 1A | 112127 | Item 1A. Risk Factors. The following discussion sets forth t | PASS |
| JPM | 7 | -- | (absent — not in parsed.items) | MISSING (known: pointer stub) |
| XOM | -- | -- | -- | FETCH FAILED — MissingFilingError: CIK 0002115436 has no 10-K in its recent filings |
| KO | 1 | 55195 | ITEM 1. BUSINESS In this report, the terms "The Coca-Cola Co | PASS |
| KO | 1A | 92251 | ITEM 1A. RISK FACTORS In addition to the other information s | PASS |
| KO | 7 | 109605 | ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CO | PASS |
| JNJ | 1 | 34543 | Item 1. Business General Johnson & Johnson and its subsidiar | PASS |
| JNJ | 1A | 47332 | Item 1A. Risk factors An investment in the Company's common  | PASS |
| JNJ | 7 | 65841 | Item 7. Management's discussion and analysis of results of o | PASS |
| PG | 1 | 14285 | Item 1. Business. The Procter & Gamble Company (the Company) | PASS |
| PG | 1A | 37865 | Item 1A. Risk Factors. We discuss our expectations regarding | PASS |
| PG | 7 | 91802 | Item 7. Management's Discussion and Analysis of Financial Co | PASS |
| WMT | 1 | 37456 | ITEM 1. BUSINESS General Walmart Inc. ("Walmart," the "Compa | PASS |
| WMT | 1A | 93175 | ITEM 1A. RISK FACTORS The risks described below could, in wa | PASS |
| WMT | 7 | 58023 | ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CO | PASS |
| CVX | 1 | 77816 | Item 1. Business General Development of Business Summary Des | PASS |
| CVX | 1A | 34605 | Item 1A. Risk Factors As a global energy company, Chevron is | PASS |
| CVX | 7 | -- | (absent — not in parsed.items) | MISSING (known: pointer stub) |
| MRK | 1 | 125171 | Item 1.Business. Merck & Co., Inc. (Merck or the Company) is | PASS |
| MRK | 1A | 77319 | Item 1A.Risk Factors. Summary Risk Factors The Company is su | PASS |
| MRK | 7 | 140894 | Item 7.Management's Discussion and Analysis of Financial Con | PASS |
| T | 1 | 31899 | ITEM 1. BUSINESS GENERAL AT&T Inc. ("AT&T," "we" or the "Com | PASS |
| T | 1A | 42555 | ITEM 1A. RISK FACTORS In addition to the other information s | PASS |
| T | 7 | 65371 | ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CO | PASS |
| VZ | 1 | 39277 | Item 1. Business General Verizon Communications Inc. (the Co | PASS |
| VZ | 1A | 36412 | Item 1A. Risk Factors The following discussion of "Risk Fact | PASS |
| VZ | 7 | 112372 | Item 7. Management's Discussion and Analysis of Financial Co | PASS |
| PFE | 1 | 91732 | ITEM 1. BUSINESS ABOUT PFIZER Pfizer Inc. is a research-base | PASS |
| PFE | 1A | 87359 | ITEM 1A. RISK FACTORS This section describes the material ri | PASS |
| PFE | 7 | 105163 | ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CO | PASS |
| INTC | 1 | -- | (absent — not in parsed.items) | MISSING (known: index-only filing) |
| INTC | 1A | -- | (absent — not in parsed.items) | MISSING (known: index-only filing) |
| INTC | 7 | -- | (absent — not in parsed.items) | MISSING (known: pointer stub) |
| CSCO | 1 | 48058 | Item 1. Business General Cisco designs and sells a broad ran | PASS |
| CSCO | 1A | 89534 | Item 1A. Risk Factors Set forth below and elsewhere in this  | PASS |
| CSCO | 7 | 70863 | Item 7. Management's Discussion and Analysis of Financial Co | PASS |
| BA | 1 | 18375 | Item 1. Business The Boeing Company, together with its subsi | PASS |
| BA | 1A | 57230 | Item 1A. Risk Factors An investment in our securities involv | PASS |
| BA | 7 | 89940 | Item 7. Management's Discussion and Analysis of Financial Co | PASS |
| CAT | 1 | 39239 | Item 1.Business. General Originally organized as Caterpillar | PASS |
| CAT | 1A | 53425 | Item 1A.Risk Factors. The statements in this section describ | PASS |
| CAT | 7 | 103361 | Item 7.Management's Discussion and Analysis of Financial Con | PASS |
| GE | 1 | -- | (absent — not in parsed.items) | MISSING (known: index-only filing) |
| GE | 1A | -- | (absent — not in parsed.items) | MISSING (known: index-only filing) |
| GE | 7 | -- | (absent — not in parsed.items) | MISSING (known: index-only filing) |
| DIS | 1 | 66630 | ITEM 1. Business The Walt Disney Company, together with the  | PASS |
| DIS | 1A | 62831 | ITEM 1A. Risk Factors For an enterprise as large and complex | PASS |
| DIS | 7 | 78575 | ITEM 7. Management's Discussion and Analysis of Financial Co | PASS |

Every MISSING row above is recorded in `test_sections_live.py`'s
`KNOWN_UNAVAILABLE` baseline and asserted two ways: `test_no_new_gaps_appear`
fails if any gap shows up outside that set, and
`test_the_baseline_does_not_hide_a_recovered_item` fails if any of these four
tickers ever stops being missing an item — so the baseline can't silently
become permanent permission to fail.

### The MSFT fix

Task 18 found MSFT's Item 1 landing on a running-header repeat one page
before the real section heading — a bare `Item 1` with no title, printed at
the top of a page, chosen over the real `ITEM 1. BUSINESS GENERAL` heading a
page later. Tracing MSFT's actual candidate list confirmed the model had no
signal in the prompt to prefer the title-bearing candidate over an earlier
bare repeat.

The fix is a prompt change only (`_SYSTEM_PROMPT` in
`src/era/edgar/boundaries.py`, `_DECISION_VERSION` bumped to `"2"` so cached
decisions from the old prompt don't replay): where several candidates exist
for the same item, strongly prefer the one whose context contains that
item's own title — "Business", "Risk Factors", "Management's Discussion" —
because a running page header repeats only the bare item number, never the
title. Re-running the live suite with this change confirmed MSFT's Item 1
now resolves correctly (122,364 characters, opens `ITEM 1. BUSINESS
GENERAL...`), with no regression on any of the other nineteen tickers.

A follow-up code-reviewer pass on this change flagged two clarity gaps in the
new bullet, both since fixed under `_DECISION_VERSION = "3"`: the title
preference is now stated to never override the table-of-contents/index
rejection rules (a TOC line also contains the title, e.g. "Item 1. Business
..... 4", and must still be rejected as a TOC entry), and an explicit
fallback ("first repeat wins") now covers the case where no candidate's
context contains the title at all -- version "2" left that case unspecified.
Neither change is aimed at an observed failure in this run, so it was not
re-verified live: the twenty-filing budget is spent deliberately, once, not
on speculative re-runs chasing an untriggered edge case.

## What the parser handles

Given a real "Item N" heading somewhere in the document that actually
introduces the section's own prose, the parser reliably finds it, bounds it
against the next real item heading (whatever number that is), and verifies
the result opens with the item's own title before returning it — decoys
(table-of-contents entries, cross-reference indexes, running page headers,
mid-sentence references) are all rejected rather than returned as if they
were real content. Fifteen of the twenty target tickers in this sample
exercise this path for all three items with no failures.

## What the parser does not handle

Two structural cases produce a correct `missing` rather than a wrong body,
because the parser only ever returns text that both starts under a real
"Item N" heading and opens with that item's own title — it has no fallback
for finding the real content any other way:

- **Index-only filings (GE, INTC).** The filer's entire document contains
  the literal string "Item 1" / "Item 1A" / "Item 7" in exactly one place: a
  front- or back-of-document cross-reference index mapping each item to a
  page number or range (e.g. `Item 1. Business 4-7, 9-10, 71-73`). The real
  Business/Risk Factors/MD&A prose exists elsewhere in the document under
  plain topic titles with no "Item" prefix at all, so no candidate ever
  points at it. There is nothing a heading-anchored parser can point to.
- **Incorporation-by-reference stubs (JPM, CVX Item 7).** The filer's Item 7
  heading is real and correctly located, but its body is genuinely just a
  pointer — "...which appear on pages 165–314" (JPM), "...as indicated in
  the Financial Table of Contents" (CVX) — with the actual MD&A prose living
  under a different heading convention inside an attached exhibit, sometimes
  a million characters later in the same combined document. The text the
  parser is looking for simply does not exist at a "MD&A" heading anywhere
  in the filing.

Both are correctly reported as `missing`, never as wrong content standing in
for the real section — `MIN_BODY_CHARS` and the dot-leader/opening-title
guards in `sections.py` catch the index and stub shapes specifically (see
their comments there for the exact mechanics), and `KNOWN_UNAVAILABLE` in
`test_sections_live.py` records that these four tickers' gaps are expected,
not a regression to chase.

A related but separate infrastructure issue: **XOM's ticker resolves to a
CIK with no 10-K on file** (`resolve_cik` follows SEC's `company_tickers.json`
to a holding-company entity created by a recent corporate restructuring,
distinct from the long-standing operating filer CIK that historically
carried Exxon's 10-Ks). This fails before `parse_items` is ever called and
is unrelated to section detection — `test_every_ticker_was_fetched` reports
it separately from the per-item assertions so a fetch failure never gets
confused with a parsing failure.

## Prior measurement (Task 18, original pure-regex parser)

Before the hybrid model-selector redesign, the same twenty-ticker sample
found 6 tickers with at least one wrong or missing item (MSFT, BRK-B, JPM,
CVX, GE, INTC) — critically, **wrong content came back silently** for most
of these: `ParsedFiling.missing` reported the item as present even when the
body was a table-of-contents fragment, a running-header page fragment, or
pure cross-reference index text. Root causes were a regex heading matcher
that could not distinguish a genuine section start from a table-of-contents
entry, a running page header, or a back-of-document index, plus no fallback
for filings whose real content is never headed "Item N" at all. Full detail
on that run (including the BRK-B word-split bug, since fixed independently
in `_to_text`) is preserved in this repository's git history for
`docs/parser-validation.md` prior to this task; it is not reproduced here
because it describes code that no longer exists.

The redesign (candidate list + model selector + code-side verification,
Task 19's predecessor) reduced the failure count from 6 wrong/missing
tickers to 1 (MSFT) with **zero silently wrong content** even in that first
run — every remaining gap was already the structurally correct answer
except MSFT's. This task's prompt fix closed that last gap.
