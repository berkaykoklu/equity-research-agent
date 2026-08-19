# Parser validation against real 10-K filings

`src/era/edgar/sections.py` had, until this task, only ever been tested against
fixtures hand-written by the plan author. This document records what happened
the first time `parse_items` was run against twenty real 10-K filings pulled
live from EDGAR, via `tests/edgar/test_sections_live.py`
(`ERA_LIVE_EDGAR=1 pytest tests/edgar/test_sections_live.py -v`).

**Result: 6 of the 20 target tickers produced at least one wrong or missing
item, plus one ticker that never reached the parser at all.** This is a
measurement, not a fix — no changes were made to `sections.py`.

## Summary

- 20 tickers attempted, 19 resolved to a fetchable 10-K, 1 (XOM) did not.
- Of the 19 filings actually parsed, 13 were completely clean (all three
  items, correct opening, plausible length): AAPL, KO, JNJ, PG, WMT, MRK, T,
  VZ, PFE, CSCO, BA, CAT, DIS.
- 6 filings had at least one bad item: MSFT, BRK-B, JPM, CVX, GE, INTC.
- Across the 57 item-slots that were actually attempted (19 filings × 3
  items), 13 came back wrong — a 23% item-level failure rate on this sample,
  concentrated in 6 of 20 companies rather than spread evenly.

## Full results

| Ticker | Item | Chars | First 60 characters | Result |
|---|---|---:|---|---|
| AAPL | 1 | 16042 | Business Company Background The Company designs, manufacture | PASS |
| AAPL | 1A | 68035 | Risk Factors The following summarizes factors that could hav | PASS |
| AAPL | 7 | 18009 | Management's Discussion and Analysis of Financial Condition  | PASS |
| MSFT | 1 | 4971 | The gamer remains at the heart of the XBOX ecosystem. We are | FAIL (wrong opening) |
| MSFT | 1A | 5899 | Commission on January 19, 2024 and amended on March 8, 2024, | FAIL (wrong opening) |
| MSFT | 7 | 4861 | Cash, Cash Equivalents, and Investments Cash, cash equivalen | FAIL (wrong opening) |
| BRK-B | 1 | 126394 | Busines s Description Berkshire Hathaway Inc. ("Berkshire,"  | FAIL (wrong opening) |
| BRK-B | 1A | 21785 | Ris k Factors Berkshire and its subsidiaries (referred to he | FAIL (wrong opening) |
| BRK-B | 7 | 119402 | Management's Discussion and Analysis of Financial Condition  | PASS |
| JPM | 1 | 39127 | Business. Overview JPMorgan Chase & Co. ("JPMorganChase" or  | PASS |
| JPM | 1A | 112508 | Risk Factors. The following discussion sets forth the materi | PASS |
| JPM | 7 | 387 | Management's Discussion and Analysis of Financial Condition  | FAIL (implausibly short) |
| XOM | -- | -- | -- | FETCH FAILED — MissingFilingError: CIK 0002115436 has no 10-K in its recent filings |
| KO | 1 | 55215 | BUSINESS In this report, the terms "The Coca-Cola Company,"  | PASS |
| KO | 1A | 92242 | RISK FACTORS In addition to the other information set forth  | PASS |
| KO | 7 | 109640 | MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION  | PASS |
| JNJ | 1 | 34549 | Business General Johnson & Johnson and its subsidiaries (the | PASS |
| JNJ | 1A | 43333 | Risk factors An investment in the Company's common stock or  | PASS |
| JNJ | 7 | 65887 | Management's discussion and analysis of results of operation | PASS |
| PG | 1 | 14286 | Business. The Procter & Gamble Company (the Company) is a wo | PASS |
| PG | 1A | 37856 | Risk Factors. We discuss our expectations regarding future p | PASS |
| PG | 7 | 91870 | Management's Discussion and Analysis of Financial Condition  | PASS |
| WMT | 1 | 37480 | BUSINESS General Walmart Inc. ("Walmart," the "Company" or " | PASS |
| WMT | 1A | 93173 | RISK FACTORS The risks described below could, in ways we may | PASS |
| WMT | 7 | 58074 | MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION  | PASS |
| CVX | 1 | 77850 | Business General Development of Business Summary Description | PASS |
| CVX | 1A | 34598 | Risk Factors As a global energy company, Chevron is subject  | PASS |
| CVX | 7 | 234 | Management's Discussion and Analysis of Financial Condition  | FAIL (implausibly short) |
| MRK | 1 | 125342 | Business. Merck & Co., Inc. (Merck or the Company) is a glob | PASS |
| MRK | 1A | 77392 | Risk Factors. Summary Risk Factors The Company is subject to | PASS |
| MRK | 7 | 141022 | Management's Discussion and Analysis of Financial Condition  | PASS |
| T | 1 | 31909 | BUSINESS GENERAL AT&T Inc. ("AT&T," "we" or the "Company") i | PASS |
| T | 1A | 42562 | RISK FACTORS In addition to the other information set forth  | PASS |
| T | 7 | 65406 | MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION  | PASS |
| VZ | 1 | 39290 | Business General Verizon Communications Inc. (the Company) i | PASS |
| VZ | 1A | 36412 | Risk Factors The following discussion of "Risk Factors" iden | PASS |
| VZ | 7 | 112450 | Management's Discussion and Analysis of Financial Condition  | PASS |
| PFE | 1 | 91874 | BUSINESS ABOUT PFIZER Pfizer Inc. is a research-based, globa | PASS |
| PFE | 1A | 87394 | RISK FACTORS This section describes the material risks to ou | PASS |
| PFE | 7 | 105433 | MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION  | PASS |
| INTC | 1 | 0 | (absent — not in parsed.items) | FAIL (missing) |
| INTC | 1A | 0 | (absent — not in parsed.items) | FAIL (missing) |
| INTC | 7 | 222 | Management's Discussion and Analysis of Financial Condition  | FAIL (implausibly short) |
| CSCO | 1 | 48051 | Business General Cisco designs and sells a broad range of te | PASS |
| CSCO | 1A | 89575 | Risk Factors Set forth below and elsewhere in this report an | PASS |
| CSCO | 7 | 70860 | Management's Discussion and Analysis of Financial Condition  | PASS |
| BA | 1 | 18371 | Business The Boeing Company, together with its subsidiaries  | PASS |
| BA | 1A | 57235 | Risk Factors An investment in our securities involves risks  | PASS |
| BA | 7 | 89957 | Management's Discussion and Analysis of Financial Condition  | PASS |
| CAT | 1 | 39256 | Business. General Originally organized as Caterpillar Tracto | PASS |
| CAT | 1A | 53445 | Risk Factors. The statements in this section describe the mo | PASS |
| CAT | 7 | 103440 | Management's Discussion and Analysis of Financial Condition  | PASS |
| GE | 1 | 25 | Business 4-7, 9-10, 71-73 | FAIL (implausibly short) |
| GE | 1A | 18 | Risk Factors 24-31 | FAIL (implausibly short) |
| GE | 7 | 90 | Management's Discussion and Analysis of Financial Condition  | FAIL (implausibly short) |
| DIS | 1 | 66757 | Business The Walt Disney Company, together with the subsidia | PASS |
| DIS | 1A | 62841 | Risk Factors For an enterprise as large and complex as the C | PASS |
| DIS | 7 | 78746 | Management's Discussion and Analysis of Financial Condition  | PASS |

For every FAIL row, `ParsedFiling.missing` reported the item as present
(INTC's items 1 and 1A are the sole exception — see below). That is the
defect class this whole task exists to surface: wrong content coming back
silently, with nothing in the return value distinguishing it from a correct
extraction.

## What the failures have in common

Six distinct fetch/parse outcomes, three of them never anticipated by any
existing fixture.

### 1. Running page headers fabricate headings and fragment the section (MSFT — new)

The two known defects going in were (a) a delimiter-less `Item N` line
faking a heading, and (b) the TOC guard discarding genuine content that ends
in digits. Defect (a) did occur, but not via the `<li>` cross-reference the
brief described — via something worse and far more common: **running page
headers**. MSFT's filing repeats a literal, delimiter-less `Item 1` (and
`Item 1A`, and `Item 7`) as a page-break marker on every single printed
page:

```
...open-source alternatives.\n7\nPART I\nItem 1\nWe believe our server products...
...first- and third-party titles.\n8\nPART I\nItem 1\nThe gamer remains at the heart...
```

Each repetition matches `_HEADING` exactly like a genuine section start,
because the parser has no way to distinguish "this is the top of a new
printed page" from "this is a new section." The real, single Item 1 section
is consequently sliced into ~12 fragments of 2–5K characters apiece — one
per page — and the longest-body-wins rule picks whichever page happened to
contain the densest prose (page 9, about Xbox) as "the" section, discarding
the genuine heading-bearing fragment and roughly 85% of the real content.
This hit all three wanted items in the same filing. It is the single most
consequential finding in this run: it affects a top-3 mega-cap, and any
filer whose HTML generator inserts running headers — which is a normal,
unremarkable rendering choice, not an edge case — will trigger it.

### 2. Words split across sibling `<span>` tags corrupt heading text (BRK-B — new)

BRK-B's Item 1 and Item 1A sections are located correctly and captured in
full (126,394 and 21,785 characters, properly bounded) — but the extracted
text opens with `Busines s Description` and `Ris k Factors`. The raw HTML
is:

```html
<span ...>Item 1. Busines</span><span ...>s Description</span>
```

The word "Business" is split across two sibling `<span>` elements at an
arbitrary character boundary — not even at a syllable or kerning boundary.
`<span>` is correctly excluded from `_BLOCK_TAGS` (it's inline formatting,
not a structural break), so the tag-to-whitespace substitution inserts a
space at that boundary, and the space lands inside the word. This is a
genuinely new failure mode: the section-selection logic gets the right
answer, but text fidelity is wrong at the very first word — which is
exactly where a downstream heading check (like this test's, or a
prompt that anchors on the section title) is most likely to look.

### 3. MD&A incorporated by reference to unlabeled prose elsewhere in the document (JPM, CVX — new)

JPM's and CVX's Item 7 sections are correctly located, but they are
genuinely short in the filed document — 387 and 234 characters. Both are
the filer's own pointer stubs:

> "...which appear on pages 165–314." (JPM)
> "...as indicated in the Financial Table of Contents." (CVX)

The real MD&A prose exists later in the same combined HTML document (JPM's
file continues for another ~1,000,000 characters past this point), but it
is never headed literally "Item 7" anywhere — it lives inside an attached
Annual Report / financial-statements exhibit that uses its own heading
conventions. No adjustment to `_HEADING`, the TOC guard, or the boundary
logic can recover this: the text a heading-anchored parser is looking for
simply does not exist at the real content's location. This is a structural
property of how certain large filers assemble a 10-K, not a bug in the
current heuristic.

### 4. Filings that only say "Item N" in a cross-reference index (GE, INTC — new)

GE's entire 10-K contains the literal string `Item 1.` / `Item 1A.` /
`Item 7.` in exactly one place: a `FORM 10-K CROSS REFERENCE INDEX` a few
thousand characters from the end of the document, mapping each item to a
page range:

```
Item 1.
Business
4-7, 9-10, 71-73
Item 1A.
Risk Factors
24-31
```

The real Business/Risk Factors/MD&A prose is elsewhere in the document
under plain topic titles with no "Item" prefix at all, so it never produces
a competing heading match. Compounding this, the TOC guard's shape
assumption — a body that ends in `\s\d{1,4}$`, i.e. a lone trailing number —
does not match a page **range** like `71-73` (the digits are preceded by a
hyphen, not whitespace), so `_looks_like_a_toc_entry` fails to recognize
this cross-reference line as TOC-shaped. It survives as the only, and
therefore winning, candidate. The result is exactly the failure mode this
task exists to catch: `ParsedFiling.missing` reports these items as
present, and the returned text is not obviously garbage at a glance
("Business\n4-7, 9-10, 71-73" reads like a plausible short entry), but it
is pure index debris, not one word of the real section.

INTC has the same root cause — the front-matter index is the only place
`Item 1`/`Item 1A` appear — but there its index lines happen to match the
TOC-guard's shape (their page lists don't end in a bare range the same way),
so they get correctly rejected and no candidate survives at all. That is
the safer failure: `missing` correctly lists `1` and `1A`, loudly, instead
of returning index text as if it were real. INTC's Item 7 is a short stub
of the same kind as JPM/CVX's (case 3 above).

The known defect #2 as originally described — a genuine section discarded
because it *ends* in whitespace-then-digits, e.g. "...most acute in fiscal
2024" — was not observed in this run. What was observed is the mirror
image of the same regex's narrowness: it also fails to *reject* TOC-shaped
content when the trailing shape is a page range rather than a single
number. Same blind spot, opposite direction.

### 5. Ticker resolution to a filer with no 10-K on file (XOM — infrastructure, not parser)

`resolve_cik(client, "XOM")` returns CIK `0002115436`, titled "ExxonMobil
Holdings Corp" in SEC's `company_tickers.json` — evidently a holding-company
entity created by a recent corporate restructuring, distinct from the
long-standing operating filer CIK that has historically carried Exxon's
10-Ks. That CIK has no 10-K in its `submissions` history, so
`latest_filings` correctly raises `MissingFilingError` before `parse_items`
is ever called. This is a fact about SEC's live ticker index changing under
a real-world corporate action, not a parser defect, and not something
`resolve_cik`'s "first ticker match wins" logic (see
`src/era/edgar/filings.py`) has any way to detect from the ticker file
alone.

## Net read

Every wrong result in this run traces to one of two root causes, both
already present in the code's own comments as known limitations, but wider
in practice than the fixtures suggested:

- **The heading matcher (`_HEADING`) cannot tell a genuine section start
  from anything else that happens to say "Item N" at the start of a line** —
  running headers (MSFT), and a front-matter cross-reference index whose
  shape evades the TOC guard (GE), both exploit exactly this.
- **The parser has no fallback when the real content is never headed
  "Item N" at all** — incorporation-by-reference stubs (JPM, CVX) and
  filings that only use "Item N" in an index (GE, INTC's Item 1/1A) are both
  cases where the correct answer for a heading-anchored strategy is "not
  found," and sometimes the code gets there (INTC) and sometimes it doesn't
  (GE, JPM, CVX).

The BRK-B word-splitting failure is a third, independent issue — a text-
fidelity bug in the HTML-to-text conversion, unrelated to heading detection —
that neither of the two pre-registered defects predicted.

No changes were made to `src/era/edgar/sections.py` as part of this task.
Parser changes are a separate, reviewed round.
