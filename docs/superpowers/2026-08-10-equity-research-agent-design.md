# equity-research-agent — Design

**Date:** 2026-08-10
**Status:** Approved
**Portfolio phase:** 1 of `2026-08-09-ai-portfolio-design-v2.md`
**Signal:** LangGraph, agents, RAG, eval engineering, LLM product sensibility

## Purpose

An agent that researches a US-listed company end-to-end from primary sources and
produces a research note in which **every claim is traceable to a filing**. The
repo's differentiator is not the agent — it is that correctness is enforced by
code rather than asserted in a README.

Target roles are AI engineering, ML engineering, and data science. This repo
carries the AI-engineering weight of the portfolio.

## Non-negotiables

- The output never implies investment advice. Enforced by a test, not a footer
  disclaimer (see Evaluation).
- No claim ships without a resolvable citation.
- Reported figures come from XBRL by exact lookup, never from generated text.

## Data sources

**SEC EDGAR only, for fundamentals and filing text.** Free, explicitly permitted
with a declared `User-Agent` carrying a real contact address, throttled at
10 req/s. Chosen over commercial APIs because citations point at primary sources
rather than a vendor's precomputed metrics, and because free-tier rate limits on
commercial APIs would break a live demo.

Four endpoints: company-tickers index (ticker → CIK), submissions feed (filing
history), companyfacts (XBRL), Archives (primary documents).

Scope limit: US issuers only. This is a property of EDGAR and is stated in the
README rather than worked around.

## Output

A fixed-section research note:

1. Business overview
2. Financial health
3. Risk factors (from 10-K Item 1A)
4. Recent developments
5. Valuation context

Fixed structure is a deliberate choice: it is what makes rigorous evaluation
possible, because the set of claims that *should* be present is known in advance.
Target length is ~800 output tokens per section — dense and fully cited beats
long and padded, and padding is where unsupported claims appear.

Explicitly out of scope: real-time data, portfolio management, open-ended Q&A,
multi-company comparison.

## Architecture

Three stages. Cost and accuracy pointed the same direction here — the cheaper
design is also the more faithful one, for reasons recorded under Design
decisions.

### Stage 1 — resolve and ingest (linear)

- `resolve_ticker` — ticker → CIK, failing fast and loudly on unknown symbols.
- `fetch_filings` — latest 10-K and most recent 10-Q. Accession numbers are the
  citation anchor throughout.
- `ingest` — parse filings into sections, chunk and embed **narrative sections
  only**, upsert to pgvector keyed by accession. Idempotent: a second run on the
  same filing is a no-op.
- `fetch_facts` — XBRL companyfacts → a normalized set of financial series, plus
  a compact **facts card** (a few hundred tokens) summarizing headline figures.

**Ingest never runs from the deployed application.** It runs from the CLI. This
avoids Vercel's beta Python Workflow SDK, keeps every request short, and bounds
storage against Neon's free-tier limit.

### Stage 2 — parallel research (fan-out)

One subgraph per section, dispatched concurrently. Each subgraph:

1. Plans its own retrieval queries.
2. Retrieves **its own** chunks — sections do not share a context prefix.
3. Receives the shared facts card, so cross-section reasoning (a risk that
   references a revenue trend) remains possible without carrying a whole filing.
4. Drafts its section with inline citations naming accession and chunk.

Numbers in the financial-health section come from the normalized XBRL series by
exact lookup. **Financial figures are never embedded and never paraphrased.**

### Stage 3 — verify and assemble (loop)

- `verify_citations` — **deterministic Python, not a model.** Every citation must
  resolve to a real accession and an existing chunk; the cited chunk must contain
  the claim's key figures; every numeric claim must match the XBRL value within
  tolerance.
- Failing sections route back to their own subgraph with the verifier's specific
  complaint attached, to a **bounded** retry count. Unbounded reflection loops are
  how agents quietly burn money.
- `assemble` — **plain Python.** The sections are already written; concatenating
  them with coverage notes, limitations, and the run's cost and latency is string
  handling. Doing it deterministically means assembly cannot introduce a claim the
  verifier never checked.

### Coverage philosophy

Applied at three layers, inherited from `stock-analyzer`:

- **Filing parsing.** Item-boundary detection is heuristic — filers vary wildly in
  markup. The parser reports what it found; an unlocatable Item 1A marks the
  risk-factors section unavailable rather than emitting a confident empty one.
- **XBRL normalization.** `us-gaap` tags differ across filers (revenue alone
  appears under several). A mapping layer tries a prioritized list per metric and
  reports which resolved.
- **The note itself** states its own coverage.

A missing input degrades coverage visibly. It never becomes a wrong number.

## Stack

| Concern | Choice |
| --- | --- |
| Orchestration | LangGraph 1.2.10 |
| Model | `gpt-5.6-terra` via `langchain-openai` 1.4.2 |
| Embeddings | `voyage-finance-2` — finance-tuned, 1024 dims, under pgvector's 2000-dim index cap |
| Storage | Neon Postgres + pgvector, provisioned via Vercel Marketplace |
| API | FastAPI, deployed as a Vercel Python Function |
| UI | Next.js, thin client over the same graph |
| CLI | Typer — the primary interface |
| Observability + evals | Opik 2.2.23 (self-hostable, replaces LangSmith) |

**Provider is behind one factory in `graph/`.** `ChatOpenAI` and `ChatAnthropic`
share the `BaseChatModel` interface, so the provider is a config change, not a
rewrite. OpenAI is the default on measured cost (~34% cheaper on this workload
than Claude Sonnet 5, the gap driven by output pricing).

**One model everywhere, initially.** Routing descriptive sections to a cheaper
model would cut cost roughly a third further, but introducing two models before
the evals exist makes quality differences unattributable. Downgrading per section
is a follow-up the eval harness justifies with measurements.

## Evaluation

### Tier 1 — gates every PR

Deterministic, runs against recorded cassettes for both EDGAR and model calls. No
network, no API key, no cost, no flakiness — so it can genuinely gate merges.

- Every citation resolves to a real accession and an existing chunk.
- The cited chunk contains the claim's key figures.
- Every numeric claim matches XBRL within tolerance.
- Every section is populated or explicitly marked unavailable with a reason.
- Output validates against its schema.
- **No recommendation language** — buy/sell/hold/price-target phrasing anywhere in
  the note fails the build. The non-negotiable is enforced, not declared.

**Tier 1 shares its implementation with the runtime verifier.** The check that
gates a merge is the check that gates a section during a run — one implementation,
two callers.

### Tier 2 — judged, on merge

Opik datasets and experiments over a golden set of ~10 tickers chosen for
awkwardness as much as coverage: a clean large-cap, one with messy item
boundaries, one with unusual XBRL tagging. Metrics: `Hallucination`,
`AnswerRelevance`, `ContextPrecision`, `ContextRecall`, plus a `GEval` metric for
section quality. Judge model is `gpt-5.6-luna`, configured explicitly via Opik's
LiteLLM string rather than left on the library default. The "one model" rule
above governs *drafting*, where mixing would make quality differences
unattributable; judging is a separate concern and uses the cheap model.

**Runs on merge, not nightly.** Output changes when the graph or prompts change,
not when the date does. Results append to a tracked file and render as a
quality-over-time table in the README.

**Judged evals never gate a merge.** They are nondeterministic and cost money, and
a flaky gate is one people learn to ignore.

### Live smoke

One ticker against live EDGAR on merges to main — catches what cassettes
structurally cannot, namely EDGAR changing its response shape.

## Deployment

- Next.js UI and FastAPI backend on Vercel. Fluid Compute is default, giving a
  300s Hobby ceiling — comfortable for a research run once ingest is out of the
  request path.
- Neon free tier: 0.5 GB storage, autosuspend after 5 minutes idle (not
  disableable, ~0.5–1s cold start). **This bounds the indexed universe to roughly
  20–30 tickers** — a stated design constraint, surfaced in the README.
- Golden-set tickers ship pre-indexed, so a first visit answers immediately.
- Research progress streams per node, which is also what makes the graph's
  parallelism visible — the reason the UI exists at all.
- **On-demand generation is open**: any visitor may generate a fresh report for an
  indexed ticker. Cost is ~$0.13 per trigger with no cap. Accepted deliberately for
  demo quality. The runs table records per-run cost, so a spend ceiling is a small
  addition if the URL ever attracts unwanted attention.

## Repo structure

```
equity-research-agent/
├── src/era/
│   ├── edgar/          client, throttling, cache, filing parsing, XBRL normalization
│   ├── index/          chunking, embeddings, pgvector
│   ├── graph/          state, nodes, section subgraphs, model factory
│   ├── verify/         deterministic verifier (shared with Tier 1 evals)
│   ├── report/         Pydantic schema, Python assembly
│   ├── observability/  Opik wiring, cost and latency capture
│   └── cli.py          Typer entry point
├── api/                Vercel Python Function (FastAPI)
├── web/                Next.js UI
├── evals/
│   ├── tier1/          pytest + cassettes
│   ├── tier2/          Opik datasets and experiments
│   └── cassettes/
├── tests/
└── .github/workflows/  ci.yml, smoke.yml, evals.yml
```

## Unit economics

~$0.13 per research run: six drafting calls (five sections plus one retry) at
~6,300 input and ~800 output tokens each, with deterministic assembly. EDGAR is
free; embedding a 20–30 filing corpus sits inside Voyage's 50M free tokens; Neon
and Opik free tiers cover storage and observability.

Projected total to build and ship: **~$68** — roughly $40 of development
iteration (~300 runs) and ~$28 of on-merge evals (~20 merges × 10 tickers, plus
judge calls on the cheap model at ~$0.07 per cycle). Steady state after launch is
near zero for stored reports, plus ~$0.13 per visitor-triggered generation.

Recorded because it drove the architecture: the original design cost $0.23/run,
of which 53% was output tokens and 35% was a shared 25k-token context prefix plus
the warm-up call needed to prime its cache. Removing the shared prefix removed two
nodes, removed a dependency on unverified cross-provider caching behaviour under
concurrency, and improved expected faithfulness — precision beats volume for
grounded generation. Cost analysis found a design flaw, not a pricing problem.

## Design decisions worth defending in an interview

1. **The verifier is code, not a judge.** Citation resolution and numeric fidelity
   are decidable. Using a model to check them would trade a guarantee for a
   probability.
2. **Numbers never pass through embeddings.** Structurally eliminates the most
   common failure mode in financial RAG demos.
3. **Per-section retrieval over a shared prefix.** Feeding each section 25k tokens
   of mostly-irrelevant context invites lost-in-the-middle degradation and drift
   toward material the section shouldn't cite.
4. **Assembly is deterministic.** A generative assembler could introduce a claim
   that no verifier ever saw.
5. **One model until the evals say otherwise.** Cost routing is a measured
   decision, not a guess.
6. **Tier 1 is deterministic and Tier 2 is not, and only Tier 1 gates.**

## Risks and spikes

| Risk | Mitigation |
| --- | --- |
| **Opik tracing under parallel fan-out is undocumented**, with a known async context-propagation gotcha requiring manual span passing, plus an open issue on non-UUID thread IDs | **Spike before building on it.** Parallelism is the architectural centrepiece. Fallback: sequential fan-out, or manual span propagation via `extract_current_langgraph_span_data`. |
| 10-K item-boundary parsing is heuristic and filer-dependent | Coverage reporting; parser tested against filings from several filers with differing markup |
| Neon 0.5 GB cap and 5-minute autosuspend | Bounded ticker universe; pre-indexed golden set; cold start accepted and documented |
| Uncapped on-demand generation | Accepted. Runs table makes a spend ceiling a small later addition |
| `voyage-finance-2` is a 2024-generation model | Evaluate `voyage-context-4` as an alternative during ingest work; `rag-evals-lab` settles it properly later |

## Success criteria

- Deployed and publicly usable, answering for the pre-indexed universe.
- Tier 1 evals gate CI and pass.
- README carries a real generated note, the quality-over-time table, the
  architecture diagram, and measured cost and latency per run.
- Every design decision above is one Berkay can explain unprompted.

## Next step

`writing-plans` → implementation plan.
