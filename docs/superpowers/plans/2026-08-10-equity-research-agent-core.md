# equity-research-agent — Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CLI-deliverable core of an agent that turns a US stock ticker into a research note where every claim is machine-verified against SEC filings.

**Architecture:** A LangGraph pipeline in three stages — linear ingest from EDGAR, parallel per-section research subgraphs, then a deterministic verify-and-retry loop. Sections emit *structured claims*, not prose, so verification is a data operation rather than regex over markdown. The same verifier module runs at runtime and as the CI eval gate.

**Tech Stack:** Python 3.12, uv, LangGraph 1.2.10, langchain-openai 1.4.2, Pydantic v2, Neon Postgres + pgvector, Voyage embeddings, Opik 2.2.23, Typer, pytest, ruff, mypy strict.

**Source spec:** `docs/superpowers/specs/2026-08-10-equity-research-agent-design.md`

**Scope:** This plan ends with a working CLI and green CI. Deployment (FastAPI, Next.js, Vercel, Neon hosting, stored reports) is Plan 2.

## Global Constraints

- Repo location: `~/Desktop/My Road/yeni/genai/equity-research-agent`. Personal git identity applies automatically via gitconfig `includeIf`; remote uses the `github-personal` SSH alias.
- Package manager is **uv**. Never `pip install`.
- Python **3.12**. `mypy --strict` on `src/`. `ruff` for lint and format.
- Conventional commits. **No Claude attribution or co-author trailers in any commit.**
- MIT license.
- Every module below `cli.py` returns plain data — no `print`, no formatting decisions. The CLI is the only layer that renders.
- **Financial figures are never embedded and never paraphrased.** They come from XBRL by exact lookup.
- **No claim ships without a resolvable citation.**
- The output never implies investment advice. Enforced by `verify.checks.no_recommendation_language`, not a disclaimer.
- Model: `gpt-5.6-terra` for all drafting. Judge: `gpt-5.6-luna`. Embeddings: `voyage-finance-2` (1024 dims).
- Tests never touch the network or a live database. External surfaces sit behind Protocols with fakes in tests.
- `.env` is gitignored; `.env.example` documents every variable.
- **Readability is a hard requirement, not a preference.** Berkay must be able to
  read any file in this repo and understand it without tracing calls. That means:
  names that say what the thing does in domain terms (`resolve_cik`, not
  `get_data`); functions short enough to hold in your head; no clever one-liners
  where three plain lines are clearer; no comments restating the code, but a short
  comment wherever a non-obvious constraint drove the design (SEC rate limits,
  why numbers bypass embeddings, why a retry is bounded). Prefer an explicit
  `if` over a nested comprehension. A reviewer flagging "this is hard to follow"
  is a valid Important finding, not a style opinion.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/era/config.py` | Settings from env via pydantic-settings |
| `src/era/edgar/client.py` | Throttled, cached HTTP access to SEC endpoints |
| `src/era/edgar/filings.py` | Ticker → CIK, filing metadata retrieval |
| `src/era/edgar/sections.py` | 10-K item-boundary parsing with coverage reporting |
| `src/era/edgar/xbrl.py` | companyfacts → normalized metrics + facts card |
| `src/era/index/chunking.py` | Section-aware chunking with metadata |
| `src/era/index/embeddings.py` | Voyage embedding client behind a Protocol |
| `src/era/index/store.py` | pgvector upsert/query behind a Protocol |
| `src/era/report/schema.py` | Pydantic models — the contract every other module shares |
| `src/era/verify/checks.py` | Deterministic verification (runtime + eval gate) |
| `src/era/graph/models.py` | Provider factory |
| `src/era/graph/state.py` | Graph state TypedDict |
| `src/era/graph/section.py` | One section's research subgraph |
| `src/era/graph/build.py` | StateGraph assembly, fan-out, retry loop |
| `src/era/report/assemble.py` | Structured claims → markdown, in plain Python |
| `src/era/observability/tracing.py` | Opik wiring, cost/latency capture |
| `src/era/cli.py` | Typer entry point |

---

### Task 1: Repo scaffold and CI

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`, `README.md`, `src/era/__init__.py`, `tests/test_smoke.py`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing
- Produces: a `uv` project where `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` all pass

- [ ] **Step 1: Create the project and directory layout**

```bash
cd ~/Desktop/My\ Road/yeni/genai
mkdir -p equity-research-agent && cd equity-research-agent
git init
mkdir -p src/era/{edgar,index,graph,verify,report,observability} tests evals/{tier1,tier2,cassettes} .github/workflows
touch src/era/__init__.py
for d in edgar index graph verify report observability; do touch "src/era/$d/__init__.py"; done
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "era"
version = "0.1.0"
description = "Agent that writes cited equity research notes from SEC filings"
requires-python = ">=3.12,<3.13"
dependencies = [
    "langgraph==1.2.10",
    "langchain-openai==1.4.2",
    "langchain-core>=1.5.2,<2.0.0",
    "pydantic>=2.7.4,<3.0.0",
    "pydantic-settings>=2.4.0",
    "httpx>=0.27.0",
    "typer>=0.12.0",
    "psycopg[binary]>=3.2.0",
    "pgvector>=0.3.0",
    "voyageai>=0.3.0",
    "opik==2.2.23",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "respx>=0.21.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
]

[project.scripts]
era = "era.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/era"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
strict = true
files = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests", "evals/tier1"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Write `.gitignore` and `.env.example`**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
.pytest_cache/
.mypy_cache/
.ruff_cache/
.cache/
```

`.env.example`:
```
# SEC requires a real contact address in the User-Agent or it will block you
EDGAR_USER_AGENT="Berkay Koklu kokluberkay@gmail.com"
OPENAI_API_KEY=
VOYAGE_API_KEY=
DATABASE_URL=
OPIK_API_KEY=
OPIK_WORKSPACE=
```

- [ ] **Step 4: Write the smoke test**

`tests/test_smoke.py`:
```python
def test_package_imports() -> None:
    import era

    assert era is not None
```

- [ ] **Step 5: Install and run**

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run mypy src
```
Expected: all three pass.

- [ ] **Step 6: Write the CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv sync --all-groups --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest -q
```

- [ ] **Step 7: Add MIT LICENSE and a placeholder README**

`README.md` starts as a single line: `# equity-research-agent` — Task 17 writes the real one.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold uv project with ruff, mypy strict, pytest, CI"
```

---

### Task 2: Opik parallel fan-out tracing spike

**This task gates Task 12.** Opik's docs do not confirm behaviour under LangGraph parallel fan-out, there is a documented async context-propagation gotcha, and an open issue about non-UUID thread IDs. Parallelism is the architectural centrepiece, so this gets answered before anything is built on it.

**Files:**
- Create: `docs/spikes/2026-08-10-opik-parallel-fanout.md`, `spikes/opik_fanout.py`

**Interfaces:**
- Consumes: nothing
- Produces: a written finding that Task 12 and Task 16 depend on — either "parallel fan-out traces correctly" or a documented fallback

- [ ] **Step 1: Write a minimal fan-out graph**

`spikes/opik_fanout.py`:
```python
"""Spike: does Opik trace LangGraph parallel fan-out correctly?

Run: uv run python spikes/opik_fanout.py
Requires OPIK_API_KEY and OPIK_WORKSPACE in the environment.
"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from opik.integrations.langchain import OpikTracer


class State(TypedDict):
    results: Annotated[list[str], operator.add]


def make_branch(name: str):
    def branch(state: State) -> State:
        time.sleep(0.5)
        return {"results": [name]}

    return branch


def build():
    g = StateGraph(State)
    for n in ["alpha", "beta", "gamma", "delta", "epsilon"]:
        g.add_node(n, make_branch(n))
        g.add_edge(START, n)
        g.add_edge(n, END)
    return g.compile()


if __name__ == "__main__":
    app = build()
    tracer = OpikTracer(project_name="era-spike", graph=app.get_graph(xray=True))
    out = app.invoke({"results": []}, config={"callbacks": [tracer]})
    tracer.flush()
    print("branches returned:", sorted(out["results"]))
```

- [ ] **Step 2: Run it and inspect the Opik UI**

```bash
uv run python spikes/opik_fanout.py
```

Check in the Opik trace view and record the answers:
1. Do all five branch nodes appear as spans?
2. Are they nested under one trace, or split across traces?
3. Does the agent graph render?
4. Does anything break if `thread_id` is a non-UUID string?

- [ ] **Step 3: Repeat with `ainvoke`**

Change `app.invoke` to `await app.ainvoke` inside an `asyncio.run` wrapper and re-run. The documented context-propagation gotcha applies to async specifically — record whether spans are still nested.

- [ ] **Step 4: Write the finding**

`docs/spikes/2026-08-10-opik-parallel-fanout.md` records: what worked, what didn't, and the decision. Use this template and fill it with observed results, not predictions:

```markdown
# Spike: Opik tracing under LangGraph parallel fan-out

**Date:** 2026-08-10
**Question:** Do parallel fan-out nodes trace as nested spans under one trace?

## Sync (`invoke`)
- All five branches appear as spans: YES / NO
- Nested under one trace: YES / NO
- Agent graph renders: YES / NO

## Async (`ainvoke`)
- All five branches appear as spans: YES / NO
- Nested under one trace: YES / NO

## Non-UUID thread_id
- Works: YES / NO

## Decision
[One of: (a) proceed with async parallel fan-out as designed;
(b) proceed with sync fan-out; (c) proceed with async plus manual span
propagation via extract_current_langgraph_span_data.]
```

- [ ] **Step 5: Commit**

```bash
git add spikes/ docs/spikes/
git commit -m "chore: spike Opik tracing under LangGraph parallel fan-out"
```

---

### Task 3: Report schema

The contract every later task shares. **Sections emit structured claims, not prose** — this is what makes verification a data operation instead of regex over markdown.

**Files:**
- Create: `src/era/report/schema.py`, `tests/report/test_schema.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ChunkRef`, `FactRef`, `Claim`, `SectionName`, `Section`, `ResearchNote`, `Coverage`

- [ ] **Step 1: Write the failing test**

`tests/report/test_schema.py`:
```python
import pytest
from pydantic import ValidationError

from era.report.schema import Claim, ChunkRef, FactRef, Section, SectionName


def test_claim_requires_at_least_one_source() -> None:
    with pytest.raises(ValidationError):
        Claim(text="Revenue grew.", chunks=[], facts=[])


def test_claim_accepts_a_chunk_citation() -> None:
    claim = Claim(
        text="The company describes itself as a designer of smartphones.",
        chunks=[ChunkRef(accession="0000320193-24-000123", chunk_id=42)],
        facts=[],
    )
    assert claim.chunks[0].chunk_id == 42


def test_unavailable_section_carries_a_reason() -> None:
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=[],
        available=False,
        unavailable_reason="Item 1A boundary not found in filing",
    )
    assert section.available is False


def test_available_section_rejects_empty_claims() -> None:
    with pytest.raises(ValidationError):
        Section(name=SectionName.RISK_FACTORS, claims=[], available=True)


def test_fact_ref_carries_the_exact_value() -> None:
    fact = FactRef(
        tag="Revenues", fiscal_period="FY2024", value=391_035_000_000.0,
        accession="0000320193-24-000123",
    )
    assert fact.value == 391_035_000_000.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/report/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'era.report.schema'`

- [ ] **Step 3: Write the schema**

`src/era/report/schema.py`:
```python
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SectionName(StrEnum):
    BUSINESS_OVERVIEW = "business_overview"
    FINANCIAL_HEALTH = "financial_health"
    RISK_FACTORS = "risk_factors"
    RECENT_DEVELOPMENTS = "recent_developments"
    VALUATION_CONTEXT = "valuation_context"


class ChunkRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    accession: str
    chunk_id: int


class FactRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    tag: str
    fiscal_period: str
    value: float
    accession: str


class Claim(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    chunks: tuple[ChunkRef, ...] = ()
    facts: tuple[FactRef, ...] = ()

    @model_validator(mode="after")
    def require_a_source(self) -> "Claim":
        if not self.chunks and not self.facts:
            raise ValueError("every claim must cite at least one chunk or fact")
        return self


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: SectionName
    claims: tuple[Claim, ...] = ()
    available: bool = True
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def check_availability(self) -> "Section":
        if self.available and not self.claims:
            raise ValueError("an available section must contain at least one claim")
        if not self.available and not self.unavailable_reason:
            raise ValueError("an unavailable section must state why")
        return self


class Coverage(BaseModel):
    model_config = ConfigDict(frozen=True)

    sections_available: int
    sections_total: int
    metrics_resolved: tuple[str, ...] = ()
    metrics_missing: tuple[str, ...] = ()


class ResearchNote(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    cik: str
    sections: tuple[Section, ...]
    coverage: Coverage
    accessions: tuple[str, ...] = Field(default=())
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/report/test_schema.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/era/report/schema.py tests/report/test_schema.py
git commit -m "feat: add research note schema with claim-level citations"
```

---

### Task 4: EDGAR HTTP client

**Files:**
- Create: `src/era/config.py`, `src/era/edgar/client.py`, `tests/edgar/test_client.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Settings`, `EdgarClient(user_agent, cache_dir)` with `get_json(url) -> dict`, `get_text(url) -> str`

- [ ] **Step 1: Write the failing test**

`tests/edgar/test_client.py`:
```python
import httpx
import pytest
import respx

from era.edgar.client import EdgarClient, MissingUserAgentError


def test_rejects_a_user_agent_without_contact_details() -> None:
    with pytest.raises(MissingUserAgentError):
        EdgarClient(user_agent="era-bot", cache_dir=None)


@respx.mock
def test_sends_the_declared_user_agent() -> None:
    route = respx.get("https://data.sec.gov/thing.json").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = EdgarClient(user_agent="Berkay Koklu kokluberkay@gmail.com", cache_dir=None)

    assert client.get_json("https://data.sec.gov/thing.json") == {"ok": True}
    assert route.calls.last.request.headers["user-agent"] == (
        "Berkay Koklu kokluberkay@gmail.com"
    )


@respx.mock
def test_caches_responses_on_disk(tmp_path) -> None:
    route = respx.get("https://data.sec.gov/thing.json").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = EdgarClient(
        user_agent="Berkay Koklu kokluberkay@gmail.com", cache_dir=tmp_path
    )

    client.get_json("https://data.sec.gov/thing.json")
    client.get_json("https://data.sec.gov/thing.json")

    assert route.call_count == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/edgar/test_client.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write config and client**

`src/era/config.py`:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    edgar_user_agent: str
    openai_api_key: str = ""
    voyage_api_key: str = ""
    database_url: str = ""
    opik_api_key: str = ""
    opik_workspace: str = ""
```

`src/era/edgar/client.py`:
```python
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

MIN_INTERVAL_SECONDS = 0.11  # SEC allows 10 req/s; stay just under


class MissingUserAgentError(ValueError):
    """SEC blocks requests whose User-Agent lacks contact details."""


class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path | None) -> None:
        if "@" not in user_agent:
            raise MissingUserAgentError(
                "SEC requires a User-Agent containing a contact email address"
            )
        self._client = httpx.Client(headers={"User-Agent": user_agent}, timeout=30.0)
        self._cache_dir = cache_dir
        self._last_request_at = 0.0

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self._cache_dir / f"{digest}.cache"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_INTERVAL_SECONDS:
            time.sleep(MIN_INTERVAL_SECONDS - elapsed)
        self._last_request_at = time.monotonic()

    def get_text(self, url: str) -> str:
        path = self._cache_path(url)
        if path is not None and path.exists():
            return path.read_text()
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        if path is not None:
            path.write_text(response.text)
        return response.text

    def get_json(self, url: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.get_text(url))
        return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/edgar/test_client.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/era/config.py src/era/edgar/client.py tests/edgar/test_client.py
git commit -m "feat: add throttled, cached EDGAR client with User-Agent enforcement"
```

---

### Task 5: Ticker resolution and filing retrieval

**Files:**
- Create: `src/era/edgar/filings.py`, `tests/edgar/test_filings.py`

**Interfaces:**
- Consumes: `EdgarClient` from Task 4
- Produces: `resolve_cik(client, ticker) -> str`, `UnknownTickerError`, `Filing(accession, form, filing_date, primary_document_url)`, `latest_filings(client, cik) -> list[Filing]`

- [ ] **Step 1: Write the failing test**

`tests/edgar/test_filings.py`:
```python
import httpx
import pytest
import respx

from era.edgar.client import EdgarClient
from era.edgar.filings import UnknownTickerError, latest_filings, resolve_cik

UA = "Berkay Koklu kokluberkay@gmail.com"

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

SUBMISSIONS = {
    "cik": "0000320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000123", "0000320193-24-000070"],
            "form": ["10-K", "10-Q"],
            "filingDate": ["2024-11-01", "2024-08-02"],
            "primaryDocument": ["aapl-20240928.htm", "aapl-20240629.htm"],
        }
    },
}


@respx.mock
def test_resolves_a_ticker_to_a_zero_padded_cik() -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    assert resolve_cik(client, "aapl") == "0000320193"


@respx.mock
def test_unknown_ticker_fails_loudly() -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    with pytest.raises(UnknownTickerError, match="ZZZZ"):
        resolve_cik(client, "ZZZZ")


@respx.mock
def test_returns_the_latest_10k_and_10q_with_document_urls() -> None:
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=SUBMISSIONS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    filings = latest_filings(client, "0000320193")

    assert [f.form for f in filings] == ["10-K", "10-Q"]
    assert filings[0].accession == "0000320193-24-000123"
    assert filings[0].primary_document_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019324000123/aapl-20240928.htm"
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/edgar/test_filings.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the implementation**

`src/era/edgar/filings.py`:
```python
from dataclasses import dataclass

from era.edgar.client import EdgarClient

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
WANTED_FORMS = ("10-K", "10-Q")


class UnknownTickerError(LookupError):
    """The ticker is not present in SEC's company index."""


@dataclass(frozen=True)
class Filing:
    accession: str
    form: str
    filing_date: str
    primary_document_url: str


def resolve_cik(client: EdgarClient, ticker: str) -> str:
    index = client.get_json(TICKERS_URL)
    wanted = ticker.upper()
    for entry in index.values():
        if entry["ticker"].upper() == wanted:
            return str(entry["cik_str"]).zfill(10)
    raise UnknownTickerError(f"{wanted} is not a US-listed issuer in SEC's index")


def latest_filings(client: EdgarClient, cik: str) -> list[Filing]:
    data = client.get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = data["filings"]["recent"]
    bare_cik = cik.lstrip("0")

    found: list[Filing] = []
    seen: set[str] = set()
    for accession, form, date, document in zip(
        recent["accessionNumber"],
        recent["form"],
        recent["filingDate"],
        recent["primaryDocument"],
        strict=True,
    ):
        if form not in WANTED_FORMS or form in seen:
            continue
        seen.add(form)
        folder = accession.replace("-", "")
        found.append(
            Filing(
                accession=accession,
                form=form,
                filing_date=date,
                primary_document_url=(
                    f"https://www.sec.gov/Archives/edgar/data/{bare_cik}/"
                    f"{folder}/{document}"
                ),
            )
        )
        if len(seen) == len(WANTED_FORMS):
            break
    return found
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/edgar/test_filings.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/era/edgar/filings.py tests/edgar/test_filings.py
git commit -m "feat: resolve tickers to CIKs and locate latest 10-K and 10-Q"
```

---

### Task 6: Filing section parsing with coverage reporting

Item-boundary detection is heuristic — filers vary wildly in markup. **The parser reports what it found.** An unlocatable Item 1A marks that section unavailable rather than producing a confident empty one.

**Files:**
- Create: `src/era/edgar/sections.py`, `tests/edgar/test_sections.py`, `tests/fixtures/filing_simple.html`, `tests/fixtures/filing_no_item_1a.html`, `tests/fixtures/filing_with_toc.html`

**Interfaces:**
- Consumes: nothing (operates on filing HTML text)
- Produces: `ParsedFiling(items: dict[str, str], missing: tuple[str, ...])`, `parse_items(html) -> ParsedFiling`

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/filing_simple.html` — a minimal filing with all three items:
```html
<html><body>
<p>Item 1. Business</p>
<p>We design and sell smartphones, computers and wearables worldwide.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business is subject to supply chain concentration in a single region.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Net sales increased in fiscal 2024 driven by services growth.</p>
<p>Item 8. Financial Statements</p>
</body></html>
```

`tests/fixtures/filing_with_toc.html` — a filing that lists its items in a table
of contents before the real sections, the way every actual 10-K does:
```html
<html><body>
<p>Table of Contents</p>
<p>Item 1. Business ..... 3</p>
<p>Item 1A. Risk Factors ..... 15</p>
<p>Item 7. Management's Discussion and Analysis ..... 42</p>
<p>Item 8. Financial Statements ..... 60</p>
<p>Item 1. Business</p>
<p>We design and sell smartphones, computers and wearables worldwide, and we
operate retail stores across many countries.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business is subject to supply chain concentration in a single region,
which could materially affect our results of operations.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Net sales increased in fiscal 2024 driven by services growth across every
geographic segment.</p>
<p>Item 8. Financial Statements</p>
</body></html>
```

`tests/fixtures/filing_no_item_1a.html` — same, but with the Item 1A heading absent:
```html
<html><body>
<p>Item 1. Business</p>
<p>We operate retail stores across several regions.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Comparable sales declined modestly year over year.</p>
<p>Item 8. Financial Statements</p>
</body></html>
```

- [ ] **Step 2: Write the failing test**

`tests/edgar/test_sections.py`:
```python
from pathlib import Path

from era.edgar.sections import parse_items

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_extracts_each_item_body() -> None:
    parsed = parse_items((FIXTURES / "filing_simple.html").read_text())

    assert "smartphones" in parsed.items["1"]
    assert "supply chain concentration" in parsed.items["1A"]
    assert "services growth" in parsed.items["7"]
    assert parsed.missing == ()


def test_item_bodies_stop_at_the_next_heading() -> None:
    parsed = parse_items((FIXTURES / "filing_simple.html").read_text())

    assert "supply chain" not in parsed.items["1"]


def test_reports_a_missing_item_rather_than_inventing_one() -> None:
    parsed = parse_items((FIXTURES / "filing_no_item_1a.html").read_text())

    assert "1A" not in parsed.items
    assert parsed.missing == ("1A",)


def test_prefers_the_real_section_over_the_table_of_contents_entry() -> None:
    # Every real 10-K lists its items twice — once in the TOC, once as content.
    # Taking the first match would capture "Risk Factors ..... 15" instead of
    # the risk factors themselves, and no other test in this file would notice.
    parsed = parse_items((FIXTURES / "filing_with_toc.html").read_text())

    assert "supply chain concentration" in parsed.items["1A"]
    assert "smartphones" in parsed.items["1"]
    assert "services growth" in parsed.items["7"]
    assert parsed.missing == ()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/edgar/test_sections.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write the parser**

`src/era/edgar/sections.py`:
```python
import re
from dataclasses import dataclass

WANTED_ITEMS = ("1", "1A", "7")

_TAGS = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_HEADING = re.compile(
    r"item\s+(?P<item>\d{1,2}[A-Z]?)\s*[.:\-–]", flags=re.IGNORECASE
)


@dataclass(frozen=True)
class ParsedFiling:
    items: dict[str, str]
    missing: tuple[str, ...]


def _to_text(html: str) -> str:
    without_tags = _TAGS.sub(" ", html)
    return _WHITESPACE.sub(" ", without_tags).strip()


def parse_items(html: str) -> ParsedFiling:
    """Split a filing into the items we care about.

    Every 10-K lists its items twice: once in the table of contents and once as
    the actual sections. TOC entries match the same heading pattern, so the
    first match for an item is usually a page-number line rather than content.
    Keeping the longest body found for each item picks the real section, because
    a TOC line is a few words and a real item is paragraphs.
    """
    text = _to_text(html)
    matches = list(_HEADING.finditer(text))

    bodies: dict[str, str] = {}
    for index, match in enumerate(matches):
        item = match.group("item").upper()
        if item not in WANTED_ITEMS:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if len(body) > len(bodies.get(item, "")):
            bodies[item] = body

    missing = tuple(item for item in WANTED_ITEMS if item not in bodies)
    return ParsedFiling(items=bodies, missing=missing)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/edgar/test_sections.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/era/edgar/sections.py tests/edgar/test_sections.py tests/fixtures/
git commit -m "feat: parse 10-K items with explicit coverage reporting"
```

---

### Task 7: XBRL normalization and facts card

`us-gaap` tags differ across filers — revenue alone appears under several. A prioritized mapping per metric, with an explicit report of what resolved.

**Files:**
- Create: `src/era/edgar/xbrl.py`, `tests/edgar/test_xbrl.py`

**Interfaces:**
- Consumes: `EdgarClient`
- Produces: `MetricValue(tag, fiscal_period, value, accession)`, `NormalizedFacts(metrics: dict[str, MetricValue], missing: tuple[str, ...])`, `normalize_facts(companyfacts) -> NormalizedFacts`, `facts_card(facts) -> str`

- [ ] **Step 1: Write the failing test**

`tests/edgar/test_xbrl.py`:
```python
from era.edgar.xbrl import facts_card, normalize_facts

COMPANYFACTS = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        {
                            "fy": 2024, "fp": "FY", "form": "10-K",
                            "val": 391035000000,
                            "accn": "0000320193-24-000123",
                        }
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "fy": 2024, "fp": "FY", "form": "10-K",
                            "val": 93736000000,
                            "accn": "0000320193-24-000123",
                        }
                    ]
                }
            },
        }
    }
}


def test_resolves_revenue_from_an_alternate_tag() -> None:
    facts = normalize_facts(COMPANYFACTS)

    assert facts.metrics["revenue"].value == 391_035_000_000
    assert facts.metrics["revenue"].tag == (
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    )


def test_reports_metrics_it_could_not_resolve() -> None:
    facts = normalize_facts(COMPANYFACTS)

    assert "total_debt" in facts.missing
    assert "revenue" not in facts.missing


def test_facts_card_is_compact_and_names_its_period() -> None:
    card = facts_card(normalize_facts(COMPANYFACTS))

    assert "revenue" in card
    assert "391,035,000,000" in card
    assert "FY2024" in card
    assert len(card) < 1200
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/edgar/test_xbrl.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the normalizer**

`src/era/edgar/xbrl.py`:
```python
from dataclasses import dataclass
from typing import Any

TAG_PRIORITY: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_income": ("OperatingIncomeLoss",),
    "total_assets": ("Assets",),
    "total_debt": ("DebtLongtermAndShorttermCombinedAmount", "LongTermDebtNoncurrent"),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
    ),
}


@dataclass(frozen=True)
class MetricValue:
    tag: str
    fiscal_period: str
    value: float
    accession: str


@dataclass(frozen=True)
class NormalizedFacts:
    metrics: dict[str, MetricValue]
    missing: tuple[str, ...]


def _latest_annual(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    annual = [e for e in entries if e.get("fp") == "FY" and e.get("form") == "10-K"]
    if not annual:
        return None
    return max(annual, key=lambda e: e["fy"])


def normalize_facts(companyfacts: dict[str, Any]) -> NormalizedFacts:
    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    metrics: dict[str, MetricValue] = {}

    for metric, tags in TAG_PRIORITY.items():
        for tag in tags:
            units = us_gaap.get(tag, {}).get("units", {}).get("USD")
            if not units:
                continue
            entry = _latest_annual(units)
            if entry is None:
                continue
            metrics[metric] = MetricValue(
                tag=tag,
                fiscal_period=f"FY{entry['fy']}",
                value=float(entry["val"]),
                accession=entry["accn"],
            )
            break

    missing = tuple(m for m in TAG_PRIORITY if m not in metrics)
    return NormalizedFacts(metrics=metrics, missing=missing)


def facts_card(facts: NormalizedFacts) -> str:
    """A few hundred tokens of headline figures, shared with every section."""
    lines = [
        f"{name}: {value.value:,.0f} USD ({value.fiscal_period}, tag {value.tag})"
        for name, value in sorted(facts.metrics.items())
    ]
    if facts.missing:
        lines.append(f"unavailable: {', '.join(sorted(facts.missing))}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/edgar/test_xbrl.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/era/edgar/xbrl.py tests/edgar/test_xbrl.py
git commit -m "feat: normalize XBRL tags with priority mapping and coverage"
```

---

### Task 8: Section-aware chunking

**Files:**
- Create: `src/era/index/chunking.py`, `tests/index/test_chunking.py`

**Interfaces:**
- Consumes: `ParsedFiling` from Task 6
- Produces: `Chunk(chunk_id, accession, item, text)`, `chunk_items(accession, items, max_chars, overlap) -> list[Chunk]`

- [ ] **Step 1: Write the failing test**

`tests/index/test_chunking.py`:
```python
from era.index.chunking import chunk_items


def test_chunk_ids_are_unique_and_stable() -> None:
    items = {"1": "alpha " * 500, "1A": "beta " * 500}

    chunks = chunk_items("0000320193-24-000123", items, max_chars=600, overlap=50)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_each_chunk_records_the_item_it_came_from() -> None:
    chunks = chunk_items(
        "0000320193-24-000123", {"1A": "risk " * 400}, max_chars=600, overlap=50
    )

    assert {c.item for c in chunks} == {"1A"}
    assert all(c.accession == "0000320193-24-000123" for c in chunks)


def test_chunks_never_span_two_items() -> None:
    chunks = chunk_items(
        "acc", {"1": "alpha " * 200, "1A": "beta " * 200}, max_chars=10_000, overlap=0
    )

    for chunk in chunks:
        assert not ("alpha" in chunk.text and "beta" in chunk.text)


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_items("acc", {"1": "x" * 1000}, max_chars=400, overlap=100)

    assert len(chunks) > 1
    assert len(chunks[0].text) == 400
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/index/test_chunking.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the chunker**

`src/era/index/chunking.py`:
```python
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 4000
DEFAULT_OVERLAP = 400


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    accession: str
    item: str
    text: str


def chunk_items(
    accession: str,
    items: dict[str, str],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk each item separately so a chunk never spans two items."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[Chunk] = []
    next_id = 0
    step = max_chars - overlap

    for item in sorted(items):
        text = items[item]
        for start in range(0, len(text), step):
            window = text[start : start + max_chars]
            if not window.strip():
                continue
            chunks.append(
                Chunk(chunk_id=next_id, accession=accession, item=item, text=window)
            )
            next_id += 1
            if start + max_chars >= len(text):
                break
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/index/test_chunking.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/era/index/chunking.py tests/index/test_chunking.py
git commit -m "feat: add section-aware chunking with per-item boundaries"
```

---

### Task 9: Embeddings and vector store behind Protocols

Both external surfaces get a Protocol and an in-memory fake, so the whole test suite runs with no network and no database — the same pattern that keeps `stock-analyzer`'s suite deterministic.

**Files:**
- Create: `src/era/index/embeddings.py`, `src/era/index/store.py`, `tests/index/test_store.py`, `tests/fakes.py`

**Interfaces:**
- Consumes: `Chunk` from Task 8
- Produces: `Embedder` Protocol with `embed(texts) -> list[list[float]]`; `VoyageEmbedder`; `ChunkStore` Protocol with `upsert(chunks, vectors)`, `query(vector, item_filter, k) -> list[StoredChunk]`, `get(accession, chunk_id) -> StoredChunk | None`; `PgVectorStore`; `InMemoryChunkStore` and `FakeEmbedder` in `tests/fakes.py`

- [ ] **Step 1: Write the failing test**

`tests/index/test_store.py`:
```python
from era.index.chunking import Chunk
from tests.fakes import FakeEmbedder, InMemoryChunkStore


def test_query_returns_nearest_chunk_first() -> None:
    store = InMemoryChunkStore()
    chunks = [
        Chunk(chunk_id=0, accession="acc", item="1", text="smartphones and wearables"),
        Chunk(chunk_id=1, accession="acc", item="1A", text="supply chain risk"),
    ]
    embedder = FakeEmbedder()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    hits = store.query(embedder.embed(["supply chain risk"])[0], item_filter=None, k=1)

    assert hits[0].chunk_id == 1


def test_query_can_restrict_to_one_item() -> None:
    store = InMemoryChunkStore()
    chunks = [
        Chunk(chunk_id=0, accession="acc", item="1", text="supply chain risk"),
        Chunk(chunk_id=1, accession="acc", item="1A", text="supply chain risk"),
    ]
    embedder = FakeEmbedder()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    hits = store.query(
        embedder.embed(["supply chain risk"])[0], item_filter="1A", k=5
    )

    assert {h.item for h in hits} == {"1A"}


def test_get_resolves_a_citation_target() -> None:
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=7, accession="acc", item="1", text="hello")
    store.upsert([chunk], FakeEmbedder().embed(["hello"]))

    assert store.get("acc", 7) is not None
    assert store.get("acc", 999) is None


def test_upsert_is_idempotent() -> None:
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=0, accession="acc", item="1", text="hello")
    vectors = FakeEmbedder().embed(["hello"])

    store.upsert([chunk], vectors)
    store.upsert([chunk], vectors)

    assert len(store.query(vectors[0], item_filter=None, k=10)) == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/index/test_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the Protocols and the fakes**

`src/era/index/embeddings.py`:
```python
from typing import Protocol

import voyageai

MODEL = "voyage-finance-2"
DIMENSIONS = 1024


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(self, api_key: str) -> None:
        self._client = voyageai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=MODEL, input_type="document")
        embeddings: list[list[float]] = result.embeddings
        return embeddings
```

`src/era/index/store.py`:
```python
from dataclasses import dataclass
from typing import Protocol

from era.index.chunking import Chunk


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: int
    accession: str
    item: str
    text: str


class ChunkStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def query(
        self, vector: list[float], item_filter: str | None, k: int
    ) -> list[StoredChunk]: ...

    def get(self, accession: str, chunk_id: int) -> StoredChunk | None: ...
```

`tests/fakes.py`:
```python
"""Deterministic stand-ins for every external surface."""

import math

from era.index.chunking import Chunk
from era.index.store import StoredChunk

_VOCAB = ["smartphones", "wearables", "supply", "chain", "risk", "revenue", "services"]


class FakeEmbedder:
    """Bag-of-words vectors — deterministic and dependency-free."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append([float(lowered.count(word)) for word in _VOCAB])
        return vectors


class InMemoryChunkStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, int], tuple[StoredChunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            stored = StoredChunk(
                chunk_id=chunk.chunk_id,
                accession=chunk.accession,
                item=chunk.item,
                text=chunk.text,
            )
            self._rows[(chunk.accession, chunk.chunk_id)] = (stored, vector)

    def query(
        self, vector: list[float], item_filter: str | None, k: int
    ) -> list[StoredChunk]:
        scored: list[tuple[float, StoredChunk]] = []
        for stored, stored_vector in self._rows.values():
            if item_filter is not None and stored.item != item_filter:
                continue
            scored.append((_cosine(vector, stored_vector), stored))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [stored for _, stored in scored[:k]]

    def get(self, accession: str, chunk_id: int) -> StoredChunk | None:
        row = self._rows.get((accession, chunk_id))
        return row[0] if row else None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/index/test_store.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the real pgvector store**

Append to `src/era/index/store.py`:
```python
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    accession TEXT NOT NULL,
    chunk_id  INTEGER NOT NULL,
    item      TEXT NOT NULL,
    text      TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    PRIMARY KEY (accession, chunk_id)
);
"""


class PgVectorStore:
    def __init__(self, dsn: str) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute(SCHEMA)
        register_vector(self._conn)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        with self._conn.cursor() as cur:
            for chunk, vector in zip(chunks, vectors, strict=True):
                cur.execute(
                    """
                    INSERT INTO chunks (accession, chunk_id, item, text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (accession, chunk_id) DO UPDATE
                      SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """,
                    (chunk.accession, chunk.chunk_id, chunk.item, chunk.text, vector),
                )

    def query(
        self, vector: list[float], item_filter: str | None, k: int
    ) -> list[StoredChunk]:
        sql = "SELECT accession, chunk_id, item, text FROM chunks"
        params: list[object] = []
        if item_filter is not None:
            sql += " WHERE item = %s"
            params.append(item_filter)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params.extend([vector, k])
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [
                StoredChunk(accession=r[0], chunk_id=r[1], item=r[2], text=r[3])
                for r in cur.fetchall()
            ]

    def get(self, accession: str, chunk_id: int) -> StoredChunk | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT accession, chunk_id, item, text FROM chunks "
                "WHERE accession = %s AND chunk_id = %s",
                (accession, chunk_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return StoredChunk(accession=row[0], chunk_id=row[1], item=row[2], text=row[3])
```

- [ ] **Step 6: Write the pgvector integration test**

This is the one test permitted to touch a real database. It skips when
`DATABASE_URL` is absent, so CI stays network-free and deterministic while a
local run with Neon configured exercises the real SQL.

`tests/index/test_store_integration.py`:
```python
import os

import pytest

from era.index.chunking import Chunk
from era.index.store import PgVectorStore

DSN = os.environ.get("DATABASE_URL")
ACCESSION = "integration-test-fixture"

pytestmark = pytest.mark.skipif(
    not DSN, reason="DATABASE_URL not set — skipping live pgvector integration test"
)


@pytest.fixture
def cleanup():
    yield
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DELETE FROM chunks WHERE accession = %s", (ACCESSION,))


def test_round_trips_a_chunk_through_real_pgvector(cleanup) -> None:
    assert DSN is not None
    store = PgVectorStore(dsn=DSN)
    chunk = Chunk(chunk_id=1, accession=ACCESSION, item="1A", text="hello pgvector")
    vector = [0.0] * 1024
    vector[0] = 1.0

    store.upsert([chunk], [vector])

    fetched = store.get(ACCESSION, 1)
    assert fetched is not None
    assert fetched.text == "hello pgvector"

    hits = store.query(vector, item_filter="1A", k=1)
    assert hits[0].chunk_id == 1


def test_upsert_is_idempotent_against_real_pgvector(cleanup) -> None:
    assert DSN is not None
    store = PgVectorStore(dsn=DSN)
    chunk = Chunk(chunk_id=2, accession=ACCESSION, item="1", text="first")
    vector = [0.0] * 1024
    vector[0] = 1.0

    store.upsert([chunk], [vector])
    store.upsert(
        [Chunk(chunk_id=2, accession=ACCESSION, item="1", text="second")], [vector]
    )

    fetched = store.get(ACCESSION, 2)
    assert fetched is not None
    assert fetched.text == "second"
```

- [ ] **Step 7: Run it and confirm it skips cleanly**

Run: `uv run pytest tests/index/test_store_integration.py -v`
Expected without `DATABASE_URL` set: 2 skipped, 0 failed. The skip reason must
name the missing variable — a test that silently passes when it did not run is
worse than no test.

- [ ] **Step 8: Verify types and commit**

```bash
uv run mypy src
uv run pytest -q
git add src/era/index/ tests/index/ tests/fakes.py
git commit -m "feat: add embedder and chunk store behind Protocols with fakes"
```

---

### Task 10: Ingest pipeline and `era ingest`

**Files:**
- Create: `src/era/index/ingest.py`, `src/era/cli.py`, `tests/index/test_ingest.py`

**Interfaces:**
- Consumes: `EdgarClient`, `resolve_cik`, `latest_filings`, `parse_items`, `chunk_items`, `Embedder`, `ChunkStore`
- Produces: `IngestResult(cik, accessions, chunks_written, items_missing)`, `ingest_ticker(ticker, client, embedder, store) -> IngestResult`

- [ ] **Step 1: Write the failing test**

`tests/index/test_ingest.py`:
```python
import httpx
import respx

from era.edgar.client import EdgarClient
from era.index.ingest import ingest_ticker
from tests.fakes import FakeEmbedder, InMemoryChunkStore

UA = "Berkay Koklu kokluberkay@gmail.com"
TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
SUBMISSIONS = {
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000123"],
            "form": ["10-K"],
            "filingDate": ["2024-11-01"],
            "primaryDocument": ["aapl.htm"],
        }
    }
}
FILING_HTML = (
    "<p>Item 1. Business</p><p>We design smartphones.</p>"
    "<p>Item 1A. Risk Factors</p><p>Supply chain risk is material.</p>"
    "<p>Item 8. Financial Statements</p>"
)


def _mock_edgar() -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=SUBMISSIONS)
    )
    respx.get(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl.htm"
    ).mock(return_value=httpx.Response(200, text=FILING_HTML))


@respx.mock
def test_ingest_writes_chunks_and_reports_what_it_found() -> None:
    _mock_edgar()
    store = InMemoryChunkStore()

    result = ingest_ticker(
        "AAPL", EdgarClient(user_agent=UA, cache_dir=None), FakeEmbedder(), store
    )

    assert result.cik == "0000320193"
    assert result.chunks_written > 0
    assert "7" in result.items_missing


@respx.mock
def test_ingest_is_idempotent() -> None:
    _mock_edgar()
    store = InMemoryChunkStore()
    client = EdgarClient(user_agent=UA, cache_dir=None)

    first = ingest_ticker("AAPL", client, FakeEmbedder(), store)
    ingest_ticker("AAPL", client, FakeEmbedder(), store)

    hits = store.query(FakeEmbedder().embed(["risk"])[0], item_filter=None, k=100)
    assert len(hits) == first.chunks_written
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/index/test_ingest.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the ingest pipeline**

`src/era/index/ingest.py`:
```python
from dataclasses import dataclass

from era.edgar.client import EdgarClient
from era.edgar.filings import latest_filings, resolve_cik
from era.edgar.sections import parse_items
from era.index.chunking import chunk_items
from era.index.embeddings import Embedder
from era.index.store import ChunkStore


@dataclass(frozen=True)
class IngestResult:
    cik: str
    accessions: tuple[str, ...]
    chunks_written: int
    items_missing: tuple[str, ...]


def ingest_ticker(
    ticker: str, client: EdgarClient, embedder: Embedder, store: ChunkStore
) -> IngestResult:
    cik = resolve_cik(client, ticker)
    filings = latest_filings(client, cik)

    accessions: list[str] = []
    missing: set[str] = set()
    written = 0

    for filing in filings:
        html = client.get_text(filing.primary_document_url)
        parsed = parse_items(html)
        missing.update(parsed.missing)
        chunks = chunk_items(filing.accession, parsed.items)
        if chunks:
            store.upsert(chunks, embedder.embed([c.text for c in chunks]))
            written += len(chunks)
        accessions.append(filing.accession)

    return IngestResult(
        cik=cik,
        accessions=tuple(accessions),
        chunks_written=written,
        items_missing=tuple(sorted(missing)),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/index/test_ingest.py -v`
Expected: 2 passed

- [ ] **Step 5: Write the CLI ingest command**

`src/era/cli.py`:
```python
from pathlib import Path

import typer

from era.config import Settings
from era.edgar.client import EdgarClient
from era.index.embeddings import VoyageEmbedder
from era.index.ingest import ingest_ticker
from era.index.store import PgVectorStore

app = typer.Typer(help="Cited equity research from SEC filings.")
CACHE_DIR = Path(".cache/edgar")


@app.command()
def ingest(ticker: str) -> None:
    """Fetch, parse, chunk and index a company's latest filings."""
    settings = Settings()  # type: ignore[call-arg]
    result = ingest_ticker(
        ticker,
        EdgarClient(user_agent=settings.edgar_user_agent, cache_dir=CACHE_DIR),
        VoyageEmbedder(api_key=settings.voyage_api_key),
        PgVectorStore(dsn=settings.database_url),
    )
    typer.echo(f"CIK {result.cik}")
    typer.echo(f"filings: {', '.join(result.accessions)}")
    typer.echo(f"chunks written: {result.chunks_written}")
    if result.items_missing:
        typer.echo(f"items not found: {', '.join(result.items_missing)}")
```

- [ ] **Step 6: Commit**

```bash
uv run mypy src && uv run pytest -q
git add src/era/index/ingest.py src/era/cli.py tests/index/test_ingest.py
git commit -m "feat: add ingest pipeline and era ingest command"
```

---

### Task 11: Deterministic verifier

**The centrepiece.** This module runs inside the graph at runtime *and* as the Tier 1 CI gate — one implementation, two callers.

**Files:**
- Create: `src/era/verify/checks.py`, `tests/verify/test_checks.py`

**Interfaces:**
- Consumes: `Claim`, `Section`, `ResearchNote` (Task 3); `ChunkStore` (Task 9); `NormalizedFacts` (Task 7)
- Produces: `Violation(kind, detail)`, `verify_section(section, store, facts) -> list[Violation]`, `verify_note(note, store, facts) -> list[Violation]`, `no_recommendation_language(text) -> list[Violation]`

- [ ] **Step 1: Write the failing test**

`tests/verify/test_checks.py`:
```python
from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.index.chunking import Chunk
from era.report.schema import ChunkRef, Claim, FactRef, Section, SectionName
from era.verify.checks import no_recommendation_language, verify_section
from tests.fakes import FakeEmbedder, InMemoryChunkStore

FACTS = NormalizedFacts(
    metrics={
        "revenue": MetricValue(
            tag="Revenues", fiscal_period="FY2024",
            value=391_035_000_000.0, accession="acc",
        )
    },
    missing=(),
)


def _store_with(text: str) -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=0, accession="acc", item="1A", text=text)
    store.upsert([chunk], FakeEmbedder().embed([text]))
    return store


def test_accepts_a_claim_whose_chunk_supports_it() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="Supply chain concentration is a material risk.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    assert verify_section(section, store, FACTS) == []


def test_rejects_a_citation_that_does_not_resolve() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="Supply chain concentration is a material risk.",
                chunks=(ChunkRef(accession="acc", chunk_id=999),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS)

    assert [v.kind for v in violations] == ["unresolvable_citation"]


def test_rejects_a_number_absent_from_the_cited_chunk() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="The company lost 42 factories.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS)

    assert [v.kind for v in violations] == ["unsupported_figure"]


def test_rejects_a_fact_value_that_disagrees_with_xbrl() -> None:
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was 400 billion USD.",
                facts=(
                    FactRef(
                        tag="Revenues", fiscal_period="FY2024",
                        value=400_000_000_000.0, accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), FACTS)

    assert [v.kind for v in violations] == ["figure_disagrees_with_xbrl"]


def test_flags_recommendation_language() -> None:
    assert no_recommendation_language("We rate the shares a Buy.")
    assert no_recommendation_language("Our price target is 250 USD.")
    assert no_recommendation_language("Revenue grew in fiscal 2024.") == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/verify/test_checks.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the verifier**

`src/era/verify/checks.py`:
```python
import re
from dataclasses import dataclass

from era.edgar.xbrl import NormalizedFacts
from era.index.store import ChunkStore
from era.report.schema import ResearchNote, Section

FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")
RECOMMENDATION_TERMS = (
    "buy", "sell", "hold", "overweight", "underweight", "price target",
    "outperform", "underperform", "strong buy", "we recommend",
)
RELATIVE_TOLERANCE = 0.005


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str


def _figures(text: str) -> set[str]:
    return {m.group().replace(",", "") for m in FIGURE.finditer(text)}


def no_recommendation_language(text: str) -> list[Violation]:
    lowered = text.lower()
    return [
        Violation(kind="recommendation_language", detail=term)
        for term in RECOMMENDATION_TERMS
        if re.search(rf"\b{re.escape(term)}\b", lowered)
    ]


def verify_section(
    section: Section, store: ChunkStore, facts: NormalizedFacts
) -> list[Violation]:
    if not section.available:
        return []

    violations: list[Violation] = []

    for claim in section.claims:
        violations.extend(no_recommendation_language(claim.text))

        supporting_text: list[str] = []
        for ref in claim.chunks:
            stored = store.get(ref.accession, ref.chunk_id)
            if stored is None:
                violations.append(
                    Violation(
                        kind="unresolvable_citation",
                        detail=f"{ref.accession}#{ref.chunk_id}",
                    )
                )
                continue
            supporting_text.append(stored.text)

        for fact in claim.facts:
            match = next(
                (m for m in facts.metrics.values() if m.tag == fact.tag), None
            )
            if match is None:
                violations.append(
                    Violation(kind="unknown_fact_tag", detail=fact.tag)
                )
            elif abs(match.value - fact.value) > abs(match.value) * RELATIVE_TOLERANCE:
                violations.append(
                    Violation(
                        kind="figure_disagrees_with_xbrl",
                        detail=f"{fact.tag}: claimed {fact.value}, XBRL {match.value}",
                    )
                )
            else:
                supporting_text.append(f"{match.value}")

        claimed = _figures(claim.text)
        if claimed:
            supported = set().union(*(_figures(t) for t in supporting_text)) \
                if supporting_text else set()
            for figure in claimed - supported:
                violations.append(
                    Violation(kind="unsupported_figure", detail=figure)
                )

    return violations


def verify_note(
    note: ResearchNote, store: ChunkStore, facts: NormalizedFacts
) -> list[Violation]:
    violations: list[Violation] = []
    for section in note.sections:
        violations.extend(verify_section(section, store, facts))
    return violations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/verify/test_checks.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/era/verify/checks.py tests/verify/test_checks.py
git commit -m "feat: add deterministic claim verifier with XBRL figure checking"
```

---

### Task 12: Provider factory and section subgraph

**Blocked by Task 2.** Follow the spike's recorded decision on sync vs async fan-out.

**Files:**
- Create: `src/era/graph/models.py`, `src/era/graph/state.py`, `src/era/graph/section.py`, `tests/graph/test_section.py`

**Interfaces:**
- Consumes: `Section`, `Claim` (Task 3); `ChunkStore`, `Embedder` (Task 9); `facts_card` (Task 7)
- Produces: `build_model(provider, model_name) -> BaseChatModel`; `SectionRequest`; `SectionDraft` (LLM structured-output model); `research_section(request, model, store, embedder, facts_card_text) -> Section`

- [ ] **Step 1: Write the failing test**

`tests/graph/test_section.py`:
```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from era.graph.section import SectionRequest, research_section
from era.index.chunking import Chunk
from era.report.schema import SectionName
from tests.fakes import FakeEmbedder, InMemoryChunkStore


class StructuredFakeModel(GenericFakeChatModel):
    """A fake whose with_structured_output returns a fixed payload."""

    payload: dict = {}

    def with_structured_output(self, schema, **kwargs):  # type: ignore[override]
        payload = self.payload

        class _Runnable:
            def invoke(self, _input, config=None):
                return schema(**payload)

        return _Runnable()


def _store() -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    chunk = Chunk(
        chunk_id=0, accession="acc", item="1A",
        text="Supply chain concentration is a material risk.",
    )
    store.upsert([chunk], FakeEmbedder().embed([chunk.text]))
    return store


def test_returns_a_section_of_claims_with_citations() -> None:
    model = StructuredFakeModel(messages=iter([AIMessage(content="")]))
    model.payload = {
        "claims": [
            {
                "text": "Supply chain concentration is a material risk.",
                "chunk_ids": [0],
                "facts": [],
            }
        ]
    }

    section = research_section(
        SectionRequest(
            name=SectionName.RISK_FACTORS, ticker="AAPL", item_filter="1A", k=4
        ),
        model,
        _store(),
        FakeEmbedder(),
        facts_card_text="revenue: 391,035,000,000 USD (FY2024)",
    )

    assert section.available is True
    assert section.claims[0].chunks[0].chunk_id == 0


def test_marks_the_section_unavailable_when_retrieval_finds_nothing() -> None:
    model = StructuredFakeModel(messages=iter([AIMessage(content="")]))
    model.payload = {"claims": []}

    section = research_section(
        SectionRequest(
            name=SectionName.RISK_FACTORS, ticker="AAPL", item_filter="1A", k=4
        ),
        model,
        InMemoryChunkStore(),
        FakeEmbedder(),
        facts_card_text="",
    )

    assert section.available is False
    assert "no indexed content" in section.unavailable_reason
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/graph/test_section.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the model factory**

`src/era/graph/models.py`:
```python
from langchain_core.language_models import BaseChatModel

DRAFTING_MODEL = "gpt-5.6-terra"
JUDGE_MODEL = "gpt-5.6-luna"


def build_model(
    provider: str = "openai", model_name: str = DRAFTING_MODEL, temperature: float = 0.2
) -> BaseChatModel:
    """One seam for the provider. Swapping is a config change, not a rewrite."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=temperature)
    raise ValueError(f"unknown provider: {provider}")
```

- [ ] **Step 4: Write the section subgraph**

`src/era/graph/section.py`:
```python
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from era.index.embeddings import Embedder
from era.index.store import ChunkStore
from era.report.schema import ChunkRef, Claim, FactRef, Section, SectionName

SECTION_QUERIES: dict[SectionName, tuple[str, str | None]] = {
    SectionName.BUSINESS_OVERVIEW: ("what the company does, its products and markets", "1"),
    SectionName.FINANCIAL_HEALTH: ("revenue, margins, cash flow and leverage", "7"),
    SectionName.RISK_FACTORS: ("principal risks and uncertainties", "1A"),
    SectionName.RECENT_DEVELOPMENTS: ("recent results and material changes", "7"),
    SectionName.VALUATION_CONTEXT: ("growth, profitability and capital returns", "7"),
}

PROMPT = """You are writing the "{section}" section of an equity research note on \
{ticker}.

Write at most 6 claims totalling about 600 words. Every claim MUST cite at least one \
source:
- cite `chunk_ids` for anything drawn from the filing excerpts below
- cite `facts` for any figure, using the exact tag and value from the facts card

Never state a figure that does not appear in an excerpt or the facts card. Never \
recommend buying, selling or holding, and never give a price target.

Facts card:
{facts_card}

Filing excerpts:
{excerpts}
"""


class _DraftFact(BaseModel):
    tag: str
    fiscal_period: str
    value: float


class _DraftClaim(BaseModel):
    text: str
    chunk_ids: list[int] = Field(default_factory=list)
    facts: list[_DraftFact] = Field(default_factory=list)


class SectionDraft(BaseModel):
    claims: list[_DraftClaim] = Field(default_factory=list)


@dataclass(frozen=True)
class SectionRequest:
    name: SectionName
    ticker: str
    item_filter: str | None
    k: int = 4
    complaint: str | None = None


def research_section(
    request: SectionRequest,
    model: BaseChatModel,
    store: ChunkStore,
    embedder: Embedder,
    facts_card_text: str,
) -> Section:
    query, _ = SECTION_QUERIES[request.name]
    vector = embedder.embed([query])[0]
    hits = store.query(vector, item_filter=request.item_filter, k=request.k)

    if not hits:
        return Section(
            name=request.name,
            available=False,
            unavailable_reason=(
                f"no indexed content for item {request.item_filter}"
            ),
        )

    excerpts = "\n\n".join(
        f"[chunk {h.chunk_id} · {h.accession} · item {h.item}]\n{h.text}" for h in hits
    )
    prompt = PROMPT.format(
        section=request.name.value.replace("_", " "),
        ticker=request.ticker,
        facts_card=facts_card_text,
        excerpts=excerpts,
    )
    if request.complaint:
        prompt += f"\n\nA previous attempt was rejected because: {request.complaint}\n"

    draft = model.with_structured_output(SectionDraft).invoke(prompt)

    by_id = {h.chunk_id: h for h in hits}
    claims: list[Claim] = []
    for drafted in draft.claims:
        chunks = tuple(
            ChunkRef(accession=by_id[cid].accession, chunk_id=cid)
            for cid in drafted.chunk_ids
            if cid in by_id
        )
        facts = tuple(
            FactRef(
                tag=f.tag, fiscal_period=f.fiscal_period, value=f.value,
                accession=hits[0].accession,
            )
            for f in drafted.facts
        )
        if not chunks and not facts:
            continue
        claims.append(Claim(text=drafted.text, chunks=chunks, facts=facts))

    if not claims:
        return Section(
            name=request.name,
            available=False,
            unavailable_reason="model produced no citable claims",
        )
    return Section(name=request.name, claims=tuple(claims))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/graph/test_section.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
uv run mypy src && uv run pytest -q
git add src/era/graph/ tests/graph/
git commit -m "feat: add provider factory and section research subgraph"
```

---

### Task 13: Graph assembly with fan-out and bounded retry

**Files:**
- Create: `src/era/graph/build.py`, `tests/graph/test_build.py`

**Interfaces:**
- Consumes: everything from Tasks 3, 7, 9, 11, 12
- Produces: `ResearchState` TypedDict; `build_research_graph(model, store, embedder, facts, max_retries) -> CompiledGraph`

- [ ] **Step 1: Write the failing test**

`tests/graph/test_build.py`:
```python
from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.graph.build import build_research_graph
from era.index.chunking import Chunk
from era.report.schema import SectionName
from tests.fakes import FakeEmbedder, InMemoryChunkStore
from tests.graph.test_section import StructuredFakeModel
from langchain_core.messages import AIMessage

FACTS = NormalizedFacts(
    metrics={
        "revenue": MetricValue(
            tag="Revenues", fiscal_period="FY2024", value=1000.0, accession="acc"
        )
    },
    missing=("total_debt",),
)


def _populated_store() -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    chunks = [
        Chunk(chunk_id=i, accession="acc", item=item, text="supply chain risk revenue")
        for i, item in enumerate(["1", "1A", "7", "7", "7"])
    ]
    store.upsert(chunks, FakeEmbedder().embed([c.text for c in chunks]))
    return store


def _model(text: str) -> StructuredFakeModel:
    model = StructuredFakeModel(messages=iter([AIMessage(content="")] * 50))
    model.payload = {"claims": [{"text": text, "chunk_ids": [0], "facts": []}]}
    return model


def test_produces_a_note_with_every_section() -> None:
    graph = build_research_graph(
        _model("Supply chain risk is material."),
        _populated_store(),
        FakeEmbedder(),
        FACTS,
        max_retries=1,
    )

    note = graph.invoke({"ticker": "AAPL", "cik": "0000320193"})

    assert {s.name for s in note["note"].sections} == set(SectionName)


def test_coverage_reflects_unavailable_sections_and_missing_metrics() -> None:
    graph = build_research_graph(
        _model("Supply chain risk is material."),
        InMemoryChunkStore(),
        FakeEmbedder(),
        FACTS,
        max_retries=1,
    )

    note = graph.invoke({"ticker": "AAPL", "cik": "0000320193"})["note"]

    assert note.coverage.sections_available == 0
    assert "total_debt" in note.coverage.metrics_missing


def test_retries_are_bounded() -> None:
    # A claim citing a figure absent from any chunk always fails verification.
    graph = build_research_graph(
        _model("Revenue was 999999 dollars."),
        _populated_store(),
        FakeEmbedder(),
        FACTS,
        max_retries=2,
    )

    result = graph.invoke({"ticker": "AAPL", "cik": "0000320193"})

    assert max(result["attempts"].values()) <= 3  # initial + 2 retries
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/graph/test_build.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the graph**

`src/era/graph/build.py`:
```python
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from era.edgar.xbrl import NormalizedFacts, facts_card
from era.graph.section import SECTION_QUERIES, SectionRequest, research_section
from era.index.embeddings import Embedder
from era.index.store import ChunkStore
from era.report.schema import Coverage, ResearchNote, Section, SectionName
from era.verify.checks import verify_section


class ResearchState(TypedDict, total=False):
    ticker: str
    cik: str
    sections: Annotated[list[Section], operator.add]
    attempts: dict[str, int]
    complaints: dict[str, str]
    note: ResearchNote


def build_research_graph(
    model: BaseChatModel,
    store: ChunkStore,
    embedder: Embedder,
    facts: NormalizedFacts,
    max_retries: int = 2,
) -> Any:
    card = facts_card(facts)

    def make_section_node(name: SectionName):
        _, item = SECTION_QUERIES[name]

        def node(state: ResearchState) -> ResearchState:
            attempts = dict(state.get("attempts") or {})
            complaints = state.get("complaints") or {}
            attempt = attempts.get(name.value, 0)

            section = research_section(
                SectionRequest(
                    name=name,
                    ticker=state["ticker"],
                    item_filter=item,
                    complaint=complaints.get(name.value),
                ),
                model,
                store,
                embedder,
                card,
            )
            attempts[name.value] = attempt + 1
            return {"sections": [section], "attempts": attempts}

        return node

    def verify(state: ResearchState) -> ResearchState:
        """Keep the best attempt per section; record complaints for retries."""
        latest: dict[SectionName, Section] = {}
        for section in state.get("sections", []):
            latest[section.name] = section

        complaints: dict[str, str] = {}
        for name, section in latest.items():
            violations = verify_section(section, store, facts)
            if violations:
                complaints[name.value] = "; ".join(
                    f"{v.kind}: {v.detail}" for v in violations[:5]
                )
        return {"complaints": complaints}

    def should_retry(state: ResearchState) -> str:
        complaints = state.get("complaints") or {}
        attempts = state.get("attempts") or {}
        retryable = [
            name for name in complaints if attempts.get(name, 0) <= max_retries
        ]
        return "retry" if retryable else "finalize"

    def retry(state: ResearchState) -> ResearchState:
        complaints = state.get("complaints") or {}
        attempts = dict(state.get("attempts") or {})
        produced: list[Section] = []

        for name_value in complaints:
            name = SectionName(name_value)
            if attempts.get(name_value, 0) > max_retries:
                continue
            _, item = SECTION_QUERIES[name]
            produced.append(
                research_section(
                    SectionRequest(
                        name=name,
                        ticker=state["ticker"],
                        item_filter=item,
                        complaint=complaints[name_value],
                    ),
                    model,
                    store,
                    embedder,
                    card,
                )
            )
            attempts[name_value] = attempts.get(name_value, 0) + 1

        return {"sections": produced, "attempts": attempts}

    def finalize(state: ResearchState) -> ResearchState:
        latest: dict[SectionName, Section] = {}
        for section in state.get("sections", []):
            latest[section.name] = section

        ordered = tuple(latest[name] for name in SectionName if name in latest)
        available = sum(1 for s in ordered if s.available)
        accessions = tuple(
            sorted(
                {
                    ref.accession
                    for section in ordered
                    for claim in section.claims
                    for ref in claim.chunks
                }
            )
        )
        note = ResearchNote(
            ticker=state["ticker"],
            cik=state["cik"],
            sections=ordered,
            coverage=Coverage(
                sections_available=available,
                sections_total=len(SectionName),
                metrics_resolved=tuple(sorted(facts.metrics)),
                metrics_missing=facts.missing,
            ),
            accessions=accessions,
        )
        return {"note": note}

    graph = StateGraph(ResearchState)
    for name in SectionName:
        graph.add_node(name.value, make_section_node(name))
        graph.add_edge(START, name.value)
        graph.add_edge(name.value, "verify")

    graph.add_node("verify", verify)
    graph.add_node("retry", retry)
    graph.add_node("finalize", finalize)
    graph.add_conditional_edges(
        "verify", should_retry, {"retry": "retry", "finalize": "finalize"}
    )
    graph.add_edge("retry", "verify")
    graph.add_edge("finalize", END)

    return graph.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/graph/test_build.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
uv run mypy src && uv run pytest -q
git add src/era/graph/build.py tests/graph/test_build.py
git commit -m "feat: assemble research graph with parallel fan-out and bounded retry"
```

---

### Task 14: Markdown assembly in plain Python

Deterministic by design: an assembler that cannot introduce a claim the verifier never saw.

**Files:**
- Create: `src/era/report/assemble.py`, `tests/report/test_assemble.py`

**Interfaces:**
- Consumes: `ResearchNote` (Task 3)
- Produces: `render_markdown(note) -> str`

- [ ] **Step 1: Write the failing test**

`tests/report/test_assemble.py`:
```python
from era.report.assemble import render_markdown
from era.report.schema import (
    ChunkRef, Claim, Coverage, ResearchNote, Section, SectionName,
)

NOTE = ResearchNote(
    ticker="AAPL",
    cik="0000320193",
    sections=(
        Section(
            name=SectionName.RISK_FACTORS,
            claims=(
                Claim(
                    text="Supply chain concentration is a material risk.",
                    chunks=(ChunkRef(accession="0000320193-24-000123", chunk_id=4),),
                ),
            ),
        ),
        Section(
            name=SectionName.BUSINESS_OVERVIEW,
            available=False,
            unavailable_reason="Item 1 boundary not found",
        ),
    ),
    coverage=Coverage(
        sections_available=1, sections_total=5,
        metrics_resolved=("revenue",), metrics_missing=("total_debt",),
    ),
    cost_usd=0.1331,
    latency_seconds=12.5,
)


def test_renders_claims_with_their_citations() -> None:
    out = render_markdown(NOTE)

    assert "Supply chain concentration is a material risk." in out
    assert "0000320193-24-000123#4" in out


def test_states_why_a_section_is_unavailable() -> None:
    out = render_markdown(NOTE)

    assert "Item 1 boundary not found" in out


def test_reports_coverage_and_run_cost() -> None:
    out = render_markdown(NOTE)

    assert "1 of 5" in out
    assert "total_debt" in out
    assert "$0.1331" in out


def test_carries_the_not_advice_notice() -> None:
    assert "not investment advice" in render_markdown(NOTE).lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/report/test_assemble.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the renderer**

`src/era/report/assemble.py`:
```python
from era.report.schema import ResearchNote

TITLES = {
    "business_overview": "Business overview",
    "financial_health": "Financial health",
    "risk_factors": "Risk factors",
    "recent_developments": "Recent developments",
    "valuation_context": "Valuation context",
}


def render_markdown(note: ResearchNote) -> str:
    lines: list[str] = [f"# {note.ticker} — research note", ""]
    lines.append(f"CIK {note.cik} · sources: {', '.join(note.accessions) or 'none'}")
    lines.append("")

    for section in note.sections:
        lines.append(f"## {TITLES[section.name.value]}")
        lines.append("")
        if not section.available:
            lines.append(f"*Unavailable — {section.unavailable_reason}.*")
            lines.append("")
            continue
        for claim in section.claims:
            citations = [f"{c.accession}#{c.chunk_id}" for c in claim.chunks]
            citations += [f"{f.tag} {f.fiscal_period}" for f in claim.facts]
            lines.append(f"{claim.text} [{'; '.join(citations)}]")
            lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append(
        f"Sections populated: {note.coverage.sections_available} of "
        f"{note.coverage.sections_total}."
    )
    if note.coverage.metrics_missing:
        lines.append(
            f"Metrics that could not be resolved from XBRL: "
            f"{', '.join(note.coverage.metrics_missing)}."
        )
    lines.append("")
    lines.append(
        f"Run cost ${note.cost_usd:.4f} · {note.latency_seconds:.1f}s"
    )
    lines.append("")
    lines.append(
        "*This is an automated summary of public filings. It is not investment "
        "advice and contains no recommendation.*"
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/report/test_assemble.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/era/report/assemble.py tests/report/test_assemble.py
git commit -m "feat: render research notes to markdown deterministically"
```

---

### Task 15: Observability and `era research`

**Files:**
- Create: `src/era/observability/tracing.py`, `tests/observability/test_tracing.py`
- Modify: `src/era/cli.py`

**Interfaces:**
- Consumes: the spike decision from Task 2; `build_research_graph` (Task 13)
- Produces: `RunRecord(ticker, cost_usd, latency_seconds, trace_id)`, `tracked_run(graph, inputs, project) -> tuple[dict, RunRecord]`

- [ ] **Step 1: Write the failing test**

`tests/observability/test_tracing.py`:
```python
from era.observability.tracing import cost_from_usage


def test_costs_input_and_output_tokens_at_their_own_rates() -> None:
    cost = cost_from_usage(
        {"input_tokens": 6_300, "output_tokens": 800}, model="gpt-5.6-terra"
    )

    assert round(cost, 6) == round(6_300 / 1e6 * 2.0 + 800 / 1e6 * 12.0, 6)


def test_cached_input_is_charged_at_the_cached_rate() -> None:
    cost = cost_from_usage(
        {"input_tokens": 1_000, "cache_read_tokens": 5_000, "output_tokens": 0},
        model="gpt-5.6-terra",
    )

    assert round(cost, 6) == round(1_000 / 1e6 * 2.0 + 5_000 / 1e6 * 0.2, 6)


def test_unknown_model_costs_nothing_rather_than_guessing() -> None:
    assert cost_from_usage({"input_tokens": 100}, model="mystery") == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/observability/test_tracing.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the tracing module**

`src/era/observability/tracing.py`:
```python
import time
from dataclasses import dataclass
from typing import Any

# USD per million tokens, from OpenAI's published rates on 2026-08-09.
PRICING: dict[str, dict[str, float]] = {
    "gpt-5.6-terra": {"input": 2.0, "cached": 0.2, "output": 12.0},
    "gpt-5.6-luna": {"input": 0.2, "cached": 0.02, "output": 1.2},
}


@dataclass(frozen=True)
class RunRecord:
    ticker: str
    cost_usd: float
    latency_seconds: float
    trace_id: str | None


def cost_from_usage(usage: dict[str, int], model: str) -> float:
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    return (
        usage.get("input_tokens", 0) / 1e6 * rates["input"]
        + usage.get("cache_read_tokens", 0) / 1e6 * rates["cached"]
        + usage.get("output_tokens", 0) / 1e6 * rates["output"]
    )


def tracked_run(
    graph: Any, inputs: dict[str, Any], project: str = "equity-research-agent"
) -> tuple[dict[str, Any], RunRecord]:
    """Run the graph under an Opik tracer, recording cost and latency.

    Follow the sync/async decision recorded in
    docs/spikes/2026-08-10-opik-parallel-fanout.md.
    """
    from opik.integrations.langchain import OpikTracer

    tracer = OpikTracer(project_name=project, graph=graph.get_graph(xray=True))
    started = time.monotonic()
    result = graph.invoke(inputs, config={"callbacks": [tracer]})
    elapsed = time.monotonic() - started
    tracer.flush()

    return result, RunRecord(
        ticker=inputs["ticker"],
        cost_usd=0.0,  # populated from usage metadata once the spike confirms capture
        latency_seconds=elapsed,
        trace_id=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/observability/test_tracing.py -v`
Expected: 3 passed

- [ ] **Step 5: Add the `research` command**

Append to `src/era/cli.py`:
```python
@app.command()
def research(ticker: str) -> None:
    """Produce a cited research note for an already-indexed ticker."""
    from era.edgar.filings import resolve_cik
    from era.edgar.xbrl import normalize_facts
    from era.graph.build import build_research_graph
    from era.graph.models import build_model
    from era.observability.tracing import tracked_run
    from era.report.assemble import render_markdown

    settings = Settings()  # type: ignore[call-arg]
    client = EdgarClient(user_agent=settings.edgar_user_agent, cache_dir=CACHE_DIR)
    cik = resolve_cik(client, ticker)
    facts = normalize_facts(
        client.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    )

    graph = build_research_graph(
        build_model(),
        PgVectorStore(dsn=settings.database_url),
        VoyageEmbedder(api_key=settings.voyage_api_key),
        facts,
    )
    result, record = tracked_run(graph, {"ticker": ticker.upper(), "cik": cik})
    # `revalidated` re-runs the schema validators; `model_copy` would skip them.
    note = revalidated(
        result["note"],
        latency_seconds=record.latency_seconds,
        cost_usd=record.cost_usd,
    )
    typer.echo(render_markdown(note))
```

- [ ] **Step 6: Commit**

```bash
uv run mypy src && uv run pytest -q
git add src/era/observability/ src/era/cli.py tests/observability/
git commit -m "feat: add Opik tracing, cost accounting and era research command"
```

---

### Task 16: Tier 1 eval suite in CI

The verifier from Task 11 becomes the merge gate, running over a recorded note fixture so it needs no network, no key, and no database.

**Files:**
- Create: `evals/tier1/test_note_contract.py`, `evals/fixtures/note_aapl.json`, `evals/fixtures/chunks_aapl.json`
- Modify: `.github/workflows/ci.yml` (already runs `pytest`, which now collects `evals/tier1`)

**Interfaces:**
- Consumes: `verify_note` (Task 11), `ResearchNote` (Task 3)
- Produces: the CI gate

- [ ] **Step 1: Record the fixtures**

Generate them once from a real run, then commit. `evals/fixtures/chunks_aapl.json` is a list of `{accession, chunk_id, item, text}`; `evals/fixtures/note_aapl.json` is `ResearchNote.model_dump(mode="json")` from that run. Write them by hand if no key is available yet — the contract, not the prose, is what is being tested.

- [ ] **Step 2: Write the eval**

`evals/tier1/test_note_contract.py`:
```python
import json
from pathlib import Path

import pytest

from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.index.chunking import Chunk
from era.report.assemble import render_markdown
from era.report.schema import ResearchNote, SectionName
from era.verify.checks import no_recommendation_language, verify_note
from tests.fakes import FakeEmbedder, InMemoryChunkStore

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def note() -> ResearchNote:
    return ResearchNote.model_validate_json((FIXTURES / "note_aapl.json").read_text())


@pytest.fixture
def store() -> InMemoryChunkStore:
    rows = json.loads((FIXTURES / "chunks_aapl.json").read_text())
    chunks = [Chunk(**row) for row in rows]
    store = InMemoryChunkStore()
    store.upsert(chunks, FakeEmbedder().embed([c.text for c in chunks]))
    return store


@pytest.fixture
def facts() -> NormalizedFacts:
    return NormalizedFacts(
        metrics={
            "revenue": MetricValue(
                tag="Revenues", fiscal_period="FY2024",
                value=391_035_000_000.0, accession="0000320193-24-000123",
            )
        },
        missing=(),
    )


def test_every_claim_is_verifiable(note, store, facts) -> None:
    violations = verify_note(note, store, facts)

    assert violations == [], f"{len(violations)} unverifiable claims: {violations[:3]}"


def test_every_section_is_present_or_explained(note) -> None:
    names = {s.name for s in note.sections}

    assert names == set(SectionName)
    for section in note.sections:
        assert section.available or section.unavailable_reason


def test_the_note_recommends_nothing(note) -> None:
    assert no_recommendation_language(render_markdown(note)) == []


def test_the_rendered_note_carries_the_not_advice_notice(note) -> None:
    assert "not investment advice" in render_markdown(note).lower()
```

- [ ] **Step 3: Run the suite**

Run: `uv run pytest evals/tier1 -v`
Expected: 4 passed

- [ ] **Step 4: Confirm CI collects it**

Run: `uv run pytest -q`
Expected: `tests/` and `evals/tier1/` both collected — `testpaths` in `pyproject.toml` already lists both.

- [ ] **Step 5: Commit**

```bash
git add evals/
git commit -m "test: add Tier 1 note contract evals as the CI merge gate"
```

---

### Task 17: Tier 2 judged evals and README

**Files:**
- Create: `evals/tier2/run_experiment.py`, `.github/workflows/evals.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `build_research_graph`, `render_markdown`, Opik
- Produces: an on-merge judged experiment and the repo's public face

- [ ] **Step 1: Write the experiment script**

`evals/tier2/run_experiment.py`:
```python
"""Judged evals over the golden ticker set. Runs on merge, never gates a merge."""

import os

import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import AnswerRelevance, ContextPrecision, Hallucination

from era.report.assemble import render_markdown

JUDGE = "openai/gpt-5.6-luna"
GOLDEN_TICKERS = [
    "AAPL",   # clean large-cap
    "BRK-B",  # unusual filing structure
    "KO", "JNJ", "XOM", "WMT", "PG", "JPM", "CVX", "MRK",
]


def build_dataset() -> opik.Dataset:
    client = opik.Opik()
    dataset = client.get_or_create_dataset(name="era-golden-tickers")
    dataset.insert([{"ticker": t} for t in GOLDEN_TICKERS])
    return dataset


def task(item: dict[str, str]) -> dict[str, str]:
    from era.cli import research_note_for  # thin wrapper added in Plan 2

    note, context = research_note_for(item["ticker"])
    return {
        "input": f"Write a research note on {item['ticker']}",
        "output": render_markdown(note),
        "context": context,
    }


if __name__ == "__main__":
    evaluate(
        dataset=build_dataset(),
        task=task,
        scoring_metrics=[
            Hallucination(model=JUDGE),
            AnswerRelevance(model=JUDGE),
            ContextPrecision(model=JUDGE),
        ],
        experiment_name=os.environ.get("GITHUB_SHA", "local")[:12],
    )
```

**Note for the implementer:** `research_note_for` does not exist yet. Add it to `src/era/cli.py` in this task as a plain function returning `(ResearchNote, list[str])` — the CLI command then calls it too, so the command and the eval exercise the same path.

- [ ] **Step 2: Write the workflow**

`.github/workflows/evals.yml`:
```yaml
name: evals
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  judged:
    runs-on: ubuntu-latest
    # Never gates a merge: this runs after merge and is allowed to fail.
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-groups --locked
      - run: uv run python evals/tier2/run_experiment.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          VOYAGE_API_KEY: ${{ secrets.VOYAGE_API_KEY }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          OPIK_API_KEY: ${{ secrets.OPIK_API_KEY }}
          OPIK_WORKSPACE: ${{ secrets.OPIK_WORKSPACE }}
          EDGAR_USER_AGENT: ${{ secrets.EDGAR_USER_AGENT }}
```

- [ ] **Step 3: Write the README**

Cover, in this order: the one-paragraph pitch; a real generated note excerpt; quickstart (`uv sync`, `era ingest AAPL`, `era research AAPL`); the architecture section described below; the measured cost and latency per run; the design decisions from the spec's "worth defending" list; and the stated limits — US issuers only, ~20–30 indexed tickers, heuristic item parsing.

**The architecture section is the most important part of this README and has a
specific shape.** It is written for a reader who has never seen the code and wants
to understand how the system *behaves*, not how it is factored.

- A **mermaid diagram at the level of capabilities, not functions.** Nodes are
  things the system does in plain language — "find the company's filings",
  "pull exact figures from XBRL", "research each section from its own sources",
  "check every claim against the filing", "assemble the note" — not module or
  function names. One diagram for the whole flow, readable at a glance.
- Under it, a **worked example following one real ticker end to end**: what goes
  in, what each stage produces, what the reader would actually see. Show a real
  claim, the chunk it cites, and the accession that chunk came from, so the
  citation chain is concrete rather than described.
- Then a short **"what happens when something is missing"** walkthrough — a filing
  whose Item 1A cannot be located, a figure that disagrees with XBRL — showing how
  coverage reporting and the verifier respond. The failure paths are what make the
  design defensible; describing only the happy path hides the actual engineering.

Function-by-function documentation belongs in docstrings, not here.

- [ ] **Step 4: Verify everything**

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q
```
Expected: all green.

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "feat: add Tier 2 judged evals and README"
git branch -M main
git remote add origin github-personal:berkaykoklu/equity-research-agent.git
git push -u origin main
```

---

## Self-Review

**Spec coverage.** EDGAR-only sourcing → Tasks 4–5, 7. Fixed five-section note → Task 3, 12. Per-section retrieval with shared facts card → Tasks 7, 12. Coverage philosophy at all three layers → Tasks 6, 7, 13. Deterministic verifier shared with evals → Tasks 11, 16. Bounded retry → Task 13. Python assembly → Task 14. Provider factory → Task 12. One model for drafting, cheap judge → Tasks 12, 17. Tier 1 gating / Tier 2 non-gating → Tasks 16, 17. No-recommendation enforcement → Tasks 11, 14, 16. Opik spike before graph work → Task 2, gating Task 12. Ingest out of the request path → Task 10 (CLI only).

**Deferred to Plan 2, deliberately:** FastAPI, Next.js, Vercel deployment, Neon provisioning, stored reports, live smoke workflow, `research_note_for` wiring into an API. Two spec items land in Plan 2 rather than here: the live-EDGAR smoke job on merges to main, and open on-demand generation.

**Known follow-up inside this plan:** `tracked_run` returns `cost_usd=0.0` until Task 2's spike confirms whether Opik captures token usage through the LangGraph callback path. If it does not, the implementer wires `usage_metadata` from each model response into `cost_from_usage` directly — the function and its tests already exist.

---

### Task 18: Validate the parser against real 10-K filings

**Sequencing:** runs immediately after Task 6 and **gates Tasks 8 and 10**. Chunking
and ingest both build directly on `parse_items` output; if the heuristic is wrong on
real filings, everything downstream inherits it silently.

**Why this exists.** Every defect found in Task 6 so far — table-of-contents capture,
cross-reference swallowing, entity-escaped headings, the longest-body rule inverting
on short items — was found against *synthetic* HTML modelled on filing conventions.
No real filing has ever been through this parser. Synthetic fixtures encode the
author's assumptions, which is exactly what is under test here.

**The check.** A correctly extracted item body begins with its own title. A body that
starts with something else is the signature of a TOC capture or a cross-reference
swallow. That single assertion catches three of Task 6's four Criticals.

**Files:**
- Create: `tests/edgar/test_sections_live.py`, `docs/parser-validation.md`

**Interfaces:**
- Consumes: `EdgarClient` (Task 4), `resolve_cik` / `latest_filings` (Task 5), `parse_items` (Task 6)
- Produces: a committed validation report; no new importable API

- [ ] **Step 1: Write the live validation test**

Network-dependent, so it is skip-guarded exactly like the pgvector integration test —
CI stays offline and deterministic, and this runs deliberately.

`tests/edgar/test_sections_live.py`:
```python
"""Validate the item parser against real filings.

Skipped unless ERA_LIVE_EDGAR=1. Everything else in the suite runs offline;
this one deliberately hits EDGAR, because synthetic fixtures only ever encode
the assumptions of whoever wrote them.
"""

import os
from pathlib import Path

import pytest

from era.edgar.client import EdgarClient
from era.edgar.filings import latest_filings, resolve_cik
from era.edgar.sections import parse_items

USER_AGENT = os.environ.get("EDGAR_USER_AGENT", "")

pytestmark = pytest.mark.skipif(
    os.environ.get("ERA_LIVE_EDGAR") != "1" or "@" not in USER_AGENT,
    reason="set ERA_LIVE_EDGAR=1 and a contactable EDGAR_USER_AGENT to run this",
)

CACHE_DIR = Path(".cache/edgar-live")

# A correctly extracted body opens with its own title. Anything else means the
# parser captured a table-of-contents line or ran past a cross-reference.
EXPECTED_OPENING = {
    "1": ("business",),
    "1A": ("risk factor",),
    "7": ("management", "discussion"),
}

# Large caps whose filings genuinely contain all three items, plus deliberate
# awkwardness: BRK-B files unconventionally, and the banks and energy names use
# very different document generators from the tech names.
TICKERS = [
    "AAPL", "MSFT", "BRK-B", "JPM", "XOM",
    "KO", "JNJ", "PG", "WMT", "CVX",
    "MRK", "T", "VZ", "PFE", "INTC",
    "CSCO", "BA", "CAT", "GE", "DIS",
]

OPENING_WINDOW = 120


def _annual_report_html(client: EdgarClient, ticker: str) -> str:
    cik = resolve_cik(client, ticker)
    filings = latest_filings(client, cik)
    annual = next(f for f in filings if f.form == "10-K")
    return client.get_text(annual.primary_document_url)


def _opening_matches(item: str, body: str) -> bool:
    opening = body[:OPENING_WINDOW].lower()
    return any(word in opening for word in EXPECTED_OPENING[item])


@pytest.fixture(scope="module")
def results() -> dict[str, dict[str, str]]:
    """Parse every ticker once; the client's disk cache makes reruns cheap."""
    client = EdgarClient(user_agent=USER_AGENT, cache_dir=CACHE_DIR)
    collected: dict[str, dict[str, str]] = {}
    for ticker in TICKERS:
        parsed = parse_items(_annual_report_html(client, ticker))
        collected[ticker] = parsed.items
    return collected


def test_every_extracted_item_opens_with_its_own_title(results) -> None:
    failures: list[str] = []
    for ticker, items in results.items():
        for item, body in items.items():
            if not _opening_matches(item, body):
                failures.append(
                    f"{ticker} item {item}: opens with {body[:80]!r}"
                )

    assert not failures, "\n".join(failures)


def test_every_ticker_yields_all_three_items(results) -> None:
    incomplete = {
        ticker: sorted(set(EXPECTED_OPENING) - set(items))
        for ticker, items in results.items()
        if set(EXPECTED_OPENING) - set(items)
    }

    assert not incomplete, f"items not found: {incomplete}"


def test_no_item_body_is_implausibly_short(results) -> None:
    # Item 1A on a company this size is always substantial. A body of a few
    # dozen characters means a table-of-contents line won the length contest.
    too_short = [
        f"{ticker} item {item}: {len(body)} chars"
        for ticker, items in results.items()
        for item, body in items.items()
        if len(body) < 500
    ]

    assert not too_short, "\n".join(too_short)
```

- [ ] **Step 2: Run it and record what actually happens**

```bash
ERA_LIVE_EDGAR=1 uv run --directory "<repo>" pytest tests/edgar/test_sections_live.py -v
```

Expect this to take several minutes on the first run — twenty 10-K documents, several
megabytes each, fetched under a 10 req/s throttle. Subsequent runs read the disk cache.

**Do not adjust the assertions to make them pass.** Failures here are the finding.
Record exactly which tickers fail, on which items, and what the body opened with.

- [ ] **Step 3: Write `docs/parser-validation.md`**

A table of ticker × item with the character count and the first 60 characters of each
body, plus a pass/fail column and a short paragraph on what the failures have in
common. This is committed evidence that the heuristic was measured rather than
assumed, and it is the thing to re-run whenever the parser changes.

- [ ] **Step 4: Report findings rather than patching blindly**

If tickers fail, report them with the diagnosis. Parser changes are a separate,
reviewed round — this task measures, it does not fix.

- [ ] **Step 5: Commit**

```bash
git -C "<repo>" add tests/edgar/test_sections_live.py docs/parser-validation.md
git -C "<repo>" commit -m "test: validate item parser against twenty real 10-K filings"
```
