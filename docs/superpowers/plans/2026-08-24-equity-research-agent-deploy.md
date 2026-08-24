# equity-research-agent — Plan 2: public deployment

**Goal:** a public URL a hiring manager can click, showing real generated notes
with working citations.

**Constraint carried from the spec:** the demo **serves pre-generated notes and
never generates on demand**. At ~$0.09 per run an open endpoint is ~54 visitors
from draining the remaining budget. Notes are produced deliberately from the
CLI and stored; the site reads them.

**Stack:** Next.js on Vercel for the UI, a Vercel Python Function for the read
API, Neon for storage (already provisioned, already holds the chunks).

---

### Task P1: Store generated notes

**Files:** `src/era/index/notes.py`, `tests/index/test_notes.py`; modify
`src/era/cli.py`

A `notes` table keyed by ticker holding the note JSON, the rendered markdown,
and when it was generated. `NoteStore` Protocol, `PgNoteStore`, and an
in-memory fake — same pattern as `ChunkStore`, so the API layer and the tests
never depend on Postgres being reachable.

`era research TICKER --save` writes the note it just produced. Without the flag
nothing is stored: publishing is a deliberate act, not a side effect of running.

**Done when:** a note round-trips through the store and re-verifies from the
retrieved JSON, offline, against the fake.

---

### Task P2: Read-only API

**Files:** `api/index.py`, `tests/test_api.py`

Two endpoints, both reads:

- `GET /api/notes` — ticker, company, generated-at for every stored note
- `GET /api/notes/{ticker}` — one note, JSON plus rendered markdown

**No endpoint triggers generation.** That is the budget guarantee, and it
belongs in a test: assert the app exposes no route that calls the graph.

**Done when:** both endpoints are covered by offline tests against the fake
store, and the no-generation test passes.

---

### Task P3: The page

**Files:** `web/` (Next.js, App Router, TypeScript strict)

- Index: the companies with stored notes, each with when it was generated.
- Note page: the rendered note, with **every citation a link to the SEC filing
  it cites**. That is the whole point of the site — a reader can click a claim
  and land on the source document.
- Coverage and cost shown plainly, including sections reported unavailable.
- A short "how this works" panel: words and numbers travel separately, the
  checker is code, gaps are reported rather than guessed.

**Done when:** it builds clean, renders a stored note, and the citation links
resolve to real SEC URLs.

---

### Task P4: Deploy

**Files:** `vercel.json`; Vercel project settings

- Push the repository to GitHub (requires approval).
- `vercel login` (interactive, run by Berkay).
- Link the project, set `DATABASE_URL` in Vercel's environment.
- Deploy, then confirm the live URL serves a real note with working citations.

**Done when:** a public URL renders a stored note and its citations resolve.

**Not done here:** live generation, authentication, or any write path. The
deployed app is read-only by construction.
