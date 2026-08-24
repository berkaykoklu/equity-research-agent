"""Read-only HTTP surface over stored notes.

Every route here is a read. Nothing in this module can produce a note, and
that is a design guarantee rather than an oversight: generating one costs real
money and takes ~80 seconds, so an endpoint that could generate is an endpoint
a stranger can make expensive. Notes are written deliberately from the CLI
(`era research TICKER --save`); this serves what is already stored.

`tests/test_api.py` enforces that guarantee two ways — no route accepts a
write method, and importing this module never pulls `era.graph` into memory.

Deliberately not using `era.config.Settings`: that requires `EDGAR_USER_AGENT`,
and needing an SEC contact address to hand back a stored document would be a
coupling with no reason behind it. The only configuration a reader needs is
where the notes live.
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from era.index.notes import NoteStore, PgNoteStore, StoredNote
from era.report.schema import ResearchNote

app = FastAPI(
    title="equity-research-agent",
    description="Cited equity research notes from SEC filings. Read-only.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _pg_store(dsn: str) -> PgNoteStore:
    """One store per process.

    A warm serverless function reuses its connection, and `PgNoteStore`
    already reconnects when Neon suspends the compute underneath it.
    """
    return PgNoteStore(dsn=dsn)


def get_store() -> NoteStore:
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        # 503, not 500: the service is fine, its storage is not configured.
        # Tests override this dependency, so no test ever needs a database.
        raise HTTPException(status_code=503, detail="storage is not configured")
    return _pg_store(dsn)


StoreDep = Annotated[NoteStore, Depends(get_store)]


class NoteSummary(BaseModel):
    """What the index needs. No company name: the note records ticker and CIK,
    and inventing a display name here would be asserting something unfiled."""

    ticker: str
    cik: str
    generated_at: datetime
    sections_available: int
    sections_total: int


class NoteDetail(NoteSummary):
    note: ResearchNote
    markdown: str


def _summary(row: StoredNote) -> NoteSummary:
    return NoteSummary(
        ticker=row.ticker,
        cik=row.cik,
        generated_at=row.generated_at,
        sections_available=row.note.coverage.sections_available,
        sections_total=row.note.coverage.sections_total,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    """Boots without touching storage, so a deploy can be confirmed separately
    from whether the database is reachable."""
    return {"status": "ok"}


@app.get("/api/notes", response_model=list[NoteSummary])
def list_notes(store: StoreDep) -> list[NoteSummary]:
    return [_summary(row) for row in store.list_tickers()]


@app.get("/api/notes/{ticker}", response_model=NoteDetail)
def get_note(ticker: str, store: StoreDep) -> NoteDetail:
    row = store.get(ticker)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no stored note for {ticker.upper()}")
    summary = _summary(row)
    return NoteDetail(
        **summary.model_dump(),
        note=row.note,
        markdown=row.markdown,
    )
