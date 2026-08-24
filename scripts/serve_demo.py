"""Serve the site's API from a stored fixture — no database, no API keys.

    uv run uvicorn scripts.serve_demo:app --port 8000

Exists so the whole site can be run from a fresh clone. The note it serves is
a real one, generated against Apple's FY2025 10-K and checked by the same
verifier that gates CI; it is the fixture the Tier 1 evals run against, so
what a reader sees locally is what actually passed.

This is still read-only. It swaps where notes are read from and nothing else,
and it is not part of the deployed function — production reads Postgres.
"""

from __future__ import annotations

import json
from pathlib import Path

from api.index import app, get_store

from era.index.notes import InMemoryNoteStore, NoteStore
from era.report.assemble import render_markdown
from era.report.schema import ResearchNote

FIXTURE = Path(__file__).resolve().parent.parent / "evals" / "fixtures" / "note_aapl.json"


def _seeded() -> NoteStore:
    note = ResearchNote.model_validate(json.loads(FIXTURE.read_text()))
    store = InMemoryNoteStore()
    store.save(note, render_markdown(note))
    return store


# Built once at import, not per request: the override is called on every call,
# and re-reading the fixture each time would make the demo slower than the
# database it stands in for.
_STORE = _seeded()

app.dependency_overrides[get_store] = lambda: _STORE
