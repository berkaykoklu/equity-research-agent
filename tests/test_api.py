"""Offline tests for the read API. No database, no network, no model.

The last two tests are the important ones: they are what keeps the deployed
site from being able to spend money.
"""

import subprocess
import sys
from pathlib import Path

from api.index import app, get_store
from fastapi.testclient import TestClient
from tests.fakes import FakeEmbedder, InMemoryChunkStore

from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.index.chunking import Chunk
from era.index.notes import InMemoryNoteStore, NoteStore
from era.report.assemble import render_markdown
from era.report.schema import ChunkRef, Claim, Coverage, ResearchNote, Section, SectionName
from era.verify.checks import verify_note

ROOT = Path(__file__).resolve().parent.parent
ACC = "0000320193-25-000079"
TEXT = "Supply chain concentration is a material risk to operations."

FACTS = NormalizedFacts(
    metrics={
        "revenue": MetricValue(
            tag="Revenues", fiscal_period="FY2025", value=416_161_000_000.0, accession=ACC
        )
    },
    missing=(),
)


def _note(ticker: str = "AAPL") -> ResearchNote:
    return ResearchNote(
        ticker=ticker,
        cik="0000320193",
        sections=(
            Section(
                name=SectionName.RISK_FACTORS,
                claims=(Claim(text=TEXT, chunks=(ChunkRef(accession=ACC, chunk_id=0),)),),
            ),
            # A real note carries its failures. Keeping one here means the API
            # is tested against the shape it will actually serve, not a
            # flattering one.
            Section(
                name=SectionName.BUSINESS_OVERVIEW,
                available=False,
                unavailable_reason="Item 1 boundary not found",
            ),
        ),
        coverage=Coverage(sections_available=1, sections_total=2),
        accessions=(ACC,),
        cost_usd=0.0922,
        latency_seconds=81.4,
    )


def _client(store: NoteStore) -> TestClient:
    app.dependency_overrides[get_store] = lambda: store
    return TestClient(app)


def _stocked() -> InMemoryNoteStore:
    store = InMemoryNoteStore()
    for ticker in ("KO", "AAPL"):
        note = _note(ticker)
        store.save(note, render_markdown(note))
    return store


def teardown_function() -> None:
    # The app object is module-level and shared; a leaked override would let
    # one test's fake decide another test's result.
    app.dependency_overrides.clear()


def test_the_index_lists_every_stored_note_in_order() -> None:
    response = _client(_stocked()).get("/api/notes")

    assert response.status_code == 200
    assert [row["ticker"] for row in response.json()] == ["AAPL", "KO"]


def test_the_index_reports_coverage_so_a_reader_sees_the_gaps() -> None:
    row = _client(_stocked()).get("/api/notes").json()[0]

    assert row["sections_available"] == 1
    assert row["sections_total"] == 2


def test_an_empty_store_is_an_empty_list_not_an_error() -> None:
    response = _client(InMemoryNoteStore()).get("/api/notes")

    assert response.status_code == 200
    assert response.json() == []


def test_a_note_served_over_http_still_verifies_against_its_sources() -> None:
    # The end-to-end honesty check. Serialising through JSON and back must not
    # lose the citations -- what a visitor reads has to be what the checker
    # approved, or the guarantee stops at the process boundary.
    body = _client(_stocked()).get("/api/notes/AAPL").json()

    chunk_store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=0, accession=ACC, item="1A", start=0, text=TEXT)
    chunk_store.upsert([chunk], FakeEmbedder().embed([chunk.text]))

    assert verify_note(ResearchNote.model_validate(body["note"]), chunk_store, FACTS) == []


def test_a_note_carries_its_rendered_markdown() -> None:
    body = _client(_stocked()).get("/api/notes/AAPL").json()

    assert TEXT in body["markdown"]
    assert f"{ACC}#0" in body["markdown"]


def test_lookup_is_case_insensitive_because_a_url_carries_whatever_was_typed() -> None:
    assert _client(_stocked()).get("/api/notes/aapl").status_code == 200


def test_an_unknown_ticker_is_a_404_naming_the_ticker() -> None:
    response = _client(_stocked()).get("/api/notes/NOPE")

    assert response.status_code == 404
    assert "NOPE" in response.json()["detail"]


def test_unconfigured_storage_is_503_rather_than_a_crash(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app.dependency_overrides.clear()

    assert TestClient(app).get("/api/notes").status_code == 503


def test_health_answers_without_touching_storage(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app.dependency_overrides.clear()

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- The budget guarantee ---------------------------------------------------


def test_no_route_accepts_a_write_method() -> None:
    # Generation would have to be triggered by something. A read-only method
    # surface is the first half of making that impossible.
    for route in app.routes:
        methods = getattr(route, "methods", set())
        assert methods <= {"GET", "HEAD"}, f"{getattr(route, 'path', route)} accepts {methods}"


def test_importing_the_api_never_loads_the_graph() -> None:
    # The second half, and the one that actually bites. A future edit that
    # imports the graph "just to reuse a helper" puts the whole generation
    # pipeline one function call away from a public endpoint. Checked in a
    # subprocess because sys.modules is already polluted by the rest of this
    # suite -- an in-process assertion here would be meaningless.
    probe = (
        "import sys, api.index;print(sorted(m for m in sys.modules if m.startswith('era.graph')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"the API pulled in {result.stdout.strip()}"
