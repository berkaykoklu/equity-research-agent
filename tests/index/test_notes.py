"""Offline tests for note storage. No database, no network."""

from tests.fakes import FakeEmbedder, InMemoryChunkStore

from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.index.chunking import Chunk
from era.index.notes import InMemoryNoteStore
from era.report.assemble import render_markdown
from era.report.schema import ChunkRef, Claim, Coverage, ResearchNote, Section, SectionName
from era.verify.checks import verify_note

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


def _note() -> ResearchNote:
    return ResearchNote(
        ticker="AAPL",
        cik="0000320193",
        sections=(
            Section(
                name=SectionName.RISK_FACTORS,
                claims=(Claim(text=TEXT, chunks=(ChunkRef(accession=ACC, chunk_id=0),)),),
            ),
        ),
        coverage=Coverage(sections_available=1, sections_total=1),
        accessions=(ACC,),
        cost_usd=0.0922,
        latency_seconds=81.4,
    )


def _chunk_store() -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=0, accession=ACC, item="1A", start=0, text=TEXT)
    store.upsert([chunk], FakeEmbedder().embed([chunk.text]))
    return store


def test_a_saved_note_comes_back_intact() -> None:
    store = InMemoryNoteStore()
    note = _note()

    store.save(note, render_markdown(note))
    stored = store.get("AAPL")

    assert stored is not None
    assert stored.note == note
    assert stored.cik == "0000320193"


def test_a_retrieved_note_still_verifies_against_its_sources() -> None:
    # Storage must not quietly lose the citations. Round-tripping through the
    # store and re-verifying is the only check that proves the note a visitor
    # reads is the note the checker approved.
    store = InMemoryNoteStore()
    note = _note()
    store.save(note, render_markdown(note))

    stored = store.get("AAPL")
    assert stored is not None
    assert verify_note(stored.note, _chunk_store(), FACTS) == []


def test_saving_the_same_ticker_replaces_rather_than_accumulates() -> None:
    store = InMemoryNoteStore()
    note = _note()

    store.save(note, "first")
    store.save(note, "second")

    stored = store.get("AAPL")
    assert stored is not None
    assert stored.markdown == "second"
    assert len(store.list_tickers()) == 1


def test_lookup_is_case_insensitive() -> None:
    # A URL will carry whatever a visitor typed.
    store = InMemoryNoteStore()
    store.save(_note(), "md")

    assert store.get("aapl") is not None
    assert store.get("AaPl") is not None


def test_an_unknown_ticker_returns_nothing_rather_than_raising() -> None:
    assert InMemoryNoteStore().get("NOPE") is None


def test_listing_is_ordered_and_empty_when_nothing_is_stored() -> None:
    store = InMemoryNoteStore()
    assert store.list_tickers() == []

    for ticker in ("KO", "AAPL", "JNJ"):
        note = _note()
        store.save(note.model_copy(update={"ticker": ticker}), "md")

    assert [row.ticker for row in store.list_tickers()] == ["AAPL", "JNJ", "KO"]
