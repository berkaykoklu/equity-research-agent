"""Tier 1: the merge gate.

Runs the *same* verifier the graph uses at runtime, against a note captured
from a real run. One implementation, two callers -- the standard that stops a
bad sentence reaching a reader is the standard that stops a bad change reaching
main.

Deliberately offline, deterministic and free: no network, no API key, no model.
A gate that costs money or fails randomly is one people learn to route around,
and a gate people route around is not a gate.

Regenerate the fixtures with a real run whenever the pipeline changes shape;
they are evidence, and stale evidence is worse than none.
"""

import json
from pathlib import Path

import pytest
from tests.fakes import FakeEmbedder, InMemoryChunkStore

from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.index.chunking import Chunk
from era.report.assemble import render_markdown
from era.report.schema import ResearchNote, SectionName
from era.verify.checks import no_recommendation_language, verify_note

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TICKER = "aapl"


@pytest.fixture(scope="module")
def note() -> ResearchNote:
    return ResearchNote.model_validate_json((FIXTURES / f"note_{TICKER}.json").read_text())


@pytest.fixture(scope="module")
def store() -> InMemoryChunkStore:
    rows = json.loads((FIXTURES / f"chunks_{TICKER}.json").read_text())
    chunks = [
        Chunk(
            chunk_id=row["chunk_id"],
            accession=row["accession"],
            item=row["item"],
            start=row.get("start", 0),
            text=row["text"],
        )
        for row in rows
    ]
    store = InMemoryChunkStore()
    store.upsert(chunks, FakeEmbedder().embed([c.text for c in chunks]))
    return store


@pytest.fixture(scope="module")
def facts() -> NormalizedFacts:
    raw = json.loads((FIXTURES / f"facts_{TICKER}.json").read_text())
    return NormalizedFacts(
        metrics={name: MetricValue(**value) for name, value in raw["metrics"].items()},
        missing=tuple(raw["missing"]),
    )


def test_every_claim_in_a_real_note_verifies(
    note: ResearchNote, store: InMemoryChunkStore, facts: NormalizedFacts
) -> None:
    violations = verify_note(note, store, facts)

    assert violations == [], (
        f"{len(violations)} unverifiable claims, first three: "
        f"{[(v.kind, v.detail, v.section) for v in violations[:3]]}"
    )


def test_every_section_is_populated_or_explains_itself(note: ResearchNote) -> None:
    for section in note.sections:
        assert section.available or section.unavailable_reason, section.name


def test_every_claim_carries_at_least_one_source(note: ResearchNote) -> None:
    # The schema enforces this on construction; asserting it here means a
    # future loosening of the schema fails the gate rather than passing quietly.
    for section in note.sections:
        for index, claim in enumerate(section.claims):
            assert claim.chunks or claim.facts, f"{section.name} claim {index} is unsourced"


def test_the_note_recommends_nothing(note: ResearchNote) -> None:
    # Runs over the *rendered* note, disclaimer included -- an earlier draft of
    # the disclaimer said "not a recommendation to buy or sell" and failed this
    # gate on the one sentence written to keep the report honest.
    assert no_recommendation_language(render_markdown(note)) == []


def test_the_rendered_note_carries_the_not_advice_notice(note: ResearchNote) -> None:
    assert "not investment advice" in render_markdown(note).lower()


def test_coverage_describes_the_sections_actually_present(note: ResearchNote) -> None:
    assert note.coverage.sections_total == len(note.sections)
    assert note.coverage.sections_available == sum(1 for s in note.sections if s.available)


def test_the_fixture_is_a_real_run_worth_gating_on(note: ResearchNote) -> None:
    # A fixture that had degraded to nothing would pass every check above
    # vacuously. This asserts the evidence still has substance in it.
    assert note.coverage.sections_available == len(SectionName)
    assert sum(len(section.claims) for section in note.sections) >= 10
    assert note.cost_usd > 0.0
