"""Offline tests for the research graph. No API calls."""

from typing import Any

from tests.fakes import FakeEmbedder, InMemoryChunkStore

from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.graph.build import build_research_graph
from era.index.chunking import Chunk
from era.report.schema import SectionName

ACC = "0000320193-24-000123"

FACTS = NormalizedFacts(
    metrics={
        "revenue": MetricValue(
            tag="Revenues",
            fiscal_period="FY2024",
            value=391_035_000_000.0,
            accession=ACC,
        )
    },
    missing=("total_debt",),
)

SUPPORTED = "Supply chain concentration is a material risk to operations."


class ScriptedModel:
    """Returns a fixed draft every call, counting how many calls it received."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def with_structured_output(self, schema: Any, **_: Any) -> Any:
        outer = self

        class _Runnable:
            def invoke(self, prompt: Any, config: Any = None) -> Any:
                outer.calls += 1
                return schema(**outer.payload)

        return _Runnable()


def _claim(text: str) -> dict[str, Any]:
    return {"text": text, "chunk_ids": [0, 1, 2], "metrics": []}


def _store() -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    chunks = [
        Chunk(chunk_id=i, accession=ACC, item=item, start=0, text=SUPPORTED)
        for i, item in enumerate(["1", "1A", "7"])
    ]
    store.upsert(chunks, FakeEmbedder().embed([c.text for c in chunks]))
    return store


def _run(model: ScriptedModel, store: InMemoryChunkStore, **kwargs: Any) -> Any:
    graph = build_research_graph(model, store, FakeEmbedder(), FACTS, **kwargs)
    return graph.invoke({"ticker": "AAPL", "cik": "0000320193", "accession": ACC})


def test_produces_a_note_covering_every_section() -> None:
    result = _run(ScriptedModel({"claims": [_claim(SUPPORTED)]}), _store())
    note = result["note"]

    assert {section.name for section in note.sections} == set(SectionName)
    assert note.ticker == "AAPL"


def test_sections_are_ordered_by_schema_not_by_completion() -> None:
    # Five branches finish in whatever order the runtime chooses; the note must
    # read the same every time.
    result = _run(ScriptedModel({"claims": [_claim(SUPPORTED)]}), _store())

    names = [section.name for section in result["note"].sections]
    assert names == [name for name in SectionName if name in set(names)]


def test_coverage_counts_match_the_attached_sections() -> None:
    result = _run(ScriptedModel({"claims": [_claim(SUPPORTED)]}), _store())
    note = result["note"]

    assert note.coverage.sections_total == len(note.sections)
    assert note.coverage.sections_available == sum(1 for s in note.sections if s.available)
    assert "total_debt" in note.coverage.metrics_missing


def test_a_section_with_no_indexed_content_is_marked_unavailable() -> None:
    result = _run(ScriptedModel({"claims": [_claim(SUPPORTED)]}), InMemoryChunkStore())
    note = result["note"]

    assert all(not section.available for section in note.sections)
    assert note.coverage.sections_available == 0


def test_a_failing_section_is_retried_and_the_retries_are_bounded() -> None:
    # This claim states a figure no source supports, so the verifier rejects it
    # every time and the loop can only stop by hitting its own ceiling.
    model = ScriptedModel({"claims": [_claim("The company closed 4,200 stores.")]})

    result = _run(model, _store(), max_retries=2)

    assert max(result["attempts"].values()) <= 3  # first attempt plus two retries
    assert result["complaints"]  # still unhappy, but it stopped anyway


def test_a_clean_run_never_enters_the_retry_loop() -> None:
    model = ScriptedModel({"claims": [_claim(SUPPORTED)]})

    result = _run(model, _store())

    assert result["complaints"] == {}
    assert set(result["attempts"].values()) == {1}
    assert model.calls == len(SectionName)


def test_the_note_cites_only_filings_it_actually_used() -> None:
    result = _run(ScriptedModel({"claims": [_claim(SUPPORTED)]}), _store())

    assert result["note"].accessions == (ACC,)


def test_unavailable_sections_are_not_retried() -> None:
    # Nothing was found to research, so there is no complaint the model could
    # act on -- spending retries here would be pure waste.
    model = ScriptedModel({"claims": [_claim(SUPPORTED)]})

    result = _run(model, InMemoryChunkStore(), max_retries=2)

    assert result["complaints"] == {}
    assert model.calls == 0
