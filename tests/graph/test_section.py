"""Offline tests for one section's research step. No API calls."""

from typing import Any

from tests.fakes import FakeEmbedder, InMemoryChunkStore

from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.graph.section import SectionDraft, SectionRequest, research_section
from era.index.chunking import Chunk
from era.report.schema import SectionName

ACC = "0000320193-24-000123"
OTHER = "9999999999-24-000001"

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


class ScriptedModel:
    """Returns a fixed draft, and records the prompt it was given.

    Deliberately not a real chat model: this task's job is retrieval, prompt
    construction and translation back into claims, none of which needs a
    network call to test.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.prompt: str | None = None

    def with_structured_output(self, schema: Any, **_: Any) -> Any:
        outer = self

        class _Runnable:
            def invoke(self, prompt: Any, config: Any = None) -> Any:
                outer.prompt = str(prompt)
                return schema(**outer.payload)

        return _Runnable()


def _store(*, accession: str = ACC, item: str = "1A") -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    chunk = Chunk(
        chunk_id=0,
        accession=accession,
        item=item,
        start=0,
        text="Supply chain concentration is a material risk.",
    )
    store.upsert([chunk], FakeEmbedder().embed([chunk.text]))
    return store


def _request(**overrides: Any) -> SectionRequest:
    base: dict[str, Any] = {
        "name": SectionName.RISK_FACTORS,
        "ticker": "AAPL",
        "accession": ACC,
        "item_filter": "1A",
        "k": 4,
    }
    base.update(overrides)
    return SectionRequest(**base)


def test_returns_a_section_of_claims_with_citations() -> None:
    model = ScriptedModel(
        {
            "claims": [
                {
                    "text": "Supply chain concentration is a material risk.",
                    "chunk_ids": [0],
                    "metrics": [],
                }
            ]
        }
    )

    section = research_section(_request(), model, _store(), FakeEmbedder(), FACTS)

    assert section.available is True
    assert section.claims[0].chunks[0].chunk_id == 0
    assert section.claims[0].chunks[0].accession == ACC


def test_marks_the_section_unavailable_when_retrieval_finds_nothing() -> None:
    model = ScriptedModel({"claims": []})

    section = research_section(_request(), model, InMemoryChunkStore(), FakeEmbedder(), FACTS)

    assert section.available is False
    assert "no indexed content" in (section.unavailable_reason or "")


def test_retrieval_is_scoped_to_this_notes_filing() -> None:
    # The store holds many companies. Task 11's verifier rejects a citation
    # outside the note's filings, so retrieving unscoped would make every
    # claim unciteable and burn the retry budget on unfixable violations.
    model = ScriptedModel({"claims": []})

    section = research_section(_request(), model, _store(accession=OTHER), FakeEmbedder(), FACTS)

    assert section.available is False
    assert "no indexed content" in (section.unavailable_reason or "")


def test_a_named_metric_is_filled_in_from_the_filings_own_data() -> None:
    # The model names "revenue"; the tag, period, value and accession all come
    # from NormalizedFacts. The model never types a figure.
    model = ScriptedModel(
        {"claims": [{"text": "Revenue grew.", "chunk_ids": [0], "metrics": ["revenue"]}]}
    )

    section = research_section(_request(), model, _store(), FakeEmbedder(), FACTS)

    fact = section.claims[0].facts[0]
    assert (fact.tag, fact.fiscal_period, fact.value) == ("Revenues", "FY2024", 391_035_000_000.0)
    assert fact.accession == ACC


def test_a_metric_name_that_is_not_in_the_facts_card_is_dropped() -> None:
    model = ScriptedModel(
        {
            "claims": [
                {"text": "Debt is manageable.", "chunk_ids": [0], "metrics": ["total_debt"]},
            ]
        }
    )

    section = research_section(_request(), model, _store(), FakeEmbedder(), FACTS)

    # total_debt is in `missing`, so no figure exists to attach. The claim
    # survives on its chunk citation alone rather than carrying a fact that
    # would not resolve.
    assert section.claims[0].facts == ()
    assert section.claims[0].chunks[0].chunk_id == 0


def test_a_chunk_id_the_model_was_never_shown_is_dropped() -> None:
    model = ScriptedModel(
        {"claims": [{"text": "Invented citation.", "chunk_ids": [99], "metrics": []}]}
    )

    section = research_section(_request(), model, _store(), FakeEmbedder(), FACTS)

    # Nothing citable survives, so the section reports unavailable rather than
    # shipping a sentence whose citation cannot resolve.
    assert section.available is False
    assert "no citable claims" in (section.unavailable_reason or "")


def test_the_prompt_carries_the_facts_card_and_the_excerpts() -> None:
    model = ScriptedModel({"claims": [{"text": "A claim.", "chunk_ids": [0], "metrics": []}]})

    research_section(_request(), model, _store(), FakeEmbedder(), FACTS)

    assert model.prompt is not None
    assert "Financial facts" in model.prompt
    assert "chunk 0" in model.prompt
    assert "Supply chain concentration" in model.prompt


def test_a_complaint_is_appended_on_a_retry() -> None:
    model = ScriptedModel({"claims": [{"text": "A claim.", "chunk_ids": [0], "metrics": []}]})

    research_section(
        _request(complaint="unsupported_figure: 4200"),
        model,
        _store(),
        FakeEmbedder(),
        FACTS,
    )

    assert model.prompt is not None
    assert "unsupported_figure: 4200" in model.prompt


def test_an_unusable_response_shape_degrades_to_unavailable() -> None:
    class WrongShapeModel(ScriptedModel):
        def with_structured_output(self, schema: Any, **_: Any) -> Any:
            class _Runnable:
                def invoke(self, prompt: Any, config: Any = None) -> Any:
                    return "not a SectionDraft"

            return _Runnable()

    section = research_section(_request(), WrongShapeModel({}), _store(), FakeEmbedder(), FACTS)

    assert section.available is False
    assert "unusable response shape" in (section.unavailable_reason or "")


def test_duplicate_citations_are_collapsed() -> None:
    model = ScriptedModel(
        {
            "claims": [
                {
                    "text": "Stated twice.",
                    "chunk_ids": [0, 0],
                    "metrics": ["revenue", "revenue"],
                }
            ]
        }
    )

    section = research_section(_request(), model, _store(), FakeEmbedder(), FACTS)

    assert len(section.claims[0].chunks) == 1
    assert len(section.claims[0].facts) == 1


def test_section_draft_defaults_to_no_claims() -> None:
    assert SectionDraft().claims == []
