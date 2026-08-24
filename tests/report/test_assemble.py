from era.report.assemble import render_markdown
from era.report.schema import (
    ChunkRef,
    Claim,
    Coverage,
    FactRef,
    ResearchNote,
    Section,
    SectionName,
)
from era.verify.checks import no_recommendation_language

ACCESSION = "0000320193-24-000123"

NOTE = ResearchNote(
    ticker="AAPL",
    cik="0000320193",
    sections=(
        Section(
            name=SectionName.RISK_FACTORS,
            claims=(
                Claim(
                    text="Supply chain concentration is a material risk.",
                    chunks=(ChunkRef(accession=ACCESSION, chunk_id=4),),
                ),
            ),
        ),
        Section(
            name=SectionName.BUSINESS_OVERVIEW,
            available=False,
            unavailable_reason="Item 1 boundary not found",
        ),
    ),
    # Coverage must describe the sections actually attached -- the schema
    # rejects a note whose counts disagree with its own contents.
    coverage=Coverage(
        sections_available=1,
        sections_total=2,
        metrics_resolved=("revenue",),
        metrics_missing=("total_debt",),
    ),
    accessions=(ACCESSION,),
    cost_usd=0.1331,
    latency_seconds=12.5,
)


def test_renders_claims_with_their_citations() -> None:
    out = render_markdown(NOTE)

    assert "Supply chain concentration is a material risk." in out
    assert f"{ACCESSION}#4" in out


def test_states_why_a_section_is_unavailable() -> None:
    out = render_markdown(NOTE)

    assert "Item 1 boundary not found" in out


def test_reports_coverage_and_run_cost() -> None:
    out = render_markdown(NOTE)

    assert "1 of 2" in out
    assert "total_debt" in out
    assert "$0.1331" in out


def test_carries_the_not_advice_notice() -> None:
    assert "not investment advice" in render_markdown(NOTE).lower()


def test_the_rendered_note_passes_the_no_advice_check() -> None:
    # The Tier 1 gate runs this over the whole rendered note, footer included.
    # An earlier draft of the disclaimer said "not a recommendation to buy or
    # sell", which tripped the checker on the one sentence written to keep the
    # report honest -- the merge gate would have failed on its own disclaimer.
    assert no_recommendation_language(render_markdown(NOTE)) == []


def test_sections_render_in_schema_order_not_insertion_order() -> None:
    # NOTE lists risk factors first; business overview is declared earlier in
    # SectionName and must therefore render first. A reader should get the same
    # document order whatever order the graph's parallel branches finished in.
    out = render_markdown(NOTE)

    assert out.index("Business overview") < out.index("Risk factors")


def test_a_fact_citation_names_its_tag_and_period() -> None:
    note = ResearchNote(
        ticker="AAPL",
        cik="0000320193",
        sections=(
            Section(
                name=SectionName.FINANCIAL_HEALTH,
                claims=(
                    Claim(
                        text="Revenue was 391,035 million USD.",
                        facts=(
                            FactRef(
                                tag="Revenues",
                                fiscal_period="FY2024",
                                value=391_035_000_000.0,
                                accession=ACCESSION,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        coverage=Coverage(sections_available=1, sections_total=1),
        accessions=(ACCESSION,),
    )

    assert "Revenues FY2024" in render_markdown(note)


def test_renders_a_note_whose_every_section_is_unavailable() -> None:
    # The honest-failure path: a filing nothing could be parsed from still
    # produces a readable document that says so, rather than an empty file.
    note = ResearchNote(
        ticker="GE",
        cik="0000040545",
        sections=(
            Section(
                name=SectionName.RISK_FACTORS,
                available=False,
                unavailable_reason="no Item 1A heading found in the filing",
            ),
        ),
        coverage=Coverage(sections_available=0, sections_total=1),
    )

    out = render_markdown(note)

    assert "no Item 1A heading found" in out
    assert "0 of 1" in out
    assert "none recorded" in out
