import pytest
from pydantic import ValidationError

from era.report.schema import (
    ChunkRef,
    Claim,
    Coverage,
    FactRef,
    ResearchNote,
    Section,
    SectionName,
    revalidated,
)


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
        tag="Revenues",
        fiscal_period="FY2024",
        value=391_035_000_000.0,
        accession="0000320193-24-000123",
    )
    assert fact.value == 391_035_000_000.0


def test_revalidated_re_raises_on_an_invalid_update() -> None:
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=[
            Claim(
                text="Competition is intense.",
                chunks=[ChunkRef(accession="0000320193-24-000123", chunk_id=1)],
            )
        ],
    )
    with pytest.raises(ValidationError):
        revalidated(section, claims=())


def test_revalidated_round_trips_a_valid_update() -> None:
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=[
            Claim(
                text="Competition is intense.",
                chunks=[ChunkRef(accession="0000320193-24-000123", chunk_id=1)],
            )
        ],
    )
    updated = revalidated(section, unavailable_reason=None)
    assert updated.claims == section.claims
    assert updated is not section


def test_research_note_rejects_coverage_that_contradicts_its_sections() -> None:
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=[
            Claim(
                text="Competition is intense.",
                chunks=[ChunkRef(accession="0000320193-24-000123", chunk_id=1)],
            )
        ],
    )
    with pytest.raises(ValidationError):
        ResearchNote(
            ticker="AAPL",
            cik="0000320193",
            sections=(section,),
            coverage=Coverage(sections_available=0, sections_total=1),
        )


# --- guards the final review found untested --------------------------------
#
# Each of these mutants survived the whole suite: the validator could be
# deleted and nothing objected. They are the schema half of the honest-coverage
# guarantee, so an unenforced one is a silent hole rather than a style issue.


def test_an_unavailable_section_must_state_why() -> None:
    # "A gap is reported, never guessed" depends on the reason existing.
    with pytest.raises(ValidationError):
        Section(name=SectionName.RISK_FACTORS, available=False)


def test_a_blank_reason_does_not_count_as_stating_why() -> None:
    for blank in ("", "   ", "\n"):
        with pytest.raises(ValidationError):
            Section(name=SectionName.RISK_FACTORS, available=False, unavailable_reason=blank)


def test_coverage_totals_must_match_the_sections_attached() -> None:
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(Claim(text="A risk.", chunks=(ChunkRef(accession="acc", chunk_id=0),)),),
    )

    with pytest.raises(ValidationError):
        ResearchNote(
            ticker="AAPL",
            cik="0000320193",
            sections=(section,),
            coverage=Coverage(sections_available=1, sections_total=5),  # 1 section attached
            accessions=("acc",),
        )

    with pytest.raises(ValidationError):
        ResearchNote(
            ticker="AAPL",
            cik="0000320193",
            sections=(section,),
            coverage=Coverage(sections_available=0, sections_total=1),  # it *is* available
            accessions=("acc",),
        )


def test_a_negative_chunk_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkRef(accession="acc", chunk_id=-1)


def test_an_unexpected_field_on_a_claim_is_rejected() -> None:
    # These are LLM structured-output targets. A silently dropped or invented
    # key should be an error, not something that reads as model drift later.
    with pytest.raises(ValidationError):
        Claim(
            text="A claim.",
            chunks=(ChunkRef(accession="acc", chunk_id=0),),
            confidence=0.9,  # type: ignore[call-arg]
        )


def test_a_note_must_state_which_filings_are_in_scope() -> None:
    # No default: an empty set means "no filing is in scope", which makes the
    # verifier reject every citation as foreign. That must be a deliberate
    # statement, not something a caller can omit by accident.
    with pytest.raises(ValidationError):
        ResearchNote(  # type: ignore[call-arg]
            ticker="AAPL",
            cik="0000320193",
            sections=(),
            coverage=Coverage(sections_available=0, sections_total=0),
        )
