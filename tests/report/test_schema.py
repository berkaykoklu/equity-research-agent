import pytest
from pydantic import ValidationError

from era.report.schema import ChunkRef, Claim, FactRef, Section, SectionName


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
