from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SectionName(StrEnum):
    """Report sections, in the order they are rendered in the final note."""

    BUSINESS_OVERVIEW = "business_overview"
    FINANCIAL_HEALTH = "financial_health"
    RISK_FACTORS = "risk_factors"
    RECENT_DEVELOPMENTS = "recent_developments"
    VALUATION_CONTEXT = "valuation_context"


class ChunkRef(BaseModel):
    """Points at one indexed filing chunk the verifier can resolve exactly."""

    model_config = ConfigDict(frozen=True)

    accession: str
    chunk_id: int


class FactRef(BaseModel):
    """Points at one exact XBRL fact the verifier can compare byte-for-byte."""

    model_config = ConfigDict(frozen=True)

    tag: str
    fiscal_period: str
    value: float
    accession: str


class Claim(BaseModel):
    """A single verifiable statement in the report, backed by its sources."""

    model_config = ConfigDict(frozen=True)

    text: str
    chunks: tuple[ChunkRef, ...] = ()
    facts: tuple[FactRef, ...] = ()

    @model_validator(mode="after")
    def require_a_source(self) -> "Claim":
        # An uncited claim can't be checked, so it can't exist in this schema.
        if not self.chunks and not self.facts:
            raise ValueError("every claim must cite at least one chunk or fact")
        return self


class Section(BaseModel):
    """One section of the report, or a stated reason it could not be produced."""

    model_config = ConfigDict(frozen=True)

    name: SectionName
    claims: tuple[Claim, ...] = ()
    available: bool = True
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def check_availability(self) -> "Section":
        # available/claims/reason must stay consistent: a rendered section needs
        # content, and a skipped section must explain itself to the reader.
        if self.available and not self.claims:
            raise ValueError("an available section must contain at least one claim")
        if not self.available and not self.unavailable_reason:
            raise ValueError("an unavailable section must state why")
        return self


class Coverage(BaseModel):
    """Summarizes how much of the report was actually backed by evidence."""

    model_config = ConfigDict(frozen=True)

    sections_available: int
    sections_total: int
    metrics_resolved: tuple[str, ...] = ()
    metrics_missing: tuple[str, ...] = ()


class ResearchNote(BaseModel):
    """The finished, citable research note for one ticker."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    cik: str
    sections: tuple[Section, ...]
    coverage: Coverage
    accessions: tuple[str, ...] = Field(default=())
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
