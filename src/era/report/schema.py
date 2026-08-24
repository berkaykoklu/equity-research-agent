from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SectionName(StrEnum):
    """Report sections, in the order they are rendered in the final note."""

    BUSINESS_OVERVIEW = "business_overview"
    FINANCIAL_HEALTH = "financial_health"
    RISK_FACTORS = "risk_factors"
    RECENT_DEVELOPMENTS = "recent_developments"
    VALUATION_CONTEXT = "valuation_context"


class ChunkRef(BaseModel):
    """Points at one indexed filing chunk the verifier can resolve exactly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accession: str = Field(min_length=1)
    chunk_id: int = Field(ge=0)


class FactRef(BaseModel):
    """Points at one XBRL fact; the verifier matches its filed value within tolerance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tag: str = Field(min_length=1)
    # Canonical form is FY<4-digit year>, e.g. "FY2024" — the verifier looks up
    # facts by this exact string, so any other shape silently fails to resolve.
    fiscal_period: str = Field(min_length=1)
    value: float
    accession: str = Field(min_length=1)


class Claim(BaseModel):
    """A single verifiable statement in the report, backed by its sources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
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

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SectionName
    claims: tuple[Claim, ...] = ()
    available: bool = True
    unavailable_reason: str | None = None

    @field_validator("unavailable_reason")
    @classmethod
    def reject_blank_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("unavailable_reason must not be blank")
        return value

    @model_validator(mode="after")
    def check_availability(self) -> "Section":
        # Only two rules are enforced: a rendered section needs content, and a
        # skipped section must explain itself. available=True may still carry a
        # stray unavailable_reason — that combination is accepted, not rejected.
        if self.available and not self.claims:
            raise ValueError("an available section must contain at least one claim")
        if not self.available and not self.unavailable_reason:
            raise ValueError("an unavailable section must state why")
        return self


class Coverage(BaseModel):
    """Summarizes how much of the report was actually backed by evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sections_available: int
    sections_total: int
    metrics_resolved: tuple[str, ...] = ()
    metrics_missing: tuple[str, ...] = ()


class ResearchNote(BaseModel):
    """The finished, citable research note for one ticker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    cik: str = Field(min_length=1)
    sections: tuple[Section, ...]
    coverage: Coverage
    # Required, deliberately no default. An empty tuple is not "unset": the
    # verifier reads it as "no filing is in scope" and rejects EVERY citation
    # as foreign -- silently, and phrased as a content violation, so it would
    # drive the retry loop to exhaustion rather than surfacing as a config
    # error. Making it required removes the whole class of bug.
    accessions: tuple[str, ...]
    cost_usd: float = 0.0
    latency_seconds: float = 0.0

    @model_validator(mode="after")
    def coverage_must_match_sections(self) -> "ResearchNote":
        # Coverage is the honesty metric of the whole report: it must describe
        # the sections actually attached, not a number someone typed separately.
        available_count = sum(1 for section in self.sections if section.available)
        if self.coverage.sections_available != available_count:
            raise ValueError("coverage.sections_available must match available sections")
        if self.coverage.sections_total != len(self.sections):
            raise ValueError("coverage.sections_total must match len(sections)")
        return self


def revalidated[ModelT: BaseModel](model: ModelT, **updates: object) -> ModelT:
    """Return a copy with updates applied, re-running validators.

    `model_copy(update=...)` skips validation, so an invariant this schema
    promises could be broken silently. Every incremental update goes through here.
    """
    return type(model).model_validate({**model.model_dump(), **updates})
