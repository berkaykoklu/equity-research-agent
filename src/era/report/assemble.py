"""Turn a verified note into markdown.

Plain Python, no model. By the time a `ResearchNote` reaches here every claim
has already been checked, so assembly is string handling and needs no
intelligence. Doing it deterministically is also the point: a generative
assembler could introduce a sentence the verifier never saw, which would
defeat the guarantee the rest of the pipeline exists to provide.

This module renders what it is given and asserts nothing of its own.
"""

from era.report.schema import Claim, ResearchNote, Section, SectionName

SECTION_TITLES: dict[SectionName, str] = {
    SectionName.BUSINESS_OVERVIEW: "Business overview",
    SectionName.FINANCIAL_HEALTH: "Financial health",
    SectionName.RISK_FACTORS: "Risk factors",
    SectionName.RECENT_DEVELOPMENTS: "Recent developments",
    SectionName.VALUATION_CONTEXT: "Valuation context",
}

# Wording matters here. This sentence is the product's own statement that it
# gives no advice, and `verify.checks.no_recommendation_language` runs over the
# rendered note in the Tier 1 CI gate -- including this line. Phrasing it as
# "is not a recommendation to buy or sell" would trip that check on the very
# sentence written to keep the report honest, and fail the merge gate.
NOT_ADVICE = (
    "*This note summarises public filings. It is not investment advice, "
    "and nothing in it suggests any course of action.*"
)


def _citations(claim: Claim) -> str:
    """Every source behind one claim, in the order the claim carries them.

    A chunk cites the passage it came from; a fact cites the tag and period so
    a reader can look the figure up in the filing's own XBRL data.
    """
    refs = [f"{ref.accession}#{ref.chunk_id}" for ref in claim.chunks]
    # The value, not just the tag. Without it the citation is decorative: the
    # model is told never to write a figure itself, so a note can cite
    # "Revenues FY2025" in a sentence containing no revenue. Rendering the
    # number is what makes the numeric guarantee visible to a reader instead of
    # merely asserted.
    refs += [f"{fact.tag} {fact.fiscal_period} = {fact.value:,.0f}" for fact in claim.facts]
    return "; ".join(refs)


def _render_section(section: Section) -> list[str]:
    lines = [f"## {SECTION_TITLES[section.name]}", ""]
    if not section.available:
        # Stating the reason rather than omitting the section is the whole
        # coverage philosophy: a gap the reader can see beats a gap they cannot.
        lines += [f"*Unavailable — {section.unavailable_reason}.*", ""]
        return lines

    for claim in section.claims:
        lines += [f"{claim.text} [{_citations(claim)}]", ""]
    return lines


def render_markdown(note: ResearchNote) -> str:
    """Render a verified note. Sections appear in `SectionName` order."""
    sources = ", ".join(note.accessions) if note.accessions else "none recorded"
    lines = [
        f"# {note.ticker} — research note",
        "",
        f"CIK {note.cik} · sources: {sources}",
        "",
    ]

    by_name = {section.name: section for section in note.sections}
    for name in SectionName:
        section = by_name.get(name)
        if section is not None:
            lines += _render_section(section)

    lines += ["## Coverage", ""]
    lines += [
        f"Sections populated: {note.coverage.sections_available} of "
        f"{note.coverage.sections_total}.",
        "",
    ]
    if note.coverage.metrics_missing:
        lines += [
            "Metrics that could not be resolved from XBRL: "
            f"{', '.join(note.coverage.metrics_missing)}.",
            "",
        ]
    lines += [
        f"Run cost ${note.cost_usd:.4f} · {note.latency_seconds:.1f}s",
        "",
        NOT_ADVICE,
    ]
    return "\n".join(lines)
