"""Deterministic, non-LLM checks that every claim in a report must pass.

This module is the citation guarantee, the numeric guarantee, and the
no-advice rule, all in one place. It runs twice: inside the graph at
runtime, rejecting bad sentences before they ship, and again as the Tier 1
CI gate (Task 16), which every future change to this repository must pass.
Both callers need the same contract -- a pure function over data structures
that never raises, so a malformed claim degrades to a Violation instead of
an unhandled exception that a caller could accidentally swallow or crash on.
"""

import re
from dataclasses import dataclass

from era.edgar.xbrl import NormalizedFacts
from era.index.store import ChunkStore
from era.report.schema import Claim, ResearchNote, Section

FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")
RECOMMENDATION_TERMS = (
    "buy",
    "sell",
    "hold",
    "overweight",
    "underweight",
    "price target",
    "outperform",
    "underperform",
    "strong buy",
    "we recommend",
)
# A fact is accepted if it lands within this fraction of the filed XBRL
# value. Not zero, because a claim may round a figure ("about $391 billion")
# while the fact's `value` field carries the unrounded number -- the
# citation is still honest at that point, only its prose rounds.
RELATIVE_TOLERANCE = 0.005
YEAR_RANGE = range(1900, 2100)
# Below this, a number in prose is a year, an ordinal, a share count or a
# section reference -- not a financial magnitude anyone could meaningfully
# hallucinate. Checking those produces constant false violations, which
# would drive the retry loop on every section and burn the run's budget
# rediscovering nothing. This threshold is a judgement call, not a measured
# one: nothing in this repository yet exercises the checker against real
# generated text (that lands in Task 13), so it may need retuning once it
# does.
MATERIAL_FIGURE_MIN = 1000


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str


def _normalise(figure: str) -> str:
    """Compare figures by value, not spelling: 391,035.0 and 391035 are one number."""
    plain = figure.replace(",", "")
    if plain.endswith(".0"):
        plain = plain[:-2]
    return plain.rstrip(".")


def _figures(text: str) -> set[str]:
    return {_normalise(match.group()) for match in FIGURE.finditer(text)}


def _is_material(figure: str) -> bool:
    try:
        value = float(figure)
    except ValueError:
        return False
    if value.is_integer() and int(value) in YEAR_RANGE:
        return False
    return abs(value) >= MATERIAL_FIGURE_MIN


def _fact_spellings(value: float) -> set[str]:
    """Filings state 391,035 (in millions); XBRL states 391035000000.

    Both are the same figure, so a claim quoting the filing's presented
    scale must not be flagged as unsupported. Accept the value at every
    scale a filing conventionally uses -- absolute dollars, thousands,
    millions, and billions.
    """
    spellings: set[str] = set()
    for scale in (1, 1_000, 1_000_000, 1_000_000_000):
        scaled = value / scale
        if scaled >= 1 and scaled.is_integer():
            spellings.add(str(int(scaled)))
    return spellings


def no_recommendation_language(text: str) -> list[Violation]:
    """The no-advice rule: this product reports facts, it does not tell anyone what to do."""
    lowered = text.lower()
    return [
        Violation(kind="recommendation_language", detail=term)
        for term in RECOMMENDATION_TERMS
        if re.search(rf"\b{re.escape(term)}\b", lowered)
    ]


def _verify_claim(claim: Claim, store: ChunkStore, facts: NormalizedFacts) -> list[Violation]:
    violations = list(no_recommendation_language(claim.text))

    supporting_text: list[str] = []
    for ref in claim.chunks:
        stored = store.get(ref.accession, ref.chunk_id)
        if stored is None:
            violations.append(
                Violation(
                    kind="unresolvable_citation",
                    detail=f"{ref.accession}#{ref.chunk_id}",
                )
            )
            continue
        supporting_text.append(stored.text)

    supported_figures: set[str] = set()
    for fact in claim.facts:
        # Matched on tag AND fiscal_period, not tag alone: FactRef.fiscal_period
        # is the claim's assertion of *which* period this figure belongs to, and
        # facts.metrics holds only the most-current period per metric. A tag-only
        # match would let a claim mislabel this year's number as last year's and
        # still pass, as long as the two years' figures happened to be close.
        exact = next(
            (
                m
                for m in facts.metrics.values()
                if m.tag == fact.tag and m.fiscal_period == fact.fiscal_period
            ),
            None,
        )
        if exact is None:
            tag_known = any(m.tag == fact.tag for m in facts.metrics.values())
            if tag_known:
                violations.append(
                    Violation(
                        kind="fact_period_mismatch", detail=f"{fact.tag} {fact.fiscal_period}"
                    )
                )
            else:
                violations.append(Violation(kind="unknown_fact_tag", detail=fact.tag))
        elif abs(exact.value - fact.value) > abs(exact.value) * RELATIVE_TOLERANCE:
            violations.append(
                Violation(
                    kind="figure_disagrees_with_xbrl",
                    detail=f"{fact.tag}: claimed {fact.value}, XBRL {exact.value}",
                )
            )
        else:
            supported_figures |= _fact_spellings(exact.value)

    # Only material figures are checked. A claim saying "in fiscal 2024" must
    # not be rejected because "2024" happens not to appear in the cited chunk.
    claimed = {figure for figure in _figures(claim.text) if _is_material(figure)}
    if claimed:
        supported = set(supported_figures)
        for text in supporting_text:
            supported |= _figures(text)
        for figure in sorted(claimed - supported):
            violations.append(Violation(kind="unsupported_figure", detail=figure))

    return violations


def verify_section(section: Section, store: ChunkStore, facts: NormalizedFacts) -> list[Violation]:
    """Check every claim in a section against its cited chunks and facts.

    Never raises: a claim that breaks a check in an unanticipated way (a
    store that errors, malformed input smuggled past the schema) becomes a
    Violation for that claim rather than an exception that aborts checking
    every other claim in the section -- and, for the CI-gate caller, could
    be worked around by catching and ignoring it.
    """
    if not section.available:
        return []

    violations: list[Violation] = []
    for claim in section.claims:
        try:
            violations.extend(_verify_claim(claim, store, facts))
        except Exception as exc:  # noqa: BLE001 -- see docstring: this must not raise
            violations.append(
                Violation(kind="verification_error", detail=f"{type(exc).__name__}: {exc}")
            )
    return violations


def verify_note(note: ResearchNote, store: ChunkStore, facts: NormalizedFacts) -> list[Violation]:
    violations: list[Violation] = []
    for section in note.sections:
        violations.extend(verify_section(section, store, facts))
    return violations
