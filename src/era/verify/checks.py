"""Deterministic, non-LLM checks that every claim in a report must pass.

This module is the citation guarantee, the numeric guarantee, and the
no-advice rule, all in one place. It runs twice: inside the graph at
runtime, rejecting bad sentences before they ship, and again as the Tier 1
CI gate (Task 16), which every future change to this repository must pass.
Both callers need the same contract -- a function over data structures
that never raises, so a malformed claim degrades to a Violation instead of
an unhandled exception that a caller could accidentally swallow or crash on.

What "citation guarantee" does and does not mean: a resolvable citation
proves the cited passage exists and that the words this module checks
appear in it. It does not prove those words are *evidence* for the claim --
a chunk mentioning "4,200 pending lawsuits" satisfies a claim that "the
company closed 4,200 stores" as far as this module can tell, because both
share the token 4200. Judging relevance is not something a deterministic,
model-free layer can do; that gap is inherent, not a bug to fix here.
"""

import math
import re
from dataclasses import dataclass, replace
from typing import Literal

from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.index.store import ChunkStore
from era.report.schema import Claim, ResearchNote, Section, SectionName

# A fact or figure is accepted if it lands within this fraction of the
# reference value. Not zero, because a claim may round a figure ("about
# $391 billion") while the underlying value is unrounded -- the citation is
# still honest at that point, only its prose rounds.
RELATIVE_TOLERANCE = 0.005
YEAR_RANGE = range(1900, 2100)
# Below this, a bare number in prose is a year, an ordinal, a share count or
# a section reference -- not a financial magnitude anyone could meaningfully
# hallucinate. Checking those produces constant false violations, which
# would drive the retry loop on every section and burn the run's budget
# rediscovering nothing. This threshold (and RELATIVE_TOLERANCE above) are
# judgement calls, not measured ones: nothing in this repository yet
# exercises the checker against real generated text (that lands in
# Task 13), so both may need retuning once it does.
MATERIAL_FIGURE_MIN = 1000.0

MAGNITUDE_WORDS = {
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
}
# A number, optionally followed by one of the words above. Financial prose
# routinely writes "$391 billion" rather than spelling out the digits, and
# the *word* carries the magnitude -- a materiality check keyed only on the
# literal digits ("391") would exempt nearly every figure in a financial
# health section from being checked at all.
_SCALE_WORD = "|".join(MAGNITUDE_WORDS)
NUMBER_WITH_SCALE = re.compile(
    # \b after the word group matters: without it, "million/billion/..."
    # match as a bare prefix of any longer word that starts the same way --
    # "2 billionth customer" or "named 10 billionaires" would otherwise
    # parse as a $2 billion / $10 billion claim.
    rf"(?P<literal>\d[\d,]*(?:\.\d+)?)(?:\s+(?P<word>{_SCALE_WORD})\b)?",
    re.IGNORECASE,
)

# Bare verbs ("buy", "sell", "hold") are common in ordinary filing prose --
# "will hold its annual meeting", "customers buy directly", "will sell the
# division" -- and flagging them by themselves makes this check fail on
# sentences that are not advice at all, including the project's own
# not-investment-advice footer (see tests: the standard disclaimer contains
# both "buy" and "sell"). Each pattern below instead matches the *frame* an
# actual recommendation is written in: a verb attached to a rating word
# ("rated a Buy"), an imperative aimed at the reader ("we recommend",
# "should buy"), or a term that has no ordinary-prose meaning at all
# ("price target").
#
# outperform/underperform get the same frame-only treatment as buy/sell/hold
# -- "we expect the sector to underperform the broader market" and "margins
# underperform versus peers" are ordinary comparative commentary, not a
# rating action, so they are only caught via a frame below (rated/should/
# rating), not as bare words. overweight/underweight are specialised enough
# finance jargon (a portfolio-weighting call, not a plain-English verb) that
# they stay bare-matched.
_RATING_VERBS = r"(?:buy|sell|hold|outperform|underperform|overweight|underweight)"
RECOMMENDATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # `rated`, never `rate`. An earlier version used `rated?`, which matched
    # the bare noun and made this check fire on the core vocabulary of the
    # financial-health section -- "our effective tax rate benefited from cash
    # we hold overseas" and "the interest rate on the notes we hold" both
    # tripped it. Those sentences are generated every run, so the false
    # positives would have burned the retry budget continuously and blocked
    # the Tier 1 merge gate for no reason.
    ("rated", re.compile(rf"\brated\b[^.\n]{{0,40}}\b{_RATING_VERBS}\b", re.IGNORECASE)),
    # The present-tense form needs an explicit analyst subject to distinguish
    # "we rate the shares a Buy" from any sentence containing the noun.
    (
        "we rate",
        re.compile(rf"\b(?:we|analysts?)\s+rate\b[^.\n]{{0,40}}\b{_RATING_VERBS}\b", re.IGNORECASE),
    ),
    ("should buy/sell/hold", re.compile(rf"\bshould\s+{_RATING_VERBS}\b", re.IGNORECASE)),
    ("we recommend", re.compile(r"\bwe\s+recommend\b", re.IGNORECASE)),
    ("price target", re.compile(r"\bprice\s+target\b", re.IGNORECASE)),
    ("strong buy", re.compile(r"\bstrong\s+buy\b", re.IGNORECASE)),
    ("overweight", re.compile(r"\boverweight\b", re.IGNORECASE)),
    ("underweight", re.compile(r"\bunderweight\b", re.IGNORECASE)),
    ("rating", re.compile(rf"\b{_RATING_VERBS}\s+rating\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str
    # "content": the claim itself is wrong (bad citation, unsupported
    # figure, advisory language). "infrastructure": the checker couldn't
    # finish (a store that errored, malformed input) -- Task 13's retry
    # loop should not spend its budget regenerating a section over a
    # database outage, so callers need to be able to tell the two apart.
    category: Literal["content", "infrastructure"] = "content"
    # Which section and which claim within it this violation came from.
    # verify_note flattens every section's violations into one list; without
    # a locator, a risk-factors problem is indistinguishable from a
    # financial-health one, and Task 13 has no way to route a complaint
    # back to the subgraph that should act on it.
    section: SectionName | None = None
    claim_index: int | None = None


@dataclass(frozen=True)
class _Figure:
    detail: str
    candidates: frozenset[float]
    material: bool
    # The single value this figure asserts at face value: "391 billion" means
    # 3.91e11 and nothing else, and a bare "4,200" asserts 4200 even though a
    # millions-denominated table might rescale it.
    exact: float
    # True when the writer supplied the scale word. A scale-explicit figure is
    # unambiguous, so it must be matched only against what a source actually
    # asserts -- never against a source's rescaled interpretations. Without
    # this distinction, "Revenue was 391 million USD" passed against a
    # $391bn fact, because the reference had been expanded down to 391,035,000
    # and the mantissa matched. A 1000x error through the numeric guarantee.
    scale_explicit: bool


def _display(literal: str, word: str | None) -> str:
    """Human-readable form of a matched figure, for Violation.detail.

    Trims *all* trailing fractional zeros ("4,200.00" -> "4200"), not just a
    single ".0" -- otherwise "4200" and "4200.00" would report as different
    detail strings and the seen-figures dedup in _verify_claim (keyed on
    this string) would treat one claim restating the same number twice as
    two figures to check instead of one.
    """
    normalised = literal.replace(",", "")
    if "." in normalised:
        normalised = normalised.rstrip("0").rstrip(".")
    return f"{normalised} {word}" if word else normalised


def _parse_figures(text: str) -> list[_Figure]:
    """Find every number in text and the dollar-value(s) it could mean.

    A number with an explicit scale word ("391 billion") has one
    unambiguous value. A bare number ("391,035") does not: filings
    routinely present a figure in thousands, millions, or billions without
    saying so in the sentence, so it is compared against the reference
    value at every one of those scales -- the same ambiguity XBRL's exact
    dollar figures create for prose, just walked in the other direction.
    """
    figures: list[_Figure] = []
    for match in NUMBER_WITH_SCALE.finditer(text):
        literal_str = match.group("literal")
        word = match.group("word")
        try:
            literal = float(literal_str.replace(",", ""))
        except ValueError:
            continue

        if word is not None:
            value = literal * MAGNITUDE_WORDS[word.lower()]
            candidates = frozenset({value})
            material = True
            exact = value
            scale_explicit = True
        else:
            exact = literal
            scale_explicit = False
            candidates = frozenset(
                {literal, literal * 1_000, literal * 1_000_000, literal * 1_000_000_000}
            )
            # A comma or decimal point means the writer punctuated the
            # number deliberately -- "2,024" or "2024.0" is a magnitude, not
            # a year, even though the bare digits fall in YEAR_RANGE.
            has_punctuation = "," in literal_str or "." in literal_str
            is_year = not has_punctuation and literal.is_integer() and int(literal) in YEAR_RANGE
            material = not is_year and abs(literal) >= MATERIAL_FIGURE_MIN

        figures.append(
            _Figure(
                detail=_display(literal_str, word),
                candidates=candidates,
                material=material,
                exact=exact,
                scale_explicit=scale_explicit,
            )
        )
    return figures


def _isclose(a: float, b: float) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return math.isclose(a, b, rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-6)


def _fact_value(value: float) -> frozenset[float]:
    """What this fact asserts, as a set so callers can union it into a pool.

    Exactly one value, never a rescaled family. XBRL reports an exact dollar
    amount -- there is no ambiguity in it to model. Expanding a fact downward
    is also what broke the numeric guarantee: against a true $391,035,000,000,
    the expansion produced 391,035,000, which "Revenue was 391 million USD"
    then matched inside tolerance. The prose side already walks the ambiguity
    in the direction it actually exists (a bare "391,035" in a millions table),
    so expanding here was both wrong and redundant.

    abs(value), not value: the digit regex never captures a leading minus sign
    (prose writes "a net loss of $5 billion", not "-$5 billion"), so a negative
    XBRL fact -- routine for net_income, operating_income and operating_cash_flow
    on a loss-making filer -- must still match a correct citation.
    """
    if not math.isfinite(value):
        return frozenset()
    return frozenset({abs(value)})


def no_recommendation_language(text: str) -> list[Violation]:
    """The no-advice rule: this product reports facts, it does not tell anyone what to do."""
    return [
        Violation(kind="recommendation_language", detail=label)
        for label, pattern in RECOMMENDATION_PATTERNS
        if pattern.search(text)
    ]


def _matching_metric(facts: NormalizedFacts, tag: str, period: str) -> MetricValue | None:
    return next(
        (m for m in facts.metrics.values() if m.tag == tag and m.fiscal_period == period),
        None,
    )


def _verify_chunks(
    claim: Claim, store: ChunkStore, accessions: frozenset[str] | None
) -> tuple[list[Violation], list[str]]:
    violations: list[Violation] = []
    supporting_text: list[str] = []
    for ref in claim.chunks:
        try:
            if accessions is not None and ref.accession not in accessions:
                violations.append(
                    Violation(
                        kind="foreign_accession",
                        detail=f"chunk {ref.accession}#{ref.chunk_id} "
                        "is not among this note's filings",
                    )
                )
                continue
            stored = store.get(ref.accession, ref.chunk_id)
        except Exception as exc:  # noqa: BLE001 -- a store failure must not abort the claim
            violations.append(
                Violation(
                    kind="verification_error",
                    detail=f"chunk {ref.accession}#{ref.chunk_id}: {type(exc).__name__}: {exc}",
                    category="infrastructure",
                )
            )
            continue
        if stored is None:
            violations.append(
                Violation(kind="unresolvable_citation", detail=f"{ref.accession}#{ref.chunk_id}")
            )
            continue
        supporting_text.append(stored.text)
    return violations, supporting_text


def _verify_facts(
    claim: Claim, facts: NormalizedFacts, accessions: frozenset[str] | None
) -> tuple[list[Violation], set[float]]:
    violations: list[Violation] = []
    supported_pool: set[float] = set()
    for fact in claim.facts:
        try:
            if accessions is not None and fact.accession not in accessions:
                violations.append(
                    Violation(
                        kind="foreign_accession",
                        detail=f"fact {fact.tag} claims accession {fact.accession}, "
                        "not among this note's filings",
                    )
                )
                continue

            exact = _matching_metric(facts, fact.tag, fact.fiscal_period)
            if exact is None:
                tag_known = any(m.tag == fact.tag for m in facts.metrics.values())
                kind = "fact_period_mismatch" if tag_known else "unknown_fact_tag"
                violations.append(Violation(kind=kind, detail=f"{fact.tag} {fact.fiscal_period}"))
                continue

            # The tag and period resolved, but to a fact filed under a
            # different accession than the claim states -- e.g. a note
            # spanning two filings that attributes this year's number to
            # last year's accession. facts.metrics itself carries only the
            # most-current period per metric, so this is the only place
            # that can catch a mismatch here.
            if exact.accession != fact.accession:
                violations.append(
                    Violation(
                        kind="foreign_accession",
                        detail=f"{fact.tag}: claim cites {fact.accession}, "
                        f"resolved fact is from {exact.accession}",
                    )
                )
                continue

            if not math.isfinite(exact.value) or not math.isfinite(fact.value):
                violations.append(
                    Violation(
                        kind="non_finite_value",
                        detail=f"{fact.tag}: claimed {fact.value}, XBRL {exact.value}",
                    )
                )
                continue

            if abs(exact.value - fact.value) > abs(exact.value) * RELATIVE_TOLERANCE:
                violations.append(
                    Violation(
                        kind="figure_disagrees_with_xbrl",
                        detail=f"{fact.tag}: claimed {fact.value}, XBRL {exact.value}",
                    )
                )
            else:
                supported_pool |= _fact_value(exact.value)
        except Exception as exc:  # noqa: BLE001 -- malformed fact data must not abort the claim
            violations.append(
                Violation(
                    kind="verification_error",
                    detail=f"fact {fact.tag}: {type(exc).__name__}: {exc}",
                    category="infrastructure",
                )
            )
    return violations, supported_pool


def _verify_claim(
    claim: Claim,
    store: ChunkStore,
    facts: NormalizedFacts,
    accessions: frozenset[str] | None,
) -> list[Violation]:
    # Computed first and never discarded: a citation lookup failing later in
    # this claim must not erase a recommendation-language finding already
    # made on its text.
    violations = list(no_recommendation_language(claim.text))

    chunk_violations, supporting_text = _verify_chunks(claim, store, accessions)
    violations.extend(chunk_violations)

    fact_violations, supported_pool = _verify_facts(claim, facts, accessions)
    violations.extend(fact_violations)

    # Only material figures are checked. A claim saying "in fiscal 2024"
    # must not be rejected because "2024" happens not to appear in the
    # cited chunk.
    claimed_figures = [figure for figure in _parse_figures(claim.text) if figure.material]
    if claimed_figures:
        # Two pools, because scale ambiguity is not symmetric.
        #
        # `asserted` is what the sources literally say: an XBRL fact's exact
        # dollar amount, and each prose figure's face value.
        # `rescaled` additionally holds the readings a *bare* source number
        # could carry -- "391,035" inside a millions-denominated table really
        # does mean $391bn, and that ambiguity is real.
        #
        # A claim that names its own scale ("391 million") is unambiguous, so
        # it is checked against `asserted` only. Letting it reach `rescaled`
        # is what allowed a 1000x error to pass: every mantissa-correct claim
        # matched some rescaling of the source, whatever magnitude it named.
        asserted = set(supported_pool)
        rescaled = set(supported_pool)
        for text in supporting_text:
            for figure in _parse_figures(text):
                asserted.add(figure.exact)
                rescaled |= figure.candidates

        seen: set[str] = set()
        for figure in claimed_figures:
            if figure.detail in seen:
                continue
            seen.add(figure.detail)
            pool = asserted if figure.scale_explicit else rescaled
            if not any(
                _isclose(claimed, supported) for claimed in figure.candidates for supported in pool
            ):
                violations.append(Violation(kind="unsupported_figure", detail=figure.detail))

    return violations


def verify_section(
    section: Section,
    store: ChunkStore,
    facts: NormalizedFacts,
    *,
    accessions: frozenset[str] | None,
) -> list[Violation]:
    """Check every claim in a section against its cited chunks and facts.

    accessions is the set of filing accessions this note is actually about
    (ResearchNote.accessions). A ChunkRef or FactRef whose own accession
    falls outside that set is flagged foreign_accession: the chunk store
    and the facts payload can each hold data for many companies at once
    (see ChunkStore's own docstring), so nothing else stops a claim about
    one filer from citing another's filing, and a chunk_id or a (tag,
    period) pair alone cannot tell the difference. Pass None only when the
    caller has deliberately decided not to enforce this; verify_note, the
    primary caller, always supplies a real set.

    Never raises, at both the claim level and the section level: a broken
    store, a malformed fact, or a section that isn't the well-formed object
    its type promises all degrade to a Violation (category="infrastructure"
    for the latter) rather than an exception that could abort every other
    claim in the section or be silently caught and ignored by a caller.
    """
    try:
        if not section.available:
            return []

        violations: list[Violation] = []
        for index, claim in enumerate(section.claims):
            try:
                claim_violations = _verify_claim(claim, store, facts, accessions)
            except Exception as exc:  # noqa: BLE001 -- see docstring
                claim_violations = [
                    Violation(
                        kind="verification_error",
                        detail=f"{type(exc).__name__}: {exc}",
                        category="infrastructure",
                    )
                ]
            violations.extend(
                replace(violation, section=section.name, claim_index=index)
                for violation in claim_violations
            )
        return violations
    except Exception as exc:  # noqa: BLE001 -- see docstring: structural failures too
        return [
            Violation(
                kind="verification_error",
                detail=f"section-level failure: {type(exc).__name__}: {exc}",
                category="infrastructure",
            )
        ]


def verify_note(note: ResearchNote, store: ChunkStore, facts: NormalizedFacts) -> list[Violation]:
    try:
        accessions = frozenset(note.accessions)
        violations: list[Violation] = []
        for section in note.sections:
            violations.extend(verify_section(section, store, facts, accessions=accessions))
        return violations
    except Exception as exc:  # noqa: BLE001 -- see verify_section's docstring
        return [
            Violation(
                kind="verification_error",
                detail=f"note-level failure: {type(exc).__name__}: {exc}",
                category="infrastructure",
            )
        ]
