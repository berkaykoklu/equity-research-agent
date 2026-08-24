import math

from tests.fakes import FakeEmbedder, InMemoryChunkStore

from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.index.chunking import Chunk
from era.index.store import StoredChunk
from era.report.schema import (
    ChunkRef,
    Claim,
    Coverage,
    FactRef,
    ResearchNote,
    Section,
    SectionName,
)
from era.verify.checks import Violation, no_recommendation_language, verify_note, verify_section

ACC = frozenset({"acc"})

FACTS = NormalizedFacts(
    metrics={
        "revenue": MetricValue(
            tag="Revenues",
            fiscal_period="FY2024",
            value=391_035_000_000.0,
            accession="acc",
        )
    },
    missing=(),
)


def _store_with(text: str, *, accession: str = "acc", chunk_id: int = 0) -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=chunk_id, accession=accession, item="1A", text=text, start=0)
    store.upsert([chunk], FakeEmbedder().embed([text]))
    return store


# --- brief's baseline tests (adapted to the accessions parameter) ----------


def test_accepts_a_claim_whose_chunk_supports_it() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="Supply chain concentration is a material risk.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    assert verify_section(section, store, FACTS, accessions=ACC) == []


def test_rejects_a_citation_that_does_not_resolve() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="Supply chain concentration is a material risk.",
                chunks=(ChunkRef(accession="acc", chunk_id=999),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["unresolvable_citation"]


def test_rejects_a_number_absent_from_the_cited_chunk() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="The company closed 4,200 retail stores.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["unsupported_figure"]


def test_ignores_years_and_small_counts_in_claim_text() -> None:
    # "2024" and "3" are a year and an ordinal, not magnitudes anyone could
    # hallucinate. Flagging them would fire the retry loop on every section.
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="In fiscal 2024 the company identified 3 principal risks.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    assert verify_section(section, store, FACTS, accessions=ACC) == []


def test_accepts_a_figure_quoted_at_the_filing_s_presented_scale() -> None:
    # Filings present revenue in millions; XBRL reports it in dollars. A claim
    # quoting 391035 must not be rejected against an XBRL value of 391035000000.
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was 391,035 million USD.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=391_035_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    assert verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC) == []


def test_rejects_a_fact_value_that_disagrees_with_xbrl() -> None:
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was 400 billion USD.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=400_000_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC)

    # "400 billion" carries an explicit magnitude word, so it is material
    # regardless of its bare digits -- and since the fact itself failed to
    # validate, nothing populates the supported pool, so the prose figure is
    # separately unsupported too. Both are true, independent statements
    # about this claim.
    assert {v.kind for v in violations} == {"figure_disagrees_with_xbrl", "unsupported_figure"}


def test_flags_recommendation_language() -> None:
    assert no_recommendation_language("We rate the shares a Buy.")
    assert no_recommendation_language("Our price target is 250 USD.")
    assert no_recommendation_language("Revenue grew in fiscal 2024.") == []


# --- fiscal-period matching (tag alone is not enough) -----------------------


def test_rejects_a_fact_whose_period_disagrees_with_the_claim() -> None:
    # The tag is right and the value happens to equal the FY2024 figure, but
    # the claim asserts it belongs to FY2023 -- a genuine mislabeling that a
    # tag-only match would silently accept. Because the fact fails to
    # resolve, it also never contributes to the supported pool, so the
    # figure quoted in prose is separately unsupported too -- both are real
    # problems with this claim and both should surface.
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="FY2023 revenue was 391,035 million USD.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2023",
                        value=391_035_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC)

    assert {v.kind for v in violations} == {"fact_period_mismatch", "unsupported_figure"}


def test_rejects_an_unknown_fact_tag() -> None:
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="EBITDA was 50 billion USD.",
                facts=(
                    FactRef(
                        tag="SomeTagNotInFacts",
                        fiscal_period="FY2024",
                        value=50_000_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC)

    # "50 billion" is material via its magnitude word, and with the fact
    # itself unresolved nothing supports it either.
    assert {v.kind for v in violations} == {"unknown_fact_tag", "unsupported_figure"}


# --- materiality and tolerance boundaries (exact, not > 0) ------------------


def test_figure_exactly_at_the_materiality_floor_is_checked() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="The company closed 1000 retail stores.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["unsupported_figure"]


def test_figure_just_below_the_materiality_floor_is_ignored() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="The company closed 999 retail stores.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    assert verify_section(section, store, FACTS, accessions=ACC) == []


def test_fact_within_tolerance_is_accepted() -> None:
    # 0.4% off -- inside the 0.5% relative tolerance.
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was about 392 billion USD.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=392_599_140_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    assert verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC) == []


def test_fact_just_outside_tolerance_is_rejected() -> None:
    # 391_035_000_000 * 1.006 -- just over the 0.5% relative tolerance. No
    # figure appears in the prose itself, so this isolates the fact/XBRL
    # tolerance check alone from the separate prose-figure check.
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue for the period diverged from the amount on file.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=393_381_210_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["figure_disagrees_with_xbrl"]


def test_fact_accepted_at_thousands_and_billions_scale_too() -> None:
    # Every scale a filing conventionally uses, not just millions -- a
    # fixture with only one scale exercised would pass even if the other
    # branches were deleted. Each case uses its own NormalizedFacts so the
    # claimed figure, the XBRL value, and the scale under test line up.
    thousands_facts = NormalizedFacts(
        metrics={
            "revenue": MetricValue(
                tag="Revenues",
                fiscal_period="FY2024",
                value=391_035_000_000.0,
                accession="acc",
            )
        },
        missing=(),
    )
    thousands = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was 391,035,000 thousand USD.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=391_035_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    billions_facts = NormalizedFacts(
        metrics={
            "revenue": MetricValue(
                tag="Revenues",
                fiscal_period="FY2024",
                value=1_234_000_000_000.0,
                accession="acc",
            )
        },
        missing=(),
    )
    billions = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was 1,234 billion USD.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=1_234_000_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    assert verify_section(thousands, InMemoryChunkStore(), thousands_facts, accessions=ACC) == []
    assert verify_section(billions, InMemoryChunkStore(), billions_facts, accessions=ACC) == []


# --- realistic, mixed claim text --------------------------------------------


def test_realistic_paragraph_flags_only_the_unsupported_figure() -> None:
    store = _store_with(
        "In fiscal 2024, the company closed 4,200 underperforming retail "
        "locations as part of a restructuring announced in the third quarter."
    )
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text=(
                    "In fiscal 2024 the company closed 4,200 retail locations "
                    "and also shut down 7,500 warehouses, its 2nd such action."
                ),
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["unsupported_figure"]
    assert violations[0].detail == "7500"


def test_multiple_violation_kinds_on_one_claim_are_all_reported() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="We rate the shares a Buy given the 4,200 store closures.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS, accessions=ACC)

    assert {v.kind for v in violations} == {"recommendation_language", "unsupported_figure"}
    assert len(violations) == 2


def test_figure_supported_by_chunk_text_not_by_a_fact() -> None:
    store = _store_with("The restructuring plan affected 4,200 locations nationwide.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="The restructuring affected 4,200 locations.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    assert verify_section(section, store, FACTS, accessions=ACC) == []


# --- structural behaviour ----------------------------------------------------


def test_unavailable_section_produces_no_violations() -> None:
    section = Section(
        name=SectionName.RISK_FACTORS,
        available=False,
        unavailable_reason="Item 1A boundary not found in filing",
    )

    assert verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC) == []


def test_verify_note_aggregates_violations_across_every_section() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    good_section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="Supply chain concentration is a material risk.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )
    bad_section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="We rate the shares a Buy.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )
    note = ResearchNote(
        ticker="AAPL",
        cik="0000320193",
        sections=(good_section, bad_section),
        coverage=Coverage(sections_available=2, sections_total=2),
        accessions=("acc",),
    )

    violations = verify_note(note, store, FACTS)

    assert [v.kind for v in violations] == ["recommendation_language"]
    assert violations[0].section == SectionName.FINANCIAL_HEALTH
    assert violations[0].claim_index == 0


def test_violation_is_a_frozen_comparable_value() -> None:
    assert Violation(kind="x", detail="y") == Violation(kind="x", detail="y")


# =============================================================================
# Findings from the second review pass
# =============================================================================


# --- Critical 1: accession scoping ------------------------------------------


def test_rejects_a_chunk_citation_outside_the_notes_own_filings() -> None:
    # The store holds chunks from many companies at once (its own contract
    # says so). Nothing about a chunk_id resolving successfully proves the
    # accession it resolved under belongs to *this* note -- a claim about
    # Apple could otherwise cite Microsoft's filing and pass. The two chunks
    # below deliberately carry identical text so a check that only asked
    # "did this resolve" -- rather than "is this accession one of ours" --
    # would still pass.
    store = InMemoryChunkStore()
    same_text = "Supply chain concentration is a material risk."
    aapl_chunk = Chunk(chunk_id=0, accession="aapl-acc", item="1A", text=same_text, start=0)
    msft_chunk = Chunk(chunk_id=0, accession="msft-acc", item="1A", text=same_text, start=0)
    store.upsert([aapl_chunk], FakeEmbedder().embed([same_text]))
    store.upsert([msft_chunk], FakeEmbedder().embed([same_text]))

    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(Claim(text=same_text, chunks=(ChunkRef(accession="msft-acc", chunk_id=0),)),),
    )

    violations = verify_section(section, store, FACTS, accessions=frozenset({"aapl-acc"}))

    assert [v.kind for v in violations] == ["foreign_accession"]


def test_citation_lookup_uses_the_refs_own_accession_not_a_hardcoded_one() -> None:
    # Guards the same blind spot from the other direction: two accessions
    # that are BOTH in the note's allowed set, with different content under
    # the same chunk_id. If the lookup ever hardcoded one accession instead
    # of reading ref.accession, this claim would resolve against the wrong
    # filing's text and get a spurious unsupported_figure.
    store = InMemoryChunkStore()
    chunk_a = Chunk(
        chunk_id=0, accession="acc-a", item="1A", text="Discusses 9,999 widgets.", start=0
    )
    chunk_b = Chunk(chunk_id=0, accession="acc-b", item="1A", text="Closed 4,200 stores.", start=0)
    store.upsert([chunk_a], FakeEmbedder().embed([chunk_a.text]))
    store.upsert([chunk_b], FakeEmbedder().embed([chunk_b.text]))

    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="The company closed 4,200 stores.",
                chunks=(ChunkRef(accession="acc-b", chunk_id=0),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS, accessions=frozenset({"acc-a", "acc-b"}))

    assert violations == []


def test_rejects_a_fact_citation_outside_the_notes_own_filings() -> None:
    # Isolated from the accession-vs-resolved-metric check below: the fact's
    # accession here matches the metric's own accession exactly (both
    # "rogue-acc"), so that check cannot be what flags this -- the only
    # thing that can is the note-level accession-scope check, since
    # "rogue-acc" is not in the note's own accessions set. No figure
    # appears in the prose either, isolating this from the separate
    # prose-figure check.
    facts = NormalizedFacts(
        metrics={
            "revenue": MetricValue(
                tag="Revenues",
                fiscal_period="FY2024",
                value=391_035_000_000.0,
                accession="rogue-acc",
            )
        },
        missing=(),
    )
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue for the period matched the amount on file.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=391_035_000_000.0,
                        accession="rogue-acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), facts, accessions=ACC)

    assert [v.kind for v in violations] == ["foreign_accession"]


def test_rejects_a_fact_whose_accession_disagrees_with_the_resolved_metric() -> None:
    # Both accessions belong to the note (it can span more than one
    # filing), but the fact attributes this figure to a different accession
    # than the one the metric actually resolved from -- MetricValue.accession
    # is right there and must be compared, not just the note-level set.
    facts = NormalizedFacts(
        metrics={
            "revenue": MetricValue(
                tag="Revenues",
                fiscal_period="FY2024",
                value=391_035_000_000.0,
                accession="fy24-acc",
            )
        },
        missing=(),
    )
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue for the period matched the amount on file.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=391_035_000_000.0,
                        accession="fy23-acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(
        section, InMemoryChunkStore(), facts, accessions=frozenset({"fy24-acc", "fy23-acc"})
    )

    assert [v.kind for v in violations] == ["foreign_accession"]


def test_verify_note_scopes_citations_to_its_own_accessions_automatically() -> None:
    # The end-to-end path: verify_note must derive the allowed set from
    # note.accessions itself, not require the caller to compute it.
    store = InMemoryChunkStore()
    other_chunk = Chunk(
        chunk_id=0, accession="other-acc", item="1A", text="Unrelated filing.", start=0
    )
    store.upsert([other_chunk], FakeEmbedder().embed([other_chunk.text]))

    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(text="Unrelated filing.", chunks=(ChunkRef(accession="other-acc", chunk_id=0),)),
        ),
    )
    note = ResearchNote(
        ticker="AAPL",
        cik="0000320193",
        sections=(section,),
        coverage=Coverage(sections_available=1, sections_total=1),
        accessions=("acc",),  # deliberately does not include "other-acc"
    )

    violations = verify_note(note, store, FACTS)

    assert [v.kind for v in violations] == ["foreign_accession"]


# --- Critical 2: magnitude words must count toward materiality -------------


def test_rejects_a_wildly_wrong_figure_written_with_a_magnitude_word() -> None:
    # "500 billion" -- the literal digits ("500") are below the materiality
    # floor, but the word carries the real magnitude (5e11), and it is 28%
    # off the true value. A check keyed on bare digits alone would miss this
    # entirely, exempting almost every figure a financial-health section
    # would ever state.
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was 500 billion USD in fiscal 2024.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=391_035_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["unsupported_figure"]
    assert violations[0].detail == "500 billion"


def test_accepts_a_correct_figure_written_with_a_magnitude_word() -> None:
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was 391 billion USD in fiscal 2024.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        # Within 0.5% of $391 billion.
                        value=391_035_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    assert verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC) == []


def test_a_word_that_merely_starts_with_a_magnitude_word_is_not_mistaken_for_one() -> None:
    # Without a \b after the magnitude-word group, "billion" matches as a
    # bare prefix of "billionth"/"billionaires"/etc, so "2 billionth
    # customer" would parse as a $2 billion claim -- material, and unsupported
    # by anything, since nothing corroborates a dollar figure that was never
    # actually stated.
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text=(
                    "The company welcomed its 2 billionth customer and named "
                    "10 billionaires among its largest shareholders."
                ),
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    assert verify_section(section, store, FACTS, accessions=ACC) == []


# --- Critical 3: negative XBRL values (losses) must still be supportable ---


def test_accepts_a_correctly_quoted_loss_figure() -> None:
    # net_income, operating_income, and operating_cash_flow are routinely
    # negative for a loss-making filer. A candidate-generation step that
    # requires the scaled value to be >= 1 silently produces zero candidates
    # for any negative fact, so a claim correctly quoting a loss would always
    # come back unsupported -- burning Task 13's retry budget on every
    # loss-making company.
    facts = NormalizedFacts(
        metrics={
            "net_income": MetricValue(
                tag="NetIncomeLoss",
                fiscal_period="FY2024",
                value=-5_000_000_000.0,
                accession="acc",
            )
        },
        missing=(),
    )
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="The company reported a net loss of 5,000 million USD.",
                facts=(
                    FactRef(
                        tag="NetIncomeLoss",
                        fiscal_period="FY2024",
                        value=-5_000_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    assert verify_section(section, InMemoryChunkStore(), facts, accessions=ACC) == []


def test_rejects_a_loss_figure_that_disagrees_with_xbrl_regardless_of_sign() -> None:
    # Guards against comparing tolerance as `exact.value * TOL` instead of
    # `abs(exact.value) * TOL`: with a negative exact.value and no abs(), the
    # threshold itself goes negative and the comparison either always fires
    # or never does, independent of how close the claim actually is.
    facts = NormalizedFacts(
        metrics={
            "net_income": MetricValue(
                tag="NetIncomeLoss",
                fiscal_period="FY2024",
                value=-5_000_000_000.0,
                accession="acc",
            )
        },
        missing=(),
    )
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                # No figure in the prose -- isolates the tolerance check.
                text="The reported net loss did not match the filed amount.",
                facts=(
                    FactRef(
                        tag="NetIncomeLoss",
                        fiscal_period="FY2024",
                        # 4% off -- well outside tolerance.
                        value=-5_200_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), facts, accessions=ACC)

    assert [v.kind for v in violations] == ["figure_disagrees_with_xbrl"]


# --- Warning 4: float comparison, not string comparison ---------------------


def test_accepts_a_fractional_scaled_figure() -> None:
    facts = NormalizedFacts(
        metrics={
            "revenue": MetricValue(
                tag="Revenues",
                fiscal_period="FY2024",
                value=12_345_600_000.0,
                accession="acc",
            )
        },
        missing=(),
    )
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was 12,345.6 million USD.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=12_345_600_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    assert verify_section(section, InMemoryChunkStore(), facts, accessions=ACC) == []


def test_chunk_text_and_claim_text_at_different_scales_still_agree() -> None:
    # Scale normalisation has to apply symmetrically to chunk text and claim
    # text, not just to FactRefs -- otherwise a chunk stating "391,035
    # million" and a claim stating the same figure in bare dollars disagree
    # with each other even though they say the same thing.
    store = _store_with("Full-year revenue of 391,035 million USD was reported.")
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was 391,035,000,000 USD.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    assert verify_section(section, store, FACTS, accessions=ACC) == []


def test_the_same_figure_restated_with_different_trailing_zeros_is_one_violation() -> None:
    # "4,200" and "4,200.00" are the same number. Violation.detail is used to
    # dedup repeated mentions of one figure within a claim (see the `seen`
    # set in _verify_claim); if the display form only stripped a literal
    # ".0" suffix, "4200" and "4200.00" would produce different detail
    # strings and be treated as two separate figures to check.
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="The company closed 4,200 stores, restating the 4,200.00 figure again.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["unsupported_figure"]
    assert violations[0].detail == "4200"


# --- Warning 5: match the advisory frame, not the bare verb -----------------


def test_the_planned_report_footer_passes_cleanly() -> None:
    # Verbatim from the disclaimer text in Task 14's plan
    # (docs/superpowers/plans/2026-08-10-equity-research-agent-core.md,
    # render_markdown's closing line) -- src/era/report/assemble.py does not
    # exist yet in this repo, so this is the exact text it is planned to
    # produce, pinned here so Task 14 has a regression test to run against.
    footer = (
        "This is an automated summary of public filings. It is not investment "
        "advice and contains no recommendation."
    )
    assert no_recommendation_language(footer) == []


def test_a_conventional_not_investment_advice_disclaimer_passes_cleanly() -> None:
    # A more adversarial, but entirely realistic, disclaimer phrasing --
    # it contains the bare words "buy" and "sell", which a check keyed on
    # bare verbs would flag on the project's own footer.
    disclaimer = (
        "This report is not investment advice and is not a recommendation "
        "to buy or sell any security."
    )
    assert no_recommendation_language(disclaimer) == []


def test_ordinary_prose_uses_of_recommendation_verbs_are_not_flagged() -> None:
    assert no_recommendation_language("The company will hold its annual meeting in June.") == []
    assert no_recommendation_language("The company will sell the division next quarter.") == []
    assert no_recommendation_language("Customers buy directly from our online store.") == []


def test_ordinary_comparative_use_of_outperform_underperform_is_not_flagged() -> None:
    # outperform/underperform got the same frame-only treatment as
    # buy/sell/hold: bare, they describe ordinary comparative performance
    # commentary, not a rating action.
    assert (
        no_recommendation_language(
            "We expect the sector to underperform the broader market next year."
        )
        == []
    )
    assert no_recommendation_language("Margins underperform versus peers this quarter.") == []
    assert no_recommendation_language("The division continued to outperform expectations.") == []


def test_outperform_underperform_are_still_caught_inside_a_rating_frame() -> None:
    # The frame requirement must not silently drop real rating language --
    # only the bare, unframed use should stop being flagged.
    assert no_recommendation_language("Analysts rated the stock Underperform.")
    assert no_recommendation_language("The stock carries an Outperform rating from the firm.")


def test_recommendation_word_embedded_in_a_longer_word_is_not_flagged() -> None:
    assert no_recommendation_language("The fund reported strong holdings this quarter.") == []


def test_recommendation_word_as_a_standalone_word_is_flagged() -> None:
    text = "Analysts believe investors should hold the stock through year end."
    violations = no_recommendation_language(text)
    assert [v.detail for v in violations] == ["should buy/sell/hold"]


def test_we_recommend_is_flagged_even_with_a_verb_attached() -> None:
    assert no_recommendation_language("We recommend buying additional shares.")


# --- Warning 6: non-finite values must not pass silently --------------------


def test_rejects_a_non_finite_claimed_value() -> None:
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was reported at an indeterminate figure.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=math.nan,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["non_finite_value"]


def test_rejects_a_non_finite_reference_value() -> None:
    facts = NormalizedFacts(
        metrics={
            "revenue": MetricValue(
                tag="Revenues", fiscal_period="FY2024", value=math.inf, accession="acc"
            )
        },
        missing=(),
    )
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                # No figure in the prose -- isolates the non-finite check.
                text="Revenue for the period matched the amount on file.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=391_035_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(section, InMemoryChunkStore(), facts, accessions=ACC)

    assert [v.kind for v in violations] == ["non_finite_value"]


# --- Warning 7: a crash on one ref must not discard other findings ---------


def test_a_store_error_on_one_chunk_does_not_discard_other_findings_on_the_claim() -> None:
    class PartlyExplodingStore:
        def upsert(self, chunks: object, vectors: object) -> None:  # pragma: no cover
            raise AssertionError("not used")

        def query(self, *args: object, **kwargs: object) -> list[StoredChunk]:  # pragma: no cover
            raise AssertionError("not used")

        def get(self, accession: str, chunk_id: int) -> StoredChunk | None:
            if chunk_id == 0:
                return StoredChunk(chunk_id=0, accession=accession, item="1A", start=0, text="ok")
            raise RuntimeError("connection reset")

    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="We rate the shares a Buy.",
                chunks=(
                    ChunkRef(accession="acc", chunk_id=0),
                    ChunkRef(accession="acc", chunk_id=1),
                ),
            ),
        ),
    )

    violations = verify_section(
        section,
        PartlyExplodingStore(),
        FACTS,
        accessions=ACC,  # type: ignore[arg-type]
    )

    kinds = {v.kind for v in violations}
    assert "recommendation_language" in kinds
    assert "verification_error" in kinds


# --- Warning 8: infrastructure failures are categorically distinct ---------


def test_content_violations_and_infrastructure_failures_have_different_categories() -> None:
    class ExplodingStore:
        def upsert(self, chunks: object, vectors: object) -> None:  # pragma: no cover
            raise AssertionError("not used")

        def query(self, *args: object, **kwargs: object) -> list[StoredChunk]:  # pragma: no cover
            raise AssertionError("not used")

        def get(self, accession: str, chunk_id: int) -> StoredChunk | None:
            raise RuntimeError("connection reset")

    content_section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="We rate the shares a Buy.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=391_035_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )
    infra_section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(Claim(text="Supply chain risk.", chunks=(ChunkRef(accession="acc", chunk_id=0),)),),
    )

    content_violations = verify_section(
        content_section, InMemoryChunkStore(), FACTS, accessions=ACC
    )
    infra_violations = verify_section(
        infra_section,
        ExplodingStore(),
        FACTS,
        accessions=ACC,  # type: ignore[arg-type]
    )

    assert all(v.category == "content" for v in content_violations)
    assert content_violations  # sanity: the Buy language was actually caught
    assert all(v.category == "infrastructure" for v in infra_violations)


# --- Warning 9: a punctuated number is a magnitude, not a year -------------


def test_a_comma_punctuated_year_range_number_is_treated_as_a_magnitude() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="The warehouse processed 2,024 orders that day.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["unsupported_figure"]
    assert violations[0].detail == "2024"


def test_a_bare_unpunctuated_year_is_still_exempt() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="In fiscal 2024 the warehouse expanded operations.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    assert verify_section(section, store, FACTS, accessions=ACC) == []


# --- Warning 10: honest "never raises" -- including malformed structure ----


def test_verify_section_never_raises_when_the_store_errors() -> None:
    class ExplodingStore:
        def upsert(self, chunks: object, vectors: object) -> None:  # pragma: no cover
            raise AssertionError("not used")

        def query(self, *args: object, **kwargs: object) -> list[StoredChunk]:  # pragma: no cover
            raise AssertionError("not used")

        def get(self, accession: str, chunk_id: int) -> StoredChunk | None:
            raise RuntimeError("connection reset")

    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="Supply chain concentration is a material risk.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
            Claim(
                text="Revenue was 391,035 million USD.",
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=391_035_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )

    violations = verify_section(
        section,
        ExplodingStore(),
        FACTS,
        accessions=ACC,  # type: ignore[arg-type]
    )

    assert [v.kind for v in violations] == ["verification_error"]
    assert violations[0].category == "infrastructure"


def test_verify_section_never_raises_on_a_structurally_malformed_section() -> None:
    # Type-correct input today, but Task 16's CI gate deserialises fixtures
    # from JSON -- exactly where malformed structure would arrive in
    # practice, bypassing the Pydantic validation a normal ResearchNote goes
    # through. The whole point of this module is that it degrades instead of
    # crashing, so this has to actually be exercised, not just claimed.
    violations = verify_section(None, InMemoryChunkStore(), FACTS, accessions=ACC)  # type: ignore[arg-type]

    assert [v.kind for v in violations] == ["verification_error"]
    assert violations[0].category == "infrastructure"


def test_verify_note_never_raises_on_a_structurally_malformed_note() -> None:
    violations = verify_note(None, InMemoryChunkStore(), FACTS)  # type: ignore[arg-type]

    assert [v.kind for v in violations] == ["verification_error"]
    assert violations[0].category == "infrastructure"


# --- Violation locator -------------------------------------------------------


def test_violations_carry_their_section_and_claim_index() -> None:
    store = _store_with("Supply chain concentration is a material risk.")
    section = Section(
        name=SectionName.RISK_FACTORS,
        claims=(
            Claim(
                text="Supply chain concentration is a material risk.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
            Claim(
                text="We rate the shares a Buy.",
                chunks=(ChunkRef(accession="acc", chunk_id=0),),
            ),
        ),
    )

    violations = verify_section(section, store, FACTS, accessions=ACC)

    assert [v.kind for v in violations] == ["recommendation_language"]
    assert violations[0].section == SectionName.RISK_FACTORS
    assert violations[0].claim_index == 1


# --- scale-blindness regression (the 1000x hole) ---------------------------
#
# The verifier once expanded an XBRL fact downward across scales and matched
# any claim candidate against any reference candidate. A claim with the right
# mantissa and the wrong magnitude word therefore passed: "Revenue was 391
# million USD" was accepted against a true $391,035,000,000, because the
# expansion produced 391,035,000 and the mantissa agreed inside tolerance.
# That is a 1000x error through the guarantee the module exists to provide.


def _revenue_claim(text: str) -> Section:
    return Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text=text,
                facts=(
                    FactRef(
                        tag="Revenues",
                        fiscal_period="FY2024",
                        value=391_035_000_000.0,
                        accession="acc",
                    ),
                ),
            ),
        ),
    )


def test_a_claim_naming_the_wrong_magnitude_is_rejected() -> None:
    store = _store_with("Revenue is discussed below.")

    for wrong in ("Revenue was 391 million USD.", "Revenue was 391 thousand USD."):
        violations = verify_section(_revenue_claim(wrong), store, FACTS, accessions=ACC)
        assert [v.kind for v in violations] == ["unsupported_figure"], wrong


def test_a_claim_naming_the_right_magnitude_is_accepted() -> None:
    store = _store_with("Revenue is discussed below.")

    assert (
        verify_section(_revenue_claim("Revenue was 391 billion USD."), store, FACTS, accessions=ACC)
        == []
    )


def test_a_figure_quoted_at_the_filings_presented_scale_is_accepted() -> None:
    # Filings state revenue in millions; XBRL states it in dollars. Both
    # spellings name the same figure and both must pass.
    store = _store_with("Revenue is discussed below.")

    for correct in ("Revenue was 391,035 million USD.", "Revenue was 391,035,000,000 USD."):
        assert verify_section(_revenue_claim(correct), store, FACTS, accessions=ACC) == [], correct


def test_a_scale_word_cannot_be_bolted_onto_a_chunks_bare_number() -> None:
    # The chunk says 4,200. A claim may restate 4,200, but not reinterpret it
    # as 4,200 million -- the source never asserted that magnitude.
    store = _store_with("The company closed 4,200 stores during the year.")

    def closed(text: str) -> Section:
        return Section(
            name=SectionName.BUSINESS_OVERVIEW,
            claims=(Claim(text=text, chunks=(ChunkRef(accession="acc", chunk_id=0),)),),
        )

    assert (
        verify_section(closed("The company closed 4,200 stores."), store, FACTS, accessions=ACC)
        == []
    )
    for inflated in ("The company closed 4,200 million stores.", "4,200 billion stores closed."):
        violations = verify_section(closed(inflated), store, FACTS, accessions=ACC)
        assert [v.kind for v in violations] == ["unsupported_figure"], inflated


# --- no-advice false positives on ordinary filing prose --------------------
#
# An earlier pattern matched the bare noun "rate", which appears constantly in
# the financial-health section this agent writes every run. False positives
# there burn the retry budget continuously and block the Tier 1 merge gate.


def test_ordinary_financial_prose_is_not_treated_as_advice() -> None:
    for ordinary in (
        "Our effective tax rate benefited from cash we hold overseas.",
        "The interest rate on the notes we hold is fixed until 2030.",
        "Foreign exchange rate exposure relates to assets we will sell.",
        "The board will hold a vote and shareholders may sell shares.",
    ):
        assert no_recommendation_language(ordinary) == [], ordinary


def test_actual_recommendations_are_still_caught() -> None:
    for advice in (
        "We rate the shares a Buy.",
        "The stock is rated a Buy.",
        "Investors should buy the shares.",
        "Our price target is 250 USD.",
    ):
        assert no_recommendation_language(advice), advice
