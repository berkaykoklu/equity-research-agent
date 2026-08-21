from tests.fakes import FakeEmbedder, InMemoryChunkStore

from era.edgar.xbrl import MetricValue, NormalizedFacts
from era.index.chunking import Chunk
from era.index.store import StoredChunk
from era.report.schema import ChunkRef, Claim, Coverage, FactRef, ResearchNote, Section, SectionName
from era.verify.checks import Violation, no_recommendation_language, verify_note, verify_section

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


def _store_with(text: str) -> InMemoryChunkStore:
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=0, accession="acc", item="1A", text=text, start=0)
    store.upsert([chunk], FakeEmbedder().embed([text]))
    return store


# --- brief's baseline tests -------------------------------------------------


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

    assert verify_section(section, store, FACTS) == []


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

    violations = verify_section(section, store, FACTS)

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

    violations = verify_section(section, store, FACTS)

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

    assert verify_section(section, store, FACTS) == []


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

    assert verify_section(section, InMemoryChunkStore(), FACTS) == []


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

    violations = verify_section(section, InMemoryChunkStore(), FACTS)

    assert [v.kind for v in violations] == ["figure_disagrees_with_xbrl"]


def test_flags_recommendation_language() -> None:
    assert no_recommendation_language("We rate the shares a Buy.")
    assert no_recommendation_language("Our price target is 250 USD.")
    assert no_recommendation_language("Revenue grew in fiscal 2024.") == []


# --- fiscal-period matching (tag alone is not enough) -----------------------


def test_rejects_a_fact_whose_period_disagrees_with_the_claim() -> None:
    # The tag is right and the value happens to equal the FY2024 figure, but
    # the claim asserts it belongs to FY2023 -- a genuine mislabeling that a
    # tag-only match would silently accept. Because the fact fails to
    # resolve, it also never contributes to supported_figures, so the
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

    violations = verify_section(section, InMemoryChunkStore(), FACTS)

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

    violations = verify_section(section, InMemoryChunkStore(), FACTS)

    assert [v.kind for v in violations] == ["unknown_fact_tag"]


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

    violations = verify_section(section, store, FACTS)

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

    assert verify_section(section, store, FACTS) == []


def test_fact_within_tolerance_is_accepted() -> None:
    # 0.4% off -- inside the 0.5% relative tolerance. The prose states the
    # figure in "billion" words so the literal digits captured from the text
    # ("392") stay below the materiality floor -- this test is only about
    # the fact/XBRL tolerance check, not the separate text-figure check.
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

    assert verify_section(section, InMemoryChunkStore(), FACTS) == []


def test_fact_just_outside_tolerance_is_rejected() -> None:
    # 391_035_000_000 * 1.006 -- just over the 0.5% relative tolerance.
    section = Section(
        name=SectionName.FINANCIAL_HEALTH,
        claims=(
            Claim(
                text="Revenue was about 393 billion USD.",
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

    violations = verify_section(section, InMemoryChunkStore(), FACTS)

    assert [v.kind for v in violations] == ["figure_disagrees_with_xbrl"]


def test_fact_accepted_at_thousands_and_billions_scale_too() -> None:
    # _fact_spellings must cover every scale a filing conventionally uses,
    # not just millions -- a fixture with only one scale exercised would pass
    # even if the other branches of the scale loop were deleted. Each case
    # uses its own NormalizedFacts so the claimed figure, the XBRL value, and
    # the scale under test all line up exactly.
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

    # A trillion-scale value so that quoting it in billions ("1,234 billion")
    # still clears the materiality floor and actually exercises the /1e9
    # branch of _fact_spellings, unlike a realistic (sub-materiality) figure.
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

    assert verify_section(thousands, InMemoryChunkStore(), thousands_facts) == []
    assert verify_section(billions, InMemoryChunkStore(), billions_facts) == []


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

    violations = verify_section(section, store, FACTS)

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

    violations = verify_section(section, store, FACTS)

    assert {v.kind for v in violations} == {"recommendation_language", "unsupported_figure"}
    assert len(violations) == 2


def test_recommendation_word_embedded_in_a_longer_word_is_not_flagged() -> None:
    # "holdings" contains "hold" but is not the word "hold" -- \b must stop it.
    assert no_recommendation_language("The fund reported strong holdings this quarter.") == []


def test_recommendation_word_as_a_standalone_word_is_flagged() -> None:
    text = "Analysts believe investors should hold the stock through year end."
    violations = no_recommendation_language(text)
    assert [v.detail for v in violations] == ["hold"]


def test_figure_supported_by_chunk_text_not_by_a_fact() -> None:
    # The 4,200 figure is corroborated by the cited chunk's own prose, with
    # no FactRef involved at all.
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

    assert verify_section(section, store, FACTS) == []


# --- structural behaviour ----------------------------------------------------


def test_unavailable_section_produces_no_violations() -> None:
    section = Section(
        name=SectionName.RISK_FACTORS,
        available=False,
        unavailable_reason="Item 1A boundary not found in filing",
    )

    assert verify_section(section, InMemoryChunkStore(), FACTS) == []


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
    )

    violations = verify_note(note, store, FACTS)

    assert [v.kind for v in violations] == ["recommendation_language"]


def test_verify_section_never_raises_when_the_store_errors() -> None:
    # Named failure mode: an error path that is never handed malformed data
    # is an error path that can crash in production. Feed it a store whose
    # get() raises, and confirm the checker degrades to a Violation instead
    # of propagating -- and still checks the section's other claims.
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

    violations = verify_section(section, ExplodingStore(), FACTS)  # type: ignore[arg-type]

    kinds = [v.kind for v in violations]
    assert kinds == ["verification_error"]


def test_violation_is_a_frozen_comparable_value() -> None:
    assert Violation(kind="x", detail="y") == Violation(kind="x", detail="y")
