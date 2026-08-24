"""Research one section of the note.

Retrieval, one model call, and a translation back into verified-shape claims.

Two rules shape this module, and both exist to make Task 11's verifier able to
do its job:

1. **Retrieval is scoped to this note's filing.** The store holds many
   companies at once; the verifier rejects any citation outside the note's
   accessions. Retrieving unscoped would make every claim unciteable and drive
   the retry loop until the budget was gone.
2. **The model never supplies a figure.** It names a metric from the facts
   card ("revenue"); code looks that name up and attaches the real tag, period,
   value and accession. A model that cannot type a number cannot misquote one.
"""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from era.edgar.xbrl import NormalizedFacts, facts_card
from era.index.embeddings import Embedder
from era.index.store import ChunkStore, StoredChunk
from era.report.schema import ChunkRef, Claim, FactRef, Section, SectionName

# What to retrieve for each section, and which 10-K item it should come from.
# Financial-health, recent-developments and valuation all draw on Item 7 (MD&A)
# but ask it different questions, so each gets its own query rather than
# sharing one retrieval.
SECTION_QUERIES: dict[SectionName, tuple[str, str | None]] = {
    SectionName.BUSINESS_OVERVIEW: ("what the company does, its products and markets", "1"),
    SectionName.FINANCIAL_HEALTH: ("revenue, margins, cash flow and leverage", "7"),
    SectionName.RISK_FACTORS: ("principal risks and uncertainties", "1A"),
    SectionName.RECENT_DEVELOPMENTS: ("recent results and material changes", "7"),
    SectionName.VALUATION_CONTEXT: ("growth, profitability and capital returns", "7"),
}

TARGET_WORDS = 600

PROMPT = """You are writing the "{section}" section of an equity research note \
on {ticker}.

Write at most 6 claims, about {words} words in total. Every claim MUST cite at \
least one source:

- `chunk_ids`: the id of any excerpt below that the claim is drawn from.
- `metrics`: the NAME of any figure from the facts card that the claim uses.
  Give the name only. Never write the number yourself -- it is filled in from
  the filing's own data, so a figure you type would be wrong by definition.

State only what the excerpts and facts support. Do not recommend buying, \
selling or holding, and do not give a price target.

{facts_card}

Filing excerpts:
{excerpts}
"""

RETRY_NOTE = """
A previous attempt was rejected. Fix exactly this and change nothing else:
{complaint}
"""


class _DraftClaim(BaseModel):
    text: str
    chunk_ids: list[int] = Field(default_factory=list)
    # Metric *names* from the facts card, never values.
    metrics: list[str] = Field(default_factory=list)


class SectionDraft(BaseModel):
    claims: list[_DraftClaim] = Field(default_factory=list)


@dataclass(frozen=True)
class SectionRequest:
    name: SectionName
    ticker: str
    # The filing this note is about. Scopes retrieval and stamps every
    # citation, so the verifier can confirm sources belong to this company.
    accession: str
    item_filter: str | None
    k: int = 4
    complaint: str | None = None


def _excerpts(hits: list[StoredChunk]) -> str:
    return "\n\n".join(f"[chunk {hit.chunk_id} · item {hit.item}]\n{hit.text}" for hit in hits)


def _claim_from_draft(
    drafted: _DraftClaim,
    hits_by_id: dict[int, StoredChunk],
    facts: NormalizedFacts,
    accession: str,
) -> Claim | None:
    """Translate a drafted claim into a verified-shape Claim, or drop it.

    Anything the model referred to that does not exist -- a chunk id it was
    never shown, a metric name absent from the facts card -- is discarded here
    rather than passed on as a citation that cannot resolve. A claim left with
    no sources at all is dropped: the schema forbids it, and an unsourced
    sentence is exactly what this product refuses to publish.
    """
    chunks = tuple(
        ChunkRef(accession=accession, chunk_id=chunk_id)
        for chunk_id in dict.fromkeys(drafted.chunk_ids)
        if chunk_id in hits_by_id
    )

    seen_tags: set[str] = set()
    facts_used: list[FactRef] = []
    for name in dict.fromkeys(drafted.metrics):
        metric = facts.metrics.get(name)
        if metric is None or metric.tag in seen_tags:
            continue
        seen_tags.add(metric.tag)
        facts_used.append(
            FactRef(
                tag=metric.tag,
                fiscal_period=metric.fiscal_period,
                value=metric.value,
                # The accession XBRL itself reports for this figure, which is
                # not necessarily the filing we indexed prose from.
                accession=metric.accession,
            )
        )

    if not chunks and not facts_used:
        return None
    return Claim(text=drafted.text.strip(), chunks=chunks, facts=tuple(facts_used))


def research_section(
    request: SectionRequest,
    model: BaseChatModel,
    store: ChunkStore,
    embedder: Embedder,
    facts: NormalizedFacts,
) -> Section:
    query, _ = SECTION_QUERIES[request.name]
    vector = embedder.embed([query])[0]
    hits = store.query(
        vector,
        item_filter=request.item_filter,
        accession_filter=request.accession,
        k=request.k,
    )

    if not hits:
        return Section(
            name=request.name,
            available=False,
            unavailable_reason=(
                f"no indexed content for item {request.item_filter} in {request.accession}"
            ),
        )

    prompt = PROMPT.format(
        section=request.name.value.replace("_", " "),
        ticker=request.ticker,
        words=TARGET_WORDS,
        facts_card=facts_card(facts),
        excerpts=_excerpts(hits),
    )
    if request.complaint:
        prompt += RETRY_NOTE.format(complaint=request.complaint)

    draft = model.with_structured_output(SectionDraft).invoke(prompt)
    if not isinstance(draft, SectionDraft):
        return Section(
            name=request.name,
            available=False,
            unavailable_reason="model returned an unusable response shape",
        )

    hits_by_id = {hit.chunk_id: hit for hit in hits}
    claims = [
        claim
        for claim in (
            _claim_from_draft(drafted, hits_by_id, facts, request.accession)
            for drafted in draft.claims
        )
        if claim is not None
    ]

    if not claims:
        return Section(
            name=request.name,
            available=False,
            unavailable_reason="model produced no citable claims",
        )
    return Section(name=request.name, claims=tuple(claims))
