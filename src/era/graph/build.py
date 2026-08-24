"""Assemble the research graph: fan out, verify, retry, finalise.

Shape:

    START ─┬─ business_overview ─┐
           ├─ financial_health ──┤
           ├─ risk_factors ──────┼─ verify ─┬─ retry ─┘ (bounded)
           ├─ recent_developments┤          └─ finalise ─ END
           └─ valuation_context ─┘

Three things here are load-bearing and easy to get wrong:

**Concurrent state writes need reducers.** Five section nodes run at once and
all write to `attempts`. Returning a whole dict from each would race; each node
returns only its own entry and a reducer merges them.

**Only content violations drive retries.** Task 11 tags a violation as
`content` (the claim is wrong) or `infrastructure` (the checker could not
finish -- a store outage, malformed input). Regenerating a section because
Postgres blinked would burn the retry budget without any possibility of
fixing the complaint.

**Retries are bounded.** An unbounded reflection loop is the standard way an
agent quietly spends a lot of money achieving nothing.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from era.edgar.xbrl import NormalizedFacts
from era.graph.section import SECTION_QUERIES, SectionRequest, research_section
from era.index.embeddings import Embedder
from era.index.store import ChunkStore
from era.report.schema import Coverage, ResearchNote, Section, SectionName
from era.verify.checks import Violation, verify_section

MAX_COMPLAINTS_PER_SECTION = 5


def _merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {**left, **right}


class ResearchState(TypedDict, total=False):
    ticker: str
    cik: str
    accession: str
    # Appended to by every section node and by retries. `finalise` keeps the
    # last entry per section, so a retry's output supersedes what it replaced.
    sections: Annotated[list[Section], operator.add]
    # Written concurrently by the fan-out, so it needs a reducer.
    attempts: Annotated[dict[str, int], _merge_counts]
    # Written only by `verify`, which is a single node -- replaced wholesale
    # each pass so a fixed complaint does not linger.
    complaints: dict[str, str]
    note: ResearchNote


def _content_complaint(violations: list[Violation]) -> str | None:
    """One line describing what a section must fix, or None if nothing can be.

    Infrastructure failures are deliberately excluded: they are real problems,
    but not ones the model can do anything about.
    """
    content = [v for v in violations if v.category == "content"]
    if not content:
        return None
    return "; ".join(f"{v.kind}: {v.detail}" for v in content[:MAX_COMPLAINTS_PER_SECTION])


def _latest_by_section(sections: list[Section]) -> dict[SectionName, Section]:
    latest: dict[SectionName, Section] = {}
    for section in sections:
        latest[section.name] = section
    return latest


def build_research_graph(
    model: BaseChatModel,
    store: ChunkStore,
    embedder: Embedder,
    facts: NormalizedFacts,
    max_retries: int = 2,
) -> Any:
    accessions = frozenset({facts.metrics[name].accession for name in facts.metrics})

    def _research(name: SectionName, state: ResearchState, complaint: str | None) -> Section:
        _, item = SECTION_QUERIES[name]
        return research_section(
            SectionRequest(
                name=name,
                ticker=state["ticker"],
                accession=state["accession"],
                item_filter=item,
                complaint=complaint,
            ),
            model,
            store,
            embedder,
            facts,
        )

    def make_section_node(name: SectionName) -> Any:
        def node(state: ResearchState) -> ResearchState:
            section = _research(name, state, complaint=None)
            return {"sections": [section], "attempts": {name.value: 1}}

        return node

    def verify(state: ResearchState) -> ResearchState:
        note_accessions = accessions | {state["accession"]}
        complaints: dict[str, str] = {}
        for name, section in _latest_by_section(state.get("sections", [])).items():
            if not section.available:
                # Nothing to regenerate: the source text was never found.
                continue
            violations = verify_section(
                section, store, facts, accessions=frozenset(note_accessions)
            )
            complaint = _content_complaint(violations)
            if complaint is not None:
                complaints[name.value] = complaint
        return {"complaints": complaints}

    def should_retry(state: ResearchState) -> str:
        complaints = state.get("complaints") or {}
        attempts = state.get("attempts") or {}
        retryable = [name for name in complaints if attempts.get(name, 0) <= max_retries]
        return "retry" if retryable else "finalise"

    def retry(state: ResearchState) -> ResearchState:
        complaints = state.get("complaints") or {}
        attempts = state.get("attempts") or {}
        produced: list[Section] = []
        counts: dict[str, int] = {}

        for value, complaint in complaints.items():
            if attempts.get(value, 0) > max_retries:
                continue
            name = SectionName(value)
            produced.append(_research(name, state, complaint=complaint))
            counts[value] = attempts.get(value, 0) + 1

        return {"sections": produced, "attempts": counts}

    def finalise(state: ResearchState) -> ResearchState:
        latest = _latest_by_section(state.get("sections", []))
        ordered = tuple(latest[name] for name in SectionName if name in latest)
        cited = {
            ref.accession for section in ordered for claim in section.claims for ref in claim.chunks
        }
        cited |= {
            fact.accession
            for section in ordered
            for claim in section.claims
            for fact in claim.facts
        }

        note = ResearchNote(
            ticker=state["ticker"],
            cik=state["cik"],
            sections=ordered,
            coverage=Coverage(
                # Must describe the sections actually attached -- the schema
                # rejects a note whose coverage disagrees with its contents.
                sections_available=sum(1 for section in ordered if section.available),
                sections_total=len(ordered),
                metrics_resolved=tuple(sorted(facts.metrics)),
                metrics_missing=facts.missing,
            ),
            accessions=tuple(sorted(cited)),
        )
        return {"note": note}

    graph = StateGraph(ResearchState)
    for name in SectionName:
        graph.add_node(name.value, make_section_node(name))
        graph.add_edge(START, name.value)
        graph.add_edge(name.value, "verify")

    graph.add_node("verify", verify)
    graph.add_node("retry", retry)
    graph.add_node("finalise", finalise)
    graph.add_conditional_edges("verify", should_retry, {"retry": "retry", "finalise": "finalise"})
    graph.add_edge("retry", "verify")
    graph.add_edge("finalise", END)

    return graph.compile()
