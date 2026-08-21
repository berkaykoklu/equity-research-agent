from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from era.edgar.boundaries import BoundarySelector
from era.edgar.client import EdgarClient
from era.edgar.filings import latest_filings, resolve_cik
from era.edgar.sections import parse_items
from era.index.chunking import chunk_items
from era.index.embeddings import Embedder
from era.index.store import ChunkStore


@dataclass(frozen=True)
class IngestResult:
    cik: str
    accessions: tuple[str, ...]
    chunks_written: int
    # Keyed by accession, not flattened across filings: a 10-K and a 10-Q in
    # the same run can have different gaps, and "item 7 is missing" is only
    # actionable if the reader also knows which filing it's missing from.
    # An accession with no gaps is simply absent from this dict rather than
    # mapped to an empty tuple, so `bool(result.items_missing)` alone tells
    # a caller whether there is anything to report at all. Mapping (backed
    # by MappingProxyType, not dict) so a caller can't mutate a result after
    # the fact -- `frozen=True` alone only stops reassigning the field, not
    # mutating what it points to.
    items_missing: Mapping[str, tuple[str, ...]]
    # accession -> "ExceptionType: message" for any filing whose processing
    # raised. A raise here is never allowed to abort the whole run or erase
    # filings that already succeeded -- see the try/except below -- so the
    # caller (era.cli) can still print a full report and still exit non-zero.
    failures: Mapping[str, str]


def ingest_ticker(
    ticker: str,
    client: EdgarClient,
    selector: BoundarySelector,
    embedder: Embedder,
    store: ChunkStore,
) -> IngestResult:
    """Fetch, parse, chunk, embed and index a ticker's latest 10-K.

    This is the only place fetching, embedding and storing all happen for
    real -- and it is only ever reachable from `era ingest` (see
    era.cli). A 10-K takes minutes to process and spends real money on
    both the boundary selector and the embedder, so nothing may call this
    from a request handler; the deployed app only ever serves what has
    already been indexed by a deliberate `era ingest` run.

    latest_filings raises MissingFilingError when the issuer has no 10-K on
    file, and resolve_cik raises UnknownTickerError for a ticker SEC doesn't
    recognize. Both are left to propagate rather than caught here: each is a
    fact worth surfacing as a real failure, not a result to silently paper
    over with an empty IngestResult.

    A single filing's own processing failing (a bad fetch, a store outage
    mid-run) is different -- caught per filing below, not propagated, so one
    bad filing can't erase or block filings that already succeeded.
    """
    cik = resolve_cik(client, ticker)

    # 10-Q support is structurally broken today, not just unimplemented: a
    # 10-Q's Item 1 is titled "Financial Statements", which never matches
    # era.edgar.sections.EXPECTED_OPENING["1"] ("business"); Item 7 does not
    # exist in a 10-Q at all; and the section that actually carries
    # quarterly commentary -- Item 2, MD&A -- isn't even a selectable start
    # in era.edgar.boundaries' response schema. Ingesting a 10-Q today would
    # produce zero chunks and a "missing: 1, 1A, 7" line on every single
    # run, training an operator to ignore the one field that reports real
    # gaps. Filtering to the 10-K here is deliberate: real 10-Q support
    # needs Item 2 added to WANTED_ITEMS and a 10-Q-shaped EXPECTED_OPENING
    # -- a future task, not this one.
    filings = [f for f in latest_filings(client, cik) if f.form == "10-K"]

    accessions: list[str] = []
    items_missing: dict[str, tuple[str, ...]] = {}
    failures: dict[str, str] = {}
    chunks_written = 0

    for filing in filings:
        try:
            html = client.get_text(filing.primary_document_url)
            # A CachedBoundarySelector (constructed once in era.cli, wrapping
            # the real LlmBoundarySelector) keys its cache on the document
            # itself, so replaying ingest against a filing already seen
            # never re-spends a boundary-selection call.
            parsed = parse_items(html, selector)
            if parsed.missing:
                items_missing[filing.accession] = parsed.missing

            chunks = chunk_items(filing.accession, parsed.items)
            if chunks:
                vectors = embedder.embed([chunk.text for chunk in chunks])
                store.upsert(chunks, vectors)
            chunks_written += len(chunks)
            accessions.append(filing.accession)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: a
            # network blip, a store outage, an embedder error -- none of it
            # may abort the run or hide a filing that already succeeded.
            # era.cli prints this and exits non-zero rather than letting a
            # partial run look like either a full success or a bare crash.
            failures[filing.accession] = f"{type(exc).__name__}: {exc}"

    return IngestResult(
        cik=cik,
        accessions=tuple(accessions),
        chunks_written=chunks_written,
        items_missing=MappingProxyType(items_missing),
        failures=MappingProxyType(failures),
    )
