from dataclasses import dataclass

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
    # a caller whether there is anything to report at all.
    items_missing: dict[str, tuple[str, ...]]


def ingest_ticker(
    ticker: str,
    client: EdgarClient,
    selector: BoundarySelector,
    embedder: Embedder,
    store: ChunkStore,
) -> IngestResult:
    """Fetch, parse, chunk, embed and index a ticker's latest filings.

    This is the only place fetching, embedding and storing all happen for
    real -- and it is only ever reachable from `era ingest` (see
    era.cli). A 10-K takes minutes to process and spends real money on
    both the boundary selector and the embedder, so nothing may call this
    from a request handler; the deployed app only ever serves what has
    already been indexed by a deliberate `era ingest` run.

    latest_filings raises MissingFilingError when the issuer has no 10-K on
    file. That is left to propagate rather than caught here: "no annual
    report exists" is a fact worth surfacing as a real failure, not a
    result to silently paper over with an empty IngestResult.
    """
    cik = resolve_cik(client, ticker)
    filings = latest_filings(client, cik)

    accessions: list[str] = []
    items_missing: dict[str, tuple[str, ...]] = {}
    chunks_written = 0

    for filing in filings:
        html = client.get_text(filing.primary_document_url)
        # A CachedBoundarySelector (constructed once in era.cli, wrapping
        # the real LlmBoundarySelector) keys its cache on the document
        # itself, so replaying ingest against a filing already seen never
        # re-spends a boundary-selection call.
        parsed = parse_items(html, selector)
        if parsed.missing:
            items_missing[filing.accession] = parsed.missing

        chunks = chunk_items(filing.accession, parsed.items)
        if chunks:
            vectors = embedder.embed([chunk.text for chunk in chunks])
            store.upsert(chunks, vectors)
        chunks_written += len(chunks)
        accessions.append(filing.accession)

    return IngestResult(
        cik=cik,
        accessions=tuple(accessions),
        chunks_written=chunks_written,
        items_missing=items_missing,
    )
