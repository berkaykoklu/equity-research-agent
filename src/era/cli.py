"""The `era` command-line entry point.

This is the only module in the project that prints or formats -- every
module beneath it (era.index.ingest included) returns plain data so the
same logic could one day back a different interface without dragging
formatting along with it.

`era ingest` is deliberately the only place ingestion can be triggered from.
Fetching a filing, choosing its boundaries and embedding its chunks takes
minutes and spends real money (a boundary-selection call per filing here,
Voyage embedding calls for every chunk); none of that may ever run inside a
web request. The deployed application serves only what a deliberate
`era ingest` run has already indexed -- keep it that way when adding
`era research` (Task 15) or any other command alongside this one.
"""

from pathlib import Path

import typer

from era.config import Settings
from era.edgar.boundaries import CachedBoundarySelector, LlmBoundarySelector
from era.edgar.client import EdgarClient
from era.edgar.filings import MissingFilingError
from era.index.embeddings import VoyageEmbedder
from era.index.ingest import IngestResult, ingest_ticker
from era.index.store import PgVectorStore

app = typer.Typer(help="Cited equity research from SEC filings.")

# Two independent caches, both safe to keep forever for the reason each of
# their owning modules documents: EdgarClient never re-fetches a published
# filing, and CachedBoundarySelector never re-asks the model about a
# document it has already decided. Splitting them under one root just keeps
# `.cache/` tidy, not because either module cares about the other's files.
CACHE_ROOT = Path(".cache")
EDGAR_CACHE_DIR = CACHE_ROOT / "edgar"
BOUNDARY_CACHE_DIR = CACHE_ROOT / "boundaries"


def _report(result: IngestResult) -> None:
    typer.echo(f"CIK {result.cik}")
    typer.echo(f"filings: {', '.join(result.accessions)}")
    typer.echo(f"chunks written: {result.chunks_written}")
    if result.items_missing:
        typer.echo("items not found:")
        for accession, items in sorted(result.items_missing.items()):
            typer.echo(f"  {accession}: {', '.join(items)}")


@app.command()
def ingest(ticker: str) -> None:
    """Fetch, parse, chunk and index a company's latest filings."""
    settings = Settings()  # type: ignore[call-arg]
    selector = CachedBoundarySelector(LlmBoundarySelector(), BOUNDARY_CACHE_DIR)

    with (
        EdgarClient(user_agent=settings.edgar_user_agent, cache_dir=EDGAR_CACHE_DIR) as client,
        PgVectorStore(dsn=settings.database_url) as store,
    ):
        try:
            result = ingest_ticker(
                ticker,
                client,
                selector,
                VoyageEmbedder(api_key=settings.voyage_api_key),
                store,
            )
        except MissingFilingError as exc:
            # Let the specific, entity-naming message through rather than a
            # bare traceback -- but still exit non-zero, so a missing 10-K
            # never looks like a quiet, successful no-op to a caller
            # scripting against this command.
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

    _report(result)
