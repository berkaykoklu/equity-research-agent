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
from era.index.embeddings import VoyageEmbedder
from era.index.ingest import IngestResult, ingest_ticker
from era.index.store import PgVectorStore

app = typer.Typer(help="Cited equity research from SEC filings.")


@app.callback()
def _main() -> None:
    """Cited equity research from SEC filings."""
    # A Typer app with exactly one @app.command() collapses to top-level
    # arguments -- `era AAPL` instead of `era ingest AAPL` -- unless a
    # callback exists to force subcommand mode. This callback does nothing
    # else and must stay in place once `era research` (Task 15) adds a
    # second command, not be treated as safe to remove because there are
    # "enough" commands to keep subcommand mode on their own.


# Shared with tests/edgar/test_sections_live.py's CACHE_DIR/BOUNDARY_CACHE_DIR
# on purpose: a real `era ingest` run and the live validation suite fetch the
# same filings and make the same boundary decisions, so pointing both at the
# same cache means the first real ingest doesn't re-download or re-decide
# anything a prior live-test run already paid for.
CACHE_ROOT = Path(".cache")
EDGAR_CACHE_DIR = CACHE_ROOT / "edgar-live"
BOUNDARY_CACHE_DIR = CACHE_ROOT / "boundaries"


def _report(result: IngestResult) -> None:
    typer.echo(f"CIK {result.cik}")
    typer.echo(f"filings: {', '.join(result.accessions)}")
    typer.echo(f"chunks written: {result.chunks_written}")
    if result.items_missing:
        typer.echo("items not found:")
        for accession, items in sorted(result.items_missing.items()):
            typer.echo(f"  {accession}: {', '.join(items)}")
    if result.failures:
        typer.echo("filings that failed:")
        for accession, reason in sorted(result.failures.items()):
            typer.echo(f"  {accession}: {reason}")


@app.command()
def ingest(ticker: str) -> None:
    """Fetch, parse, chunk and index a company's latest 10-K."""
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
        except LookupError as exc:
            # Covers both UnknownTickerError (bad ticker) and
            # MissingFilingError (no 10-K on file) -- both are facts worth a
            # clear message and a nonzero exit, never a bare traceback.
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

    _report(result)

    # A required 10-K that contributes zero chunks -- whether every item
    # failed verification (see era.edgar.sections) or processing it raised
    # and landed in result.failures -- must not look like a quiet success.
    # This is the one command in the project that spends real money; a
    # silent no-op here is worse than a loud one.
    if result.failures or result.chunks_written == 0:
        raise typer.Exit(code=1)
