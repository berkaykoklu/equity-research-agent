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

import sys
import traceback
from pathlib import Path

import typer

from era.config import Settings
from era.edgar.boundaries import CachedBoundarySelector, LlmBoundarySelector
from era.edgar.client import EdgarClient
from era.graph.models import CHEAP_MODEL, build_model
from era.index.embeddings import VoyageEmbedder
from era.index.ingest import IngestResult, ingest_ticker
from era.index.notes import PgNoteStore
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
    # Everything that can raise on a first, misconfigured run lives inside
    # this try, not just ingest_ticker itself: Settings() raises pydantic's
    # ValidationError when EDGAR_USER_AGENT is unset; EdgarClient(...) raises
    # MissingUserAgentError when it's set but malformed (no contact info);
    # PgVectorStore(...) raises psycopg.OperationalError against an empty or
    # unreachable DATABASE_URL; and resolve_cik (inside ingest_ticker) raises
    # an httpx error when SEC is unreachable. era.cli is the only module
    # that prints or formats -- constructing any of these outside the try
    # would let their exceptions reach the user as a bare traceback instead
    # of a message naming what failed.
    try:
        settings = Settings()  # type: ignore[call-arg]
        # The key is handed over explicitly: it lives in Settings, while the
        # provider client reads the process environment, and nothing bridges
        # the two on its own.
        selector = CachedBoundarySelector(
            LlmBoundarySelector(
                build_model(model_name=CHEAP_MODEL, api_key=settings.openai_api_key)
            ),
            BOUNDARY_CACHE_DIR,
        )

        with (
            EdgarClient(user_agent=settings.edgar_user_agent, cache_dir=EDGAR_CACHE_DIR) as client,
            PgVectorStore(dsn=settings.database_url) as store,
        ):
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
    except Exception as exc:
        # Everything else that can go wrong constructing or running this
        # command: bad config (ValidationError, MissingUserAgentError), an
        # unreachable database (psycopg.OperationalError), being offline or
        # SEC being down (httpx errors) before the per-filing loop in
        # ingest_ticker even starts. Name the exception and its message on
        # one friendly line first -- that's what a config typo needs -- but
        # also print the full traceback to stderr, mirroring
        # ingest_ticker's own per-filing handler: an unanticipated bug here
        # (an AttributeError while constructing Settings, the selector, the
        # EDGAR client or the store) deserves the same "don't lose it"
        # treatment during a money-spending run, not just a tidy one-liner
        # with no way to locate it.
        typer.echo(f"ingest failed: {type(exc).__name__}: {exc}", err=True)
        traceback.print_exc(file=sys.stderr)
        raise typer.Exit(code=1) from exc

    _report(result)

    # A required 10-K that contributes zero chunks -- whether every item
    # failed verification (see era.edgar.sections) or processing it raised
    # and landed in result.failures -- must not look like a quiet success.
    # This is the one command in the project that spends real money; a
    # silent no-op here is worse than a loud one.
    if result.failures or result.chunks_written == 0:
        raise typer.Exit(code=1)


@app.command()
def research(
    ticker: str,
    save: bool = typer.Option(
        False,
        "--save",
        help="Store the note so the deployed site can serve it.",
    ),
) -> None:
    """Write a cited research note for an already-indexed company.

    Reads only what `era ingest` has already stored. Nothing here fetches or
    embeds a filing: retrieval is a database query, so a company that was never
    ingested reports that plainly rather than quietly indexing itself.
    """
    from era.edgar.filings import latest_filings, resolve_cik
    from era.edgar.xbrl import normalize_facts
    from era.graph.build import build_research_graph
    from era.graph.models import DRAFTING_MODEL
    from era.observability.tracing import measured
    from era.report.assemble import render_markdown
    from era.report.schema import revalidated

    try:
        settings = Settings()  # type: ignore[call-arg]
        with (
            EdgarClient(user_agent=settings.edgar_user_agent, cache_dir=EDGAR_CACHE_DIR) as client,
            PgVectorStore(dsn=settings.database_url) as store,
        ):
            cik = resolve_cik(client, ticker)
            annual = next(f for f in latest_filings(client, cik) if f.form == "10-K")
            facts = normalize_facts(
                client.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
            )
            graph = build_research_graph(
                build_model(model_name=DRAFTING_MODEL, api_key=settings.openai_api_key),
                store,
                VoyageEmbedder(api_key=settings.voyage_api_key),
                facts,
            )
            with measured(ticker.upper(), graph.get_graph(xray=True)) as (
                record,
                callbacks,
            ):
                result = graph.invoke(
                    {
                        "ticker": ticker.upper(),
                        "cik": cik,
                        "accession": annual.accession,
                    },
                    config={"callbacks": callbacks},
                )
    except LookupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"research failed: {type(exc).__name__}: {exc}", err=True)
        traceback.print_exc(file=sys.stderr)
        raise typer.Exit(code=1) from exc

    # revalidated, not model_copy: model_copy skips the schema's validators,
    # and the coverage-vs-sections invariant is the note's honesty metric.
    note = revalidated(
        result["note"],
        cost_usd=record.cost_usd,
        latency_seconds=record.latency_seconds,
    )
    markdown = render_markdown(note)
    typer.echo(markdown)

    if save:
        # Deliberate, never automatic. The deployed site serves what is stored,
        # so storing is publishing -- and publishing a note whose sections all
        # failed would put an empty document in front of a reader.
        if note.coverage.sections_available == 0:
            typer.echo("not saved: no section was available", err=True)
        else:
            with PgNoteStore(dsn=settings.database_url) as notes:
                notes.save(note, markdown)
            typer.echo(f"saved {note.ticker}", err=True)

    if note.coverage.sections_available == 0:
        # Every section failed. The note still renders and says so, but this is
        # not a success and must not exit as though it were.
        raise typer.Exit(code=1)
