"""Tests for era.cli's wiring and output formatting.

Every dependency era.cli constructs for real -- EdgarClient, PgVectorStore,
VoyageEmbedder, the boundary selector -- is monkeypatched here. Constructing
a real PgVectorStore alone opens a live database connection, and a real
LlmBoundarySelector eventually calls OpenAI; neither may ever happen in this
suite (see the project's $7 budget). ingest_ticker itself is also
monkeypatched, so these tests exercise only what era.cli owns: wiring
Settings' fields (and the real selector construction) to the right
constructor arguments, and turning an IngestResult into the text and exit
code a human/script sees.
"""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import era.cli as cli
from era.edgar.filings import MissingFilingError, UnknownTickerError
from era.index.ingest import IngestResult

runner = CliRunner()


class _DummySettings:
    edgar_user_agent = "Test Bot test@example.com"
    voyage_api_key = "voyage-fake"
    openai_api_key = "sk-fake"
    database_url = "postgresql://fake"


class _DummyEdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path) -> None:
        self.user_agent = user_agent
        self.cache_dir = cache_dir

    def __enter__(self) -> "_DummyEdgarClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _DummyPgVectorStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def __enter__(self) -> "_DummyPgVectorStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _DummyVoyageEmbedder:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


class _DummyLlmSelector:
    """Distinguishable from _DummyCachedSelector so wiring can be asserted."""

    def __init__(self, model: object = None) -> None:
        # The CLI now builds the chat model itself and injects it, rather than
        # letting the selector construct one from the ambient environment.
        self.model = model


class _DummyCachedSelector:
    def __init__(self, inner: object, cache_dir: Path) -> None:
        self.inner = inner
        self.cache_dir = cache_dir


@pytest.fixture(autouse=True)
def _no_real_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    # Autouse: every test in this module gets the same safety net, so a new
    # test added later can't forget to patch a construction path and
    # accidentally open a real database connection or call OpenAI.
    monkeypatch.setattr(cli, "Settings", lambda: _DummySettings())
    monkeypatch.setattr(cli, "EdgarClient", _DummyEdgarClient)
    monkeypatch.setattr(cli, "PgVectorStore", _DummyPgVectorStore)
    monkeypatch.setattr(cli, "VoyageEmbedder", _DummyVoyageEmbedder)
    monkeypatch.setattr(cli, "CachedBoundarySelector", _DummyCachedSelector)
    monkeypatch.setattr(cli, "LlmBoundarySelector", _DummyLlmSelector)
    # The CLI hands the key from Settings to the provider client; the real
    # factory would try to construct one, so stub it out.
    monkeypatch.setattr(cli, "build_model", lambda **_: object())


def test_the_app_requires_a_subcommand_rather_than_collapsing_to_bare_arguments() -> None:
    # Task 10's original bug: a Typer app with exactly one @app.command()
    # collapses to top-level arguments ("era AAPL") unless a callback forces
    # subcommand mode, silently breaking every documented "era ingest ..."
    # invocation. This locks in the fix -- a bare ticker with no subcommand
    # must fail, and --help must list "ingest" as a real subcommand.
    bare = runner.invoke(cli.app, ["AAPL"])
    assert bare.exit_code != 0

    help_result = runner.invoke(cli.app, ["--help"])
    assert help_result.exit_code == 0
    assert "ingest" in help_result.output


def test_ingest_reports_cik_filings_and_chunk_count(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_ingest_ticker(
        ticker: str, client: object, selector: object, embedder: object, store: object
    ) -> IngestResult:
        captured["ticker"] = ticker
        captured["client"] = client
        captured["store"] = store
        captured["embedder"] = embedder
        captured["selector"] = selector
        return IngestResult(
            cik="0000320193",
            accessions=("0000320193-24-000123",),
            chunks_written=7,
            items_missing={},
            failures={},
        )

    monkeypatch.setattr(cli, "ingest_ticker", _fake_ingest_ticker)

    result = runner.invoke(cli.app, ["ingest", "AAPL"])

    assert result.exit_code == 0
    assert "CIK 0000320193" in result.stdout
    assert "0000320193-24-000123" in result.stdout
    assert "chunks written: 7" in result.stdout
    assert "items not found" not in result.stdout
    assert "failed" not in result.stdout
    assert captured["ticker"] == "AAPL"
    # Settings' fields must reach the right constructor argument, not just
    # any argument -- swapping voyage_api_key and database_url would still
    # type-check and still pass a test that only checked call counts.
    assert isinstance(captured["client"], _DummyEdgarClient)
    assert captured["client"].user_agent == "Test Bot test@example.com"
    assert isinstance(captured["store"], _DummyPgVectorStore)
    assert captured["store"].dsn == "postgresql://fake"
    assert isinstance(captured["embedder"], _DummyVoyageEmbedder)
    assert captured["embedder"].api_key == "voyage-fake"
    # Dropping CachedBoundarySelector and passing the raw LlmBoundarySelector
    # straight through would still type-check and still pass every other
    # assertion here -- but it re-spends a boundary-selection model call on
    # every filing, every run, against a hard $7 budget. This is the only
    # thing that would catch that regression.
    assert isinstance(captured["selector"], _DummyCachedSelector)
    assert isinstance(captured["selector"].inner, _DummyLlmSelector)
    assert captured["selector"].cache_dir == cli.BOUNDARY_CACHE_DIR


def test_ingest_reports_which_items_are_missing_and_from_which_filing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_ingest_ticker(*args: object, **kwargs: object) -> IngestResult:
        return IngestResult(
            cik="0000320193",
            accessions=("0000320193-24-000123",),
            chunks_written=4,
            items_missing={"0000320193-24-000123": ("7",)},
            failures={},
        )

    monkeypatch.setattr(cli, "ingest_ticker", _fake_ingest_ticker)

    result = runner.invoke(cli.app, ["ingest", "AAPL"])

    assert result.exit_code == 0
    assert "items not found:" in result.stdout
    assert "0000320193-24-000123: 7" in result.stdout


def test_ingest_exits_nonzero_and_names_the_entity_when_no_10k_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_ingest_ticker(*args: object, **kwargs: object) -> IngestResult:
        raise MissingFilingError(
            "Some Corp (CIK 0000000001) has filed no 10-K. Forms on file: 8-K."
        )

    monkeypatch.setattr(cli, "ingest_ticker", _fake_ingest_ticker)

    result = runner.invoke(cli.app, ["ingest", "NOFILING"])

    assert result.exit_code == 1
    assert "has filed no 10-K" in result.output
    # A failure must never also print a success-shaped report.
    assert "chunks written" not in result.output


def test_ingest_exits_nonzero_and_reports_a_typo_d_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    # UnknownTickerError is a LookupError, same as MissingFilingError, but
    # was not caught by the original except clause -- a typo'd ticker
    # produced a raw traceback while a missing 10-K got a clean message.
    def _fake_ingest_ticker(*args: object, **kwargs: object) -> IngestResult:
        raise UnknownTickerError("NOTATICKER is not a US-listed issuer in SEC's index")

    monkeypatch.setattr(cli, "ingest_ticker", _fake_ingest_ticker)

    result = runner.invoke(cli.app, ["ingest", "NOTATICKER"])

    assert result.exit_code == 1
    assert "not a US-listed issuer" in result.output
    assert "chunks written" not in result.output


def test_ingest_exits_nonzero_when_a_filing_fails_but_still_prints_what_succeeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A store outage on one filing must not look like a bare crash (no
    # report at all) or a quiet success (exit 0). The operator needs both:
    # what did succeed, printed, and a nonzero exit saying not everything did.
    def _fake_ingest_ticker(*args: object, **kwargs: object) -> IngestResult:
        return IngestResult(
            cik="0000320193",
            accessions=("0000320193-24-000123",),
            chunks_written=3,
            items_missing={},
            failures={"0000320193-23-000456": "RuntimeError: simulated store outage"},
        )

    monkeypatch.setattr(cli, "ingest_ticker", _fake_ingest_ticker)

    result = runner.invoke(cli.app, ["ingest", "AAPL"])

    assert result.exit_code == 1
    assert "chunks written: 3" in result.output
    assert "0000320193-23-000456: RuntimeError: simulated store outage" in result.output


def test_ingest_exits_nonzero_when_the_required_10k_contributes_zero_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The GE/INTC shape: every item fails verification, so processing
    # "succeeds" with zero rows written and no exception raised at all.
    # That must not exit 0 -- a required 10-K contributing nothing is
    # exactly the quiet-success case this command's own docstring refuses
    # to accept for a missing 10-K.
    def _fake_ingest_ticker(*args: object, **kwargs: object) -> IngestResult:
        return IngestResult(
            cik="0000040545",
            accessions=("0000040545-24-000012",),
            chunks_written=0,
            items_missing={"0000040545-24-000012": ("1", "1A", "7")},
            failures={},
        )

    monkeypatch.setattr(cli, "ingest_ticker", _fake_ingest_ticker)

    result = runner.invoke(cli.app, ["ingest", "GE"])

    assert result.exit_code == 1
    assert "chunks written: 0" in result.output


def test_ingest_names_the_failure_when_settings_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Settings(), EdgarClient(...), PgVectorStore(...) and
    # LlmBoundarySelector() are all constructed outside ingest_ticker, in
    # era.cli.ingest itself -- an unset EDGAR_USER_AGENT raises pydantic's
    # ValidationError right here, before the per-filing try/except inside
    # ingest_ticker ever gets a chance to run. Without a try wrapping this
    # construction, the CliRunner would show a raw traceback (or, outside a
    # test harness, print one directly) with no friendly message at all.
    class _ExplodingSettings:
        def __init__(self) -> None:
            raise ValueError("edgar_user_agent\n  Field required")

    monkeypatch.setattr(cli, "Settings", _ExplodingSettings)

    result = runner.invoke(cli.app, ["ingest", "AAPL"])

    assert result.exit_code == 1
    assert "ingest failed" in result.stderr
    assert "Field required" in result.stderr
    # The friendly one-liner must never land on stdout, the stream a report
    # or a script parsing this command's output would read.
    assert "ingest failed" not in result.stdout


def test_ingest_names_the_failure_when_the_store_cannot_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A malformed or unreachable DATABASE_URL raises inside PgVectorStore's
    # constructor -- also outside ingest_ticker, also before the per-filing
    # try/except exists to catch anything.
    class _ExplodingStore:
        def __init__(self, dsn: str) -> None:
            raise RuntimeError("connection to server failed")

    monkeypatch.setattr(cli, "PgVectorStore", _ExplodingStore)

    result = runner.invoke(cli.app, ["ingest", "AAPL"])

    assert result.exit_code == 1
    assert "ingest failed" in result.stderr
    assert "connection to server failed" in result.stderr
    # A failure this early must not also print a success-shaped report.
    assert "chunks written" not in result.output


def test_ingest_prints_the_construction_failure_traceback_to_stderr_not_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The one-line "ingest failed: ..." message is right for a config typo,
    # but an unanticipated bug (an AttributeError while constructing
    # Settings, the selector, the EDGAR client or the store) needs the full
    # traceback to be locatable -- mirroring ingest_ticker's own per-filing
    # handler, which already prints one to stderr. Nothing pinned the
    # destination: swapping file=sys.stderr for file=sys.stdout in the CLI's
    # except block would still leave every other test in this module
    # passing, since none of them separate the two streams. This one does.
    class _ExplodingSettings:
        def __init__(self) -> None:
            raise ValueError("edgar_user_agent\n  Field required")

    monkeypatch.setattr(cli, "Settings", _ExplodingSettings)

    result = runner.invoke(cli.app, ["ingest", "AAPL"])

    assert result.exit_code == 1
    assert "Traceback (most recent call last)" in result.stderr
    assert "ValueError" in result.stderr
    assert "Traceback" not in result.stdout
