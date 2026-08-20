"""Tests for era.cli's wiring and output formatting.

Every dependency era.cli constructs for real -- EdgarClient, PgVectorStore,
VoyageEmbedder, the boundary selector -- is monkeypatched here. Constructing
a real PgVectorStore alone opens a live database connection, and a real
LlmBoundarySelector eventually calls OpenAI; neither may ever happen in this
suite (see the project's $7 budget). ingest_ticker itself is also
monkeypatched, so these tests exercise only what era.cli owns: wiring
Settings' fields to the right constructor arguments, and turning an
IngestResult into the text a human reads.
"""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import era.cli as cli
from era.edgar.filings import MissingFilingError
from era.index.ingest import IngestResult

runner = CliRunner()


class _DummySettings:
    edgar_user_agent = "Test Bot test@example.com"
    voyage_api_key = "voyage-fake"
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


class _DummySelector:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_real_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    # Autouse: every test in this module gets the same safety net, so a new
    # test added later can't forget to patch a construction path and
    # accidentally open a real database connection or call OpenAI.
    monkeypatch.setattr(cli, "Settings", lambda: _DummySettings())
    monkeypatch.setattr(cli, "EdgarClient", _DummyEdgarClient)
    monkeypatch.setattr(cli, "PgVectorStore", _DummyPgVectorStore)
    monkeypatch.setattr(cli, "VoyageEmbedder", _DummyVoyageEmbedder)
    monkeypatch.setattr(cli, "CachedBoundarySelector", _DummySelector)
    monkeypatch.setattr(cli, "LlmBoundarySelector", _DummySelector)


def test_ingest_reports_cik_filings_and_chunk_count(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_ingest_ticker(
        ticker: str, client: object, selector: object, embedder: object, store: object
    ) -> IngestResult:
        captured["ticker"] = ticker
        captured["client"] = client
        captured["store"] = store
        captured["embedder"] = embedder
        return IngestResult(
            cik="0000320193",
            accessions=("0000320193-24-000123",),
            chunks_written=7,
            items_missing={},
        )

    monkeypatch.setattr(cli, "ingest_ticker", _fake_ingest_ticker)

    result = runner.invoke(cli.app, ["AAPL"])

    assert result.exit_code == 0
    assert "CIK 0000320193" in result.stdout
    assert "0000320193-24-000123" in result.stdout
    assert "chunks written: 7" in result.stdout
    assert "items not found" not in result.stdout
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


def test_ingest_reports_which_items_are_missing_and_from_which_filing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_ingest_ticker(*args: object, **kwargs: object) -> IngestResult:
        return IngestResult(
            cik="0000320193",
            accessions=("0000320193-24-000123",),
            chunks_written=4,
            items_missing={"0000320193-24-000123": ("7",)},
        )

    monkeypatch.setattr(cli, "ingest_ticker", _fake_ingest_ticker)

    result = runner.invoke(cli.app, ["AAPL"])

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

    result = runner.invoke(cli.app, ["NOFILING"])

    assert result.exit_code == 1
    assert "has filed no 10-K" in result.output
    # A failure must never also print a success-shaped report.
    assert "chunks written" not in result.output
