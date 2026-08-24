"""Storage for generated notes.

The deployed site reads from here and never writes. Producing a note costs real
money and takes ~80 seconds, so it happens deliberately from the CLI and the
result is stored; a visitor reads what is already there. That separation is the
budget guarantee, not an optimisation — an endpoint that could generate is an
endpoint that can be made to spend.

Same shape as `ChunkStore`: a Protocol with a real Postgres implementation and
an in-memory fake, so the API layer and its tests never need a database.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Protocol

from era.report.schema import ResearchNote

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    ticker       TEXT PRIMARY KEY,
    cik          TEXT NOT NULL,
    note         JSONB NOT NULL,
    markdown     TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL
);
"""


@dataclass(frozen=True)
class StoredNote:
    ticker: str
    cik: str
    note: ResearchNote
    markdown: str
    generated_at: datetime


class NoteStore(Protocol):
    def save(self, note: ResearchNote, markdown: str) -> None:
        """Replace any stored note for this ticker.

        One note per company: the site shows the current view of a filing, and
        keeping older ones would mean deciding which is authoritative.
        """
        ...

    def get(self, ticker: str) -> StoredNote | None: ...

    def list_tickers(self) -> list[StoredNote]: ...


class InMemoryNoteStore:
    """The fake. Behaviour must match PgNoteStore, including replacement."""

    def __init__(self) -> None:
        self._rows: dict[str, StoredNote] = {}

    def save(self, note: ResearchNote, markdown: str) -> None:
        self._rows[note.ticker.upper()] = StoredNote(
            ticker=note.ticker.upper(),
            cik=note.cik,
            note=note,
            markdown=markdown,
            generated_at=datetime.now(UTC),
        )

    def get(self, ticker: str) -> StoredNote | None:
        return self._rows.get(ticker.upper())

    def list_tickers(self) -> list[StoredNote]:
        return sorted(self._rows.values(), key=lambda row: row.ticker)


class PgNoteStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = self._connect()

    def _connect(self) -> Any:
        import psycopg

        conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=10)
        conn.execute(SCHEMA)
        return conn

    def _live(self) -> Any:
        # Same reason as PgVectorStore: Neon's free tier suspends a compute
        # after five minutes idle, and a note is saved at the end of an
        # ~80-second run that may follow a long ingest.
        import psycopg

        if self._conn.closed:
            self._conn = self._connect()
        try:
            self._conn.execute("SELECT 1")
        except psycopg.OperationalError:
            self._conn = self._connect()
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PgNoteStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def save(self, note: ResearchNote, markdown: str) -> None:
        conn = self._live()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO notes (ticker, cik, note, markdown, generated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    cik = EXCLUDED.cik,
                    note = EXCLUDED.note,
                    markdown = EXCLUDED.markdown,
                    generated_at = EXCLUDED.generated_at
                """,
                (
                    note.ticker.upper(),
                    note.cik,
                    note.model_dump_json(),
                    markdown,
                    datetime.now(UTC),
                ),
            )

    def _row_to_note(self, row: tuple[Any, ...]) -> StoredNote:
        stored = row[2]
        return StoredNote(
            ticker=row[0],
            cik=row[1],
            # psycopg returns JSONB already parsed; a driver that hands back a
            # string must still work, so accept both rather than assuming.
            note=(
                ResearchNote.model_validate(stored)
                if isinstance(stored, dict)
                else ResearchNote.model_validate_json(stored)
            ),
            markdown=row[3],
            generated_at=row[4],
        )

    def get(self, ticker: str) -> StoredNote | None:
        conn = self._live()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, cik, note, markdown, generated_at FROM notes WHERE ticker = %s",
                (ticker.upper(),),
            )
            row = cur.fetchone()
        return self._row_to_note(row) if row else None

    def list_tickers(self) -> list[StoredNote]:
        conn = self._live()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, cik, note, markdown, generated_at FROM notes ORDER BY ticker"
            )
            return [self._row_to_note(row) for row in cur.fetchall()]
