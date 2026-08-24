from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, TypeVar

from era.index.chunking import Chunk
from era.index.embeddings import DIMENSIONS

T = TypeVar("T")


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: int
    accession: str
    item: str
    # Offset into the item's text this chunk was sliced from (see
    # era.index.chunking.Chunk.start). A citation resolver needs the exact
    # source location, not just which chunk matched.
    start: int
    text: str


class VectorDimensionError(ValueError):
    """A vector's width didn't match DIMENSIONS.

    Without this check, a mis-sized embedder (wrong model, a stub client
    someone forgot to fix) becomes an opaque psycopg error partway through
    a transaction against the real store, or a silently-accepted vector
    against the fake -- either way, only discovered downstream. Both
    PgVectorStore and InMemoryChunkStore raise this at the store boundary
    instead.
    """


def require_matching_dimensions(vectors: list[list[float]]) -> None:
    for vector in vectors:
        if len(vector) != DIMENSIONS:
            raise VectorDimensionError(
                f"expected {DIMENSIONS}-dimensional vectors, got {len(vector)}"
            )


class ChunkStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace every stored row for the chunks' accession(s) with these.

        chunk_id is a counter assigned per chunking run (see
        era.index.chunking), not a stable identity: item boundaries are
        picked by a model, so re-parsing the same filing can legitimately
        produce a different chunk set. Implementations must delete every
        existing row for an accession before writing its new chunks --
        upserting row-by-row and leaving surplus old rows in place would let
        a stale chunk_id resolve to a different item's text after a
        re-index, and a citation verifier would silently pass it. A batch
        can carry chunks for more than one accession; every accession
        present must have its old rows deleted, not just the first chunk's.
        """
        ...

    def query(
        self,
        vector: list[float],
        item_filter: str | None,
        accession_filter: str | None,
        k: int,
    ) -> list[StoredChunk]:
        """Return up to k chunks nearest to vector, most similar first.

        accession_filter scopes retrieval to one filing. The store holds
        chunks from many companies at once; without this filter a query
        issued while writing a note about one company can retrieve -- and a
        citation can then point to -- a passage from a different company's
        filing. item_filter narrows further, to one item within that
        filing. Both are optional so a caller can search as broadly or as
        narrowly as the task needs. k must not be negative.
        """
        ...

    def get(self, accession: str, chunk_id: int) -> StoredChunk | None: ...


# pgvector 0.8.6 is already enabled on the target database; CREATE EXTENSION
# is harmless to repeat and only matters for a fresh database. The ALTER
# TABLE guards against schema drift: a `chunks` table created by an earlier
# iteration of this project (before start_offset existed) would otherwise
# make CREATE TABLE IF NOT EXISTS a no-op and every insert would then fail
# on a missing column.
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    accession TEXT NOT NULL,
    chunk_id  INTEGER NOT NULL,
    item      TEXT NOT NULL,
    start_offset INTEGER NOT NULL DEFAULT 0,
    text      TEXT NOT NULL,
    -- voyage-finance-2 (era.index.embeddings.MODEL) emits 1024-dimensional
    -- vectors; the column width has to match the embedder that fills it.
    embedding vector(1024) NOT NULL,
    PRIMARY KEY (accession, chunk_id)
);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_offset INTEGER NOT NULL DEFAULT 0;
"""


class PgVectorStore:
    """The real ChunkStore, backed by Postgres + pgvector.

    Every other component that talks to storage depends on the ChunkStore
    Protocol, not this class, so tests can run InMemoryChunkStore instead
    and the whole suite stays free of network and database calls.
    """

    def __init__(self, dsn: str) -> None:
        # Imported here, not at module level, so importing era.index.store
        # -- which the fake-only test suite does transitively, via
        # StoredChunk -- never requires psycopg or pgvector to be
        # installed. Only constructing a real PgVectorStore does.
        self._dsn = dsn
        self._conn = self._connect()

    def _connect(self) -> Any:
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=10)
        conn.execute(SCHEMA)
        register_vector(conn)
        return conn

    def _with_reconnect(self, operation: Callable[[Any], T]) -> T:
        """Run an operation, reconnecting once if the server had hung up.

        `psycopg` does not mark a connection closed until something actually
        fails on it, so checking a flag first is not enough -- the failure is
        how you find out. One retry: a genuinely unreachable database should
        surface, not be retried in a loop while a run appears to hang.
        """
        import psycopg

        try:
            return operation(self._live_connection())
        except psycopg.OperationalError:
            self._conn = self._connect()
            return operation(self._conn)

    def _live_connection(self) -> Any:
        """Reconnect if the server hung up while we were busy elsewhere.

        Neon's free tier suspends a compute after five minutes idle. Ingest
        holds this connection open while embedding a filing, which on Voyage's
        free tier takes twenty minutes or more for a large 10-K -- so by the
        time the chunks are ready to write, the connection is long dead and the
        write fails with AdminShutdown after all that work.

        Reconnecting here rather than pinging on a timer keeps the cost where
        it belongs: nothing happens on a healthy connection, and a suspended
        compute is woken by the query that actually needs it.
        """
        if self._conn.closed:
            self._conn = self._connect()
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        require_matching_dimensions(vectors)
        # psycopg's registered dumpers cover pgvector.Vector and
        # numpy.ndarray, not a plain list -- a bare list[float] falls
        # through to psycopg's generic ListDumper and is sent to Postgres
        # as float8[], which pgvector's `<=>` operator does not accept.
        # The assignment cast from array to vector happens to cover column
        # INSERT/UPDATE, so this would look fine here and only break at
        # query() -- wrap explicitly so both paths use the real vector type.
        from pgvector import Vector

        # Delete-then-insert, in one transaction, rather than an
        # INSERT ... ON CONFLICT DO UPDATE keyed by (accession, chunk_id).
        # A per-row upsert can only ever touch chunk_ids present in the new
        # set -- it can never remove a row whose chunk_id no longer exists
        # in a smaller re-index, so stale rows from a larger prior run
        # would survive and a citation could resolve to the wrong item's
        # text. autocommit is off for this block specifically so a crash
        # between the delete and the inserts can't leave the accession with
        # zero rows.
        accessions = {chunk.accession for chunk in chunks}

        def _write(conn: Any) -> None:
            with conn.transaction(), conn.cursor() as cur:
                for accession in accessions:
                    cur.execute("DELETE FROM chunks WHERE accession = %s", (accession,))
                for chunk, vector in zip(chunks, vectors, strict=True):
                    cur.execute(
                        """
                        INSERT INTO chunks
                            (accession, chunk_id, item, start_offset, text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chunk.accession,
                            chunk.chunk_id,
                            chunk.item,
                            chunk.start,
                            chunk.text,
                            Vector(vector),
                        ),
                    )

        self._with_reconnect(_write)

    def query(
        self,
        vector: list[float],
        item_filter: str | None,
        accession_filter: str | None,
        k: int,
    ) -> list[StoredChunk]:
        if k < 0:
            raise ValueError("k must not be negative")
        require_matching_dimensions([vector])
        from pgvector import Vector

        sql = "SELECT accession, chunk_id, item, start_offset, text FROM chunks"
        clauses: list[str] = []
        params: list[object] = []
        if item_filter is not None:
            clauses.append("item = %s")
            params.append(item_filter)
        if accession_filter is not None:
            clauses.append("accession = %s")
            params.append(accession_filter)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params.extend([Vector(vector), k])

        def _read(conn: Any) -> list[StoredChunk]:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [
                    StoredChunk(accession=r[0], chunk_id=r[1], item=r[2], start=r[3], text=r[4])
                    for r in cur.fetchall()
                ]

        return self._with_reconnect(_read)

    def get(self, accession: str, chunk_id: int) -> StoredChunk | None:
        def _read(conn: Any) -> StoredChunk | None:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT accession, chunk_id, item, start_offset, text FROM chunks "
                    "WHERE accession = %s AND chunk_id = %s",
                    (accession, chunk_id),
                )
                row = cur.fetchone()
            if row is None:
                return None
            return StoredChunk(
                accession=row[0], chunk_id=row[1], item=row[2], start=row[3], text=row[4]
            )

        return self._with_reconnect(_read)
