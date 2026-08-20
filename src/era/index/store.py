from dataclasses import dataclass
from typing import Protocol

from era.index.chunking import Chunk


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
        re-index, and a citation verifier would silently pass it.
        """
        ...

    def query(self, vector: list[float], item_filter: str | None, k: int) -> list[StoredChunk]: ...

    def get(self, accession: str, chunk_id: int) -> StoredChunk | None: ...


# pgvector 0.8.6 is already enabled on the target database; this statement
# is harmless to repeat and only matters for a fresh database.
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    accession TEXT NOT NULL,
    chunk_id  INTEGER NOT NULL,
    item      TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    text      TEXT NOT NULL,
    -- voyage-finance-2 (era.index.embeddings.MODEL) emits 1024-dimensional
    -- vectors; the column width has to match the embedder that fills it.
    embedding vector(1024) NOT NULL,
    PRIMARY KEY (accession, chunk_id)
);
"""


class PgVectorStore:
    """The real ChunkStore, backed by Postgres + pgvector.

    Every other component that talks to storage depends on the ChunkStore
    Protocol, not this class, so tests can run InMemoryChunkStore instead
    and the whole suite stays free of network and database calls.
    """

    def __init__(self, dsn: str) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute(SCHEMA)
        register_vector(self._conn)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
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
        with self._conn.transaction(), self._conn.cursor() as cur:
            for accession in accessions:
                cur.execute("DELETE FROM chunks WHERE accession = %s", (accession,))
            for chunk, vector in zip(chunks, vectors, strict=True):
                cur.execute(
                    """
                    INSERT INTO chunks (accession, chunk_id, item, start_offset, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (chunk.accession, chunk.chunk_id, chunk.item, chunk.start, chunk.text, vector),
                )

    def query(self, vector: list[float], item_filter: str | None, k: int) -> list[StoredChunk]:
        sql = "SELECT accession, chunk_id, item, start_offset, text FROM chunks"
        params: list[object] = []
        if item_filter is not None:
            sql += " WHERE item = %s"
            params.append(item_filter)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params.extend([vector, k])
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [
                StoredChunk(accession=r[0], chunk_id=r[1], item=r[2], start=r[3], text=r[4])
                for r in cur.fetchall()
            ]

    def get(self, accession: str, chunk_id: int) -> StoredChunk | None:
        with self._conn.cursor() as cur:
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
