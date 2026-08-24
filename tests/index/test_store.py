import pytest
from tests.fakes import FakeEmbedder, InMemoryChunkStore

from era.index.chunking import Chunk
from era.index.store import PgVectorStore, VectorDimensionError


def test_query_returns_nearest_chunk_first() -> None:
    store = InMemoryChunkStore()
    chunks = [
        Chunk(chunk_id=0, accession="acc", item="1", start=0, text="smartphones and wearables"),
        Chunk(chunk_id=1, accession="acc", item="1A", start=0, text="supply chain risk"),
    ]
    embedder = FakeEmbedder()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    hits = store.query(
        embedder.embed(["supply chain risk"])[0], item_filter=None, accession_filter=None, k=1
    )

    assert hits[0].chunk_id == 1


def test_query_can_restrict_to_one_item() -> None:
    store = InMemoryChunkStore()
    chunks = [
        Chunk(chunk_id=0, accession="acc", item="1", start=0, text="supply chain risk"),
        Chunk(chunk_id=1, accession="acc", item="1A", start=0, text="supply chain risk"),
    ]
    embedder = FakeEmbedder()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    hits = store.query(
        embedder.embed(["supply chain risk"])[0], item_filter="1A", accession_filter=None, k=5
    )

    assert {h.item for h in hits} == {"1A"}


def test_query_can_restrict_to_one_accession() -> None:
    # The store holds chunks from many filings at once. Without this
    # filter, a query issued while writing a note about one company could
    # retrieve -- and a citation could then point to -- a passage from a
    # different company's filing.
    store = InMemoryChunkStore()
    embedder = FakeEmbedder()
    chunks = [
        Chunk(chunk_id=0, accession="acc-a", item="1", start=0, text="supply chain risk"),
        Chunk(chunk_id=0, accession="acc-b", item="1", start=0, text="supply chain risk"),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    hits = store.query(
        embedder.embed(["supply chain risk"])[0],
        item_filter=None,
        accession_filter="acc-b",
        k=5,
    )

    assert {h.accession for h in hits} == {"acc-b"}


def test_query_respects_k() -> None:
    store = InMemoryChunkStore()
    embedder = FakeEmbedder()
    chunks = [
        Chunk(chunk_id=i, accession="acc", item="1", start=0, text="supply chain risk")
        for i in range(5)
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    hits = store.query(
        embedder.embed(["supply chain risk"])[0], item_filter=None, accession_filter=None, k=2
    )

    assert len(hits) == 2


def test_query_rejects_negative_k() -> None:
    store = InMemoryChunkStore()
    vector = FakeEmbedder().embed(["x"])[0]

    with pytest.raises(ValueError):
        store.query(vector, item_filter=None, accession_filter=None, k=-1)


def test_get_resolves_a_citation_target() -> None:
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=7, accession="acc", item="1", start=0, text="hello")
    store.upsert([chunk], FakeEmbedder().embed(["hello"]))

    assert store.get("acc", 7) is not None
    assert store.get("acc", 999) is None


def test_get_resolves_the_start_offset() -> None:
    # start is the offset into the item's text this chunk came from; a
    # citation resolver needs it to point back at the right passage, not
    # just the right chunk_id.
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=0, accession="acc", item="1", start=42, text="hello")
    store.upsert([chunk], FakeEmbedder().embed(["hello"]))

    fetched = store.get("acc", 0)
    assert fetched is not None
    assert fetched.start == 42


def test_upsert_is_idempotent() -> None:
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=0, accession="acc", item="1", start=0, text="hello")
    vectors = FakeEmbedder().embed(["hello"])

    store.upsert([chunk], vectors)
    store.upsert([chunk], vectors)

    assert len(store.query(vectors[0], item_filter=None, accession_filter=None, k=10)) == 1


def test_upsert_deletes_stale_rows_from_a_smaller_reindex() -> None:
    # chunk_id is a per-run counter (see era.index.chunking). A re-parse of
    # the same accession can legitimately produce fewer chunks than the
    # last run. If upsert only overwrote rows by (accession, chunk_id) and
    # never deleted, the surplus rows from the earlier, larger run would
    # survive -- and a citation minted against the new run could resolve
    # chunk_id 5 to stale text from an item that no longer has that many
    # chunks. upsert must delete every existing row for the accession
    # before writing the new set.
    store = InMemoryChunkStore()
    embedder = FakeEmbedder()

    first_run = [
        Chunk(chunk_id=i, accession="acc", item="1", start=i, text=f"chunk {i}") for i in range(5)
    ]
    store.upsert(first_run, embedder.embed([c.text for c in first_run]))

    second_run = [
        Chunk(chunk_id=0, accession="acc", item="1", start=0, text="only chunk"),
    ]
    store.upsert(second_run, embedder.embed([c.text for c in second_run]))

    # A real (non-zero) query vector, not the "return everything" zero
    # idiom: pgvector yields NaN distances for a zero vector, so a test
    # relying on that would pass here and fail against the real store.
    query_vector = embedder.embed(["only chunk"])[0]
    all_hits = store.query(query_vector, item_filter=None, accession_filter=None, k=100)
    assert len(all_hits) == 1
    assert all_hits[0].text == "only chunk"
    assert store.get("acc", 4) is None


def test_upsert_does_not_delete_a_different_accessions_rows() -> None:
    store = InMemoryChunkStore()
    embedder = FakeEmbedder()

    other = Chunk(chunk_id=0, accession="other-acc", item="1", start=0, text="unrelated")
    store.upsert([other], embedder.embed(["unrelated"]))

    mine = Chunk(chunk_id=0, accession="acc", item="1", start=0, text="mine")
    store.upsert([mine], embedder.embed(["mine"]))
    store.upsert([mine], embedder.embed(["mine"]))

    assert store.get("other-acc", 0) is not None


def test_upsert_scopes_the_delete_to_every_accession_in_the_batch() -> None:
    # A single upsert call can carry chunks for more than one accession --
    # e.g. re-indexing several filings together. A delete scoped only to
    # the first chunk's accession would leave every other accession's
    # stale rows in place instead of replacing them.
    store = InMemoryChunkStore()
    embedder = FakeEmbedder()

    old_a = [Chunk(chunk_id=i, accession="A", item="1", start=i, text=f"a{i}") for i in range(3)]
    old_b = [Chunk(chunk_id=i, accession="B", item="1", start=i, text=f"b{i}") for i in range(3)]
    store.upsert(old_a, embedder.embed([c.text for c in old_a]))
    store.upsert(old_b, embedder.embed([c.text for c in old_b]))

    new_batch = [
        Chunk(chunk_id=0, accession="A", item="1", start=0, text="new a"),
        Chunk(chunk_id=0, accession="B", item="1", start=0, text="new b"),
    ]
    store.upsert(new_batch, embedder.embed([c.text for c in new_batch]))

    assert store.get("A", 1) is None
    assert store.get("A", 2) is None
    assert store.get("B", 1) is None
    assert store.get("B", 2) is None
    assert store.get("A", 0) is not None
    assert store.get("B", 0) is not None


def test_upsert_raises_on_a_chunk_vector_length_mismatch() -> None:
    # A dropped strict=True would silently pair the wrong vector with the
    # wrong chunk instead of failing -- exactly the kind of corruption this
    # store exists to prevent.
    store = InMemoryChunkStore()
    chunks = [
        Chunk(chunk_id=0, accession="acc", item="1", start=0, text="a"),
        Chunk(chunk_id=1, accession="acc", item="1", start=0, text="b"),
    ]
    vectors = FakeEmbedder().embed(["only one"])

    with pytest.raises(ValueError):
        store.upsert(chunks, vectors)


def test_upsert_raises_on_a_chunk_vector_length_mismatch_before_deleting_anything() -> None:
    store = InMemoryChunkStore()
    embedder = FakeEmbedder()
    existing = Chunk(chunk_id=0, accession="acc", item="1", start=0, text="hello")
    store.upsert([existing], embedder.embed(["hello"]))

    bad_chunks = [
        Chunk(chunk_id=0, accession="acc", item="1", start=0, text="a"),
        Chunk(chunk_id=1, accession="acc", item="1", start=0, text="b"),
    ]
    with pytest.raises(ValueError):
        store.upsert(bad_chunks, embedder.embed(["only one"]))

    assert store.get("acc", 0) is not None


def test_upsert_rejects_a_wrong_dimension_vector() -> None:
    store = InMemoryChunkStore()
    chunk = Chunk(chunk_id=0, accession="acc", item="1", start=0, text="hello")

    with pytest.raises(VectorDimensionError):
        store.upsert([chunk], [[0.0, 1.0, 2.0]])


def test_query_rejects_a_wrong_dimension_vector() -> None:
    store = InMemoryChunkStore()

    with pytest.raises(VectorDimensionError):
        store.query([0.0, 1.0], item_filter=None, accession_filter=None, k=1)


# --- reconnect after the server hangs up -----------------------------------
#
# Neon's free tier suspends a compute after five minutes idle. Ingest holds a
# connection open while embedding, which on Voyage's free tier takes twenty
# minutes or more for a large 10-K -- so the write arrives at a dead
# connection and fails with AdminShutdown after all that work. Found on a real
# ingest of KO, which embedded the whole filing and then lost every chunk.


class _DeadThenLiveConnection:
    """Fails once the way a suspended Postgres does, then works."""

    def __init__(self, fail_times: int) -> None:
        self.remaining_failures = fail_times
        self.closed = False
        self.executed: list[str] = []

    def execute(self, sql: str, *args: object) -> None:
        self.executed.append(sql)

    def cursor(self) -> "_DeadThenLiveConnection":
        return self

    def __enter__(self) -> "_DeadThenLiveConnection":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def fetchone(self) -> None:
        return None


def test_a_dropped_connection_is_reconnected_rather_than_failing_the_write() -> None:
    import psycopg

    store = PgVectorStore.__new__(PgVectorStore)  # bypass connecting for real
    store._dsn = "postgresql://unused"
    attempts: list[int] = []

    live = _DeadThenLiveConnection(fail_times=0)
    store._conn = live
    store._connect = lambda: live  # type: ignore[method-assign]

    def operation(conn: object) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise psycopg.OperationalError("terminating connection due to administrator command")
        return "written"

    assert store._with_reconnect(operation) == "written"
    assert len(attempts) == 2, "should have retried exactly once"


def test_a_genuinely_unreachable_database_surfaces_rather_than_looping() -> None:
    import psycopg

    store = PgVectorStore.__new__(PgVectorStore)
    store._dsn = "postgresql://unused"
    store._conn = _DeadThenLiveConnection(fail_times=0)
    store._connect = lambda: _DeadThenLiveConnection(fail_times=0)  # type: ignore[method-assign]

    def always_fails(conn: object) -> str:
        raise psycopg.OperationalError("connection refused")

    with pytest.raises(psycopg.OperationalError):
        store._with_reconnect(always_fails)
