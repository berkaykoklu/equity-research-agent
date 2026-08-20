import os
from collections.abc import Iterator

import pytest

from era.index.chunking import Chunk
from era.index.store import PgVectorStore

DSN = os.environ.get("DATABASE_URL")
ACCESSION = "integration-test-fixture"

pytestmark = pytest.mark.skipif(
    not DSN, reason="DATABASE_URL not set — skipping live pgvector integration test"
)


@pytest.fixture
def cleanup() -> Iterator[None]:
    yield
    import psycopg

    assert DSN is not None
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DELETE FROM chunks WHERE accession = %s", (ACCESSION,))


def test_round_trips_a_chunk_through_real_pgvector(cleanup: None) -> None:
    assert DSN is not None
    store = PgVectorStore(dsn=DSN)
    chunk = Chunk(chunk_id=1, accession=ACCESSION, item="1A", start=0, text="hello pgvector")
    vector = [0.0] * 1024
    vector[0] = 1.0

    store.upsert([chunk], [vector])

    fetched = store.get(ACCESSION, 1)
    assert fetched is not None
    assert fetched.text == "hello pgvector"
    assert fetched.start == 0

    hits = store.query(vector, item_filter="1A", k=1)
    assert hits[0].chunk_id == 1


def test_upsert_deletes_stale_rows_against_real_pgvector(cleanup: None) -> None:
    assert DSN is not None
    store = PgVectorStore(dsn=DSN)
    vector = [0.0] * 1024
    vector[0] = 1.0

    first_run = [
        Chunk(chunk_id=i, accession=ACCESSION, item="1", start=i, text=f"chunk {i}")
        for i in range(3)
    ]
    store.upsert(first_run, [vector] * len(first_run))

    second_run = [Chunk(chunk_id=0, accession=ACCESSION, item="1", start=0, text="only chunk")]
    store.upsert(second_run, [vector])

    assert store.get(ACCESSION, 2) is None
    fetched = store.get(ACCESSION, 0)
    assert fetched is not None
    assert fetched.text == "only chunk"
