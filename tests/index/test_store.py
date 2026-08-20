from tests.fakes import FakeEmbedder, InMemoryChunkStore

from era.index.chunking import Chunk


def test_query_returns_nearest_chunk_first() -> None:
    store = InMemoryChunkStore()
    chunks = [
        Chunk(chunk_id=0, accession="acc", item="1", start=0, text="smartphones and wearables"),
        Chunk(chunk_id=1, accession="acc", item="1A", start=0, text="supply chain risk"),
    ]
    embedder = FakeEmbedder()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    hits = store.query(embedder.embed(["supply chain risk"])[0], item_filter=None, k=1)

    assert hits[0].chunk_id == 1


def test_query_can_restrict_to_one_item() -> None:
    store = InMemoryChunkStore()
    chunks = [
        Chunk(chunk_id=0, accession="acc", item="1", start=0, text="supply chain risk"),
        Chunk(chunk_id=1, accession="acc", item="1A", start=0, text="supply chain risk"),
    ]
    embedder = FakeEmbedder()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    hits = store.query(embedder.embed(["supply chain risk"])[0], item_filter="1A", k=5)

    assert {h.item for h in hits} == {"1A"}


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

    assert len(store.query(vectors[0], item_filter=None, k=10)) == 1


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

    all_hits = store.query([0.0] * len(embedder.embed(["x"])[0]), item_filter=None, k=100)
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
