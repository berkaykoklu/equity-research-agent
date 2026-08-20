import pytest

from era.index.chunking import chunk_items


def test_chunk_ids_are_unique_and_stable() -> None:
    items = {"1": "alpha " * 500, "1A": "beta " * 500}

    chunks = chunk_items("0000320193-24-000123", items, max_chars=600, overlap=50)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_each_chunk_records_the_item_it_came_from() -> None:
    chunks = chunk_items("0000320193-24-000123", {"1A": "risk " * 400}, max_chars=600, overlap=50)

    assert {c.item for c in chunks} == {"1A"}
    assert all(c.accession == "0000320193-24-000123" for c in chunks)


def test_chunks_never_span_two_items() -> None:
    chunks = chunk_items(
        "acc", {"1": "alpha " * 200, "1A": "beta " * 200}, max_chars=10_000, overlap=0
    )

    for chunk in chunks:
        assert not ("alpha" in chunk.text and "beta" in chunk.text)


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_items("acc", {"1": "x" * 1000}, max_chars=400, overlap=100)

    assert len(chunks) > 1
    assert len(chunks[0].text) == 400


def test_consecutive_chunks_share_overlapping_text() -> None:
    # The overlap is what stops a sentence that sits on a cut boundary from
    # being unretrievable from either chunk. Assert the actual shared text,
    # not just chunk count, so a bug that changes the stride without
    # changing the count would still be caught.
    chunks = chunk_items("acc", {"1": "x" * 1000}, max_chars=400, overlap=100)

    first, second = chunks[0], chunks[1]
    assert first.text[-100:] == second.text[:100]


def test_empty_item_body_produces_no_chunks() -> None:
    chunks = chunk_items("acc", {"1": "", "1A": "risk " * 400}, max_chars=600, overlap=50)

    assert all(c.item != "1" for c in chunks)
    assert any(c.item == "1A" for c in chunks)


def test_whitespace_only_item_body_produces_no_chunks() -> None:
    chunks = chunk_items("acc", {"1": "   \n\t  "}, max_chars=600, overlap=50)

    assert chunks == []


def test_item_shorter_than_chunk_size_becomes_a_single_chunk() -> None:
    chunks = chunk_items("acc", {"1": "short body"}, max_chars=4000, overlap=400)

    assert len(chunks) == 1
    assert chunks[0].text == "short body"


def test_item_exactly_at_boundary_becomes_a_single_chunk() -> None:
    text = "y" * 600

    chunks = chunk_items("acc", {"1": text}, max_chars=600, overlap=50)

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_item_one_char_over_boundary_produces_two_chunks() -> None:
    text = "y" * 601

    chunks = chunk_items("acc", {"1": text}, max_chars=600, overlap=50)

    assert len(chunks) == 2


def test_overlap_equal_to_max_chars_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_items("acc", {"1": "text"}, max_chars=600, overlap=600)


def test_overlap_larger_than_max_chars_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_items("acc", {"1": "text"}, max_chars=600, overlap=700)


def test_negative_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_items("acc", {"1": "text"}, max_chars=600, overlap=-1)


def test_non_positive_max_chars_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_items("acc", {"1": "text"}, max_chars=0, overlap=0)

    with pytest.raises(ValueError, match="max_chars"):
        chunk_items("acc", {"1": "text"}, max_chars=-1, overlap=0)


def test_chunk_ids_restart_from_zero_per_call() -> None:
    # chunk_id is only an address within a single filing (accession is the
    # other half of the key store.get uses), so two independent calls for
    # two different filings both starting at 0 is correct, not a collision.
    first_call = chunk_items("acc-1", {"1": "alpha " * 400}, max_chars=600, overlap=50)
    second_call = chunk_items("acc-2", {"1": "alpha " * 400}, max_chars=600, overlap=50)

    assert first_call[0].chunk_id == second_call[0].chunk_id == 0


def test_no_content_is_lost_between_chunks() -> None:
    # Each chunk must be the exact slice its position implies (start = i *
    # stride), and the final chunk must reach the end of the text --
    # otherwise content vanished at a seam instead of merely being
    # duplicated across it.
    text = "0123456789" * 130  # 1300 chars, not a multiple of the stride
    max_chars, overlap = 400, 100
    stride = max_chars - overlap

    chunks = chunk_items("acc", {"1": text}, max_chars=max_chars, overlap=overlap)

    covered_end = 0
    for i, chunk in enumerate(chunks):
        start = i * stride
        assert text[start : start + len(chunk.text)] == chunk.text
        covered_end = max(covered_end, start + len(chunk.text))
    assert covered_end == len(text)
