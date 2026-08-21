import pytest

from era.index.embeddings import (
    MAX_BATCH_SIZE,
    MAX_CHARS_PER_REQUEST,
    VoyageEmbedder,
    _batch_texts,
)


def test_batch_texts_splits_on_the_document_count_cap() -> None:
    texts = ["x"] * (MAX_BATCH_SIZE + 10)

    batches = _batch_texts(texts)

    assert len(batches[0]) == MAX_BATCH_SIZE
    assert sum(len(b) for b in batches) == len(texts)


def test_batch_texts_splits_on_the_character_budget_even_under_the_count_cap() -> None:
    # 128 documents at era.index.chunking's default 4000-character chunk
    # size is roughly 512,000 characters -- comfortably past
    # MAX_CHARS_PER_REQUEST. The count cap alone must not be trusted to
    # keep a request under Voyage's real per-request token cap.
    big_text = "x" * 4000
    texts = [big_text] * MAX_BATCH_SIZE

    batches = _batch_texts(texts)

    assert len(batches) > 1
    for batch in batches:
        assert sum(len(t) for t in batch) <= MAX_CHARS_PER_REQUEST


def test_batch_texts_never_drops_or_reorders_input() -> None:
    texts = [f"chunk-{i}" for i in range(300)]

    batches = _batch_texts(texts)

    assert [t for batch in batches for t in batch] == texts


def test_batch_texts_keeps_a_single_oversized_text_in_its_own_batch() -> None:
    # era.index.chunking never produces a chunk this large by default, but
    # _batch_texts can't assume that about every caller. A text alone over
    # the budget must not be silently dropped or truncated.
    oversized = "x" * (MAX_CHARS_PER_REQUEST + 1)

    batches = _batch_texts([oversized, "small"])

    assert batches == [[oversized], ["small"]]


def test_batch_texts_returns_nothing_for_no_input() -> None:
    assert _batch_texts([]) == []


class _RecordingClient:
    """Stands in for voyageai.client.Client -- offline, no network, no key.

    Returns a text-derived vector (the text's length, distinct per input)
    rather than a constant one. A constant vector (e.g. [[0.0] * 3 for _ in
    batch]) makes every result indistinguishable, so a bug that pairs the
    wrong vector with the wrong text -- e.g. iterating _batch_texts(texts) in
    reversed order -- would still make every test pass. With distinct
    per-text vectors, an ordering bug becomes a visible mismatch.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, batch: list[str], model: str, input_type: str) -> "_FakeEmbedResult":
        self.calls.append(list(batch))
        return _FakeEmbedResult([[float(len(text))] * 3 for text in batch])


class _FakeEmbedResult:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


def test_voyage_embedder_sends_one_request_per_batch_and_concatenates_results() -> None:
    embedder = VoyageEmbedder(api_key="voyage-fake")
    recording = _RecordingClient()
    embedder._client = recording  # type: ignore[assignment]
    texts = ["x" * 4000] * MAX_BATCH_SIZE

    vectors = embedder.embed(texts)

    assert len(recording.calls) > 1
    assert sum(len(call) for call in recording.calls) == len(texts)
    assert len(vectors) == len(texts)


def test_voyage_embedder_returns_vectors_in_input_order_across_multiple_batches() -> None:
    # Regression for a surviving mutant: `for batch in
    # reversed(_batch_texts(texts))` still passes count- and
    # batch-count-only assertions, because store.upsert's zip(strict=True)
    # only checks that the number of vectors matches the number of chunks --
    # it can't tell a vector landed on the wrong chunk. Distinct-length
    # texts, each producing a distinct vector, make a swapped pairing
    # visible: the returned vectors must match the *input* order, not
    # whatever order the batches happened to be sent in.
    embedder = VoyageEmbedder(api_key="voyage-fake")
    embedder._client = _RecordingClient()  # type: ignore[assignment]
    # Distinct lengths, spanning multiple batches (MAX_BATCH_SIZE=128).
    texts = [f"text-{i}-{'x' * i}" for i in range(150)]

    vectors = embedder.embed(texts)

    expected = [[float(len(text))] * 3 for text in texts]
    assert vectors == expected


class _ShortResponseClient:
    """Returns one fewer embedding than documents sent -- a malformed response."""

    def embed(self, batch: list[str], model: str, input_type: str) -> "_FakeEmbedResult":
        return _FakeEmbedResult([[0.0] * 3 for _ in batch[:-1]])


def test_voyage_embedder_raises_locally_on_a_short_response() -> None:
    # Without this check, a short response silently shifts every subsequent
    # chunk-vector pairing in the batch, then either surfaces far downstream
    # as store.upsert's zip(strict=True) ValueError -- pointing a debugger
    # at the store, not the embedder -- or, if strict zip happens not to
    # trigger, corrupts data silently. Catch it at the boundary that
    # actually caused it.
    embedder = VoyageEmbedder(api_key="voyage-fake")
    embedder._client = _ShortResponseClient()  # type: ignore[assignment]

    with pytest.raises(ValueError, match="embeddings"):
        embedder.embed(["a", "b", "c"])
