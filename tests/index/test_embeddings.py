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
    """Stands in for voyageai.client.Client -- offline, no network, no key."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, batch: list[str], model: str, input_type: str) -> "_FakeEmbedResult":
        self.calls.append(list(batch))
        return _FakeEmbedResult([[0.0] * 3 for _ in batch])


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
