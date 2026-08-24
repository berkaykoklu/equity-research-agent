import pytest

from era.index.embeddings import (
    FREE_TIER_REQUESTS_PER_MINUTE,
    MAX_BATCH_SIZE,
    MAX_CHARS_PER_REQUEST,
    RATE_LIMIT_MAX_ATTEMPTS,
    VoyageEmbedder,
    _batch_texts,
)

# Tests that are about batching or ordering -- not pacing -- must not inherit
# the free-tier throttle, or they spend real minutes asleep. Rate limiting has
# its own tests further down, with an injected clock.
UNTHROTTLED = {"requests_per_minute": 10_000, "tokens_per_minute": 10_000_000}


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
    embedder = VoyageEmbedder(api_key="voyage-fake", **UNTHROTTLED)  # type: ignore[arg-type]
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
    embedder = VoyageEmbedder(api_key="voyage-fake", **UNTHROTTLED)  # type: ignore[arg-type]
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
    embedder = VoyageEmbedder(api_key="voyage-fake", **UNTHROTTLED)  # type: ignore[arg-type]
    embedder._client = _ShortResponseClient()  # type: ignore[assignment]

    with pytest.raises(ValueError, match="embeddings"):
        embedder.embed(["a", "b", "c"])


# --- free-tier rate limiting -----------------------------------------------
#
# Voyage throttles an account with no payment method to 3 requests and 10,000
# tokens per minute. The first real `era ingest` hit both and wrote nothing.
# Clock and sleep are injected so these prove the pacing without waiting.


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def _embedder(clock: _FakeClock, client: object, **kwargs: object) -> VoyageEmbedder:
    embedder = VoyageEmbedder(
        api_key="fake",
        sleep=clock.sleep,
        now=clock.now,
        **kwargs,  # type: ignore[arg-type]
    )
    embedder._client = client  # type: ignore[assignment]
    return embedder


def test_batches_are_sized_against_the_per_minute_token_budget() -> None:
    # A batch built only against the 120k per-request cap is rejected outright
    # on the free tier, where the minute budget is 10k.
    clock = _FakeClock()
    client = _RecordingClient()
    embedder = _embedder(clock, client)

    # Twelve chunks of 6,000 chars ~= 2,000 estimated tokens each.
    embedder.embed(["x" * 6_000] * 12)

    assert all(len(batch) <= 5 for batch in client.calls), client.calls


def test_requests_are_paced_under_the_per_minute_ceiling() -> None:
    clock = _FakeClock()
    client = _RecordingClient()
    embedder = _embedder(clock, client)

    # Enough data to need more requests than the 3/minute allowance.
    embedder.embed(["y" * 6_000] * 24)

    assert len(client.calls) > FREE_TIER_REQUESTS_PER_MINUTE
    assert clock.slept, "no pacing happened at all"


def test_a_generous_account_is_not_throttled() -> None:
    # Raising the limits is a constructor argument, not an edit.
    clock = _FakeClock()
    client = _RecordingClient()
    embedder = _embedder(clock, client, requests_per_minute=2_000, tokens_per_minute=1_000_000)

    embedder.embed(["z" * 6_000] * 12)

    assert clock.slept == []


def test_a_rate_limit_error_is_retried_then_surfaces() -> None:
    from voyageai.error import RateLimitError

    class _AlwaysLimited:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, batch: list[str], **_: object) -> object:
            self.calls += 1
            raise RateLimitError("slow down")

    clock = _FakeClock()
    client = _AlwaysLimited()
    embedder = _embedder(clock, client)

    with pytest.raises(RuntimeError, match="rate limit not cleared"):
        embedder.embed(["a" * 100])

    assert client.calls == RATE_LIMIT_MAX_ATTEMPTS
    assert clock.slept, "backoff never slept"


def test_the_limiter_paces_correctly_under_concurrent_callers() -> None:
    # The graph fans out five sections in parallel and each embeds its query.
    # Without a lock every thread reads the window, sees room, and fires --
    # straight through the ceiling. This asserts the ceiling holds.
    import threading

    from era.index.embeddings import _RateLimiter

    fired: list[float] = []
    clock = _FakeClock()
    lock = threading.Lock()
    limiter = _RateLimiter(
        requests_per_minute=3, tokens_per_minute=10_000, sleep=clock.sleep, now=clock.now
    )

    def worker() -> None:
        limiter.acquire(1_000)
        with lock:
            fired.append(clock.now())

    threads = [threading.Thread(target=worker) for _ in range(9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(fired) == 9
    # Nine requests against a 3-per-minute ceiling cannot all land in one
    # window; the clock must have been advanced by waiting.
    assert clock.t > 0.0, "nine requests fired without any pacing"
