import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

# voyageai's package __init__ resolves some attributes through a module-level
# __getattr__, which defeats mypy's static export check on `voyageai.Client`.
# Importing straight from the submodule sidesteps that without an ignore comment.
from voyageai.client import Client

MODEL = "voyage-finance-2"
# voyage-finance-2's native output width. era.index.store.SCHEMA declares
# the embedding column as vector(1024) to match -- changing this without
# migrating the column would make every insert fail dimension checks, or
# worse, silently truncate. era.index.store imports this constant and
# rejects any vector whose length disagrees with it, in both PgVectorStore
# and the fake, so a mismatch fails loudly at the store boundary rather
# than surfacing as an opaque psycopg error mid-transaction.
DIMENSIONS = 1024

# Voyage rejects a single embed request carrying more than 128 documents.
# A 10-K's chunks routinely number in the hundreds, so VoyageEmbedder.embed
# batches internally -- every other caller in this project just calls
# embed(texts) once per set of chunks and shouldn't have to know Voyage's
# per-request cap to stay under it.
MAX_BATCH_SIZE = 128

# voyage-finance-2 also caps a single request at 120,000 total tokens
# (confirmed against Voyage's current docs, docs.voyageai.com/docs/embeddings,
# checked 2026-08-20 -- grouped with voyage-3-large, voyage-code-3,
# voyage-law-2 and others under the same 120K figure; not a number to guess
# at, since it changes per model and Voyage's docs are the only source of
# truth for it). MAX_BATCH_SIZE alone doesn't protect against this: a batch
# of 128 chunks at era.index.chunking's default 4000-character chunk size is
# roughly 512,000 characters, well past 120K tokens under any reasonable
# estimate. `era ingest` is the first caller in this project to ever send
# full-size batches, so under-batching here would fail partway through a
# real, money-spending run rather than in a test.
#
# There is no exact tokenizer available client-side, so this estimates
# conservatively: 3 characters per token. English prose is often closer to
# 4, but a 10-K is denser with numbers, ticker-style tokens and legal
# boilerplate, which tend to tokenize less efficiently than prose -- 3 errs
# toward overestimating the token count, i.e. toward smaller, safer batches.
# The byte budget itself also targets a fraction of the real 120,000-token
# cap, not the cap itself, so both the estimate and the margin favor
# under-filling a request over risking Voyage's InvalidRequestError.
_CHARS_PER_TOKEN_ESTIMATE = 3
_TOKEN_BUDGET_PER_REQUEST = 100_000
MAX_CHARS_PER_REQUEST = _TOKEN_BUDGET_PER_REQUEST * _CHARS_PER_TOKEN_ESTIMATE


# Voyage throttles an account with no payment method on file to 3 requests and
# 10,000 tokens per minute. Both ceilings bite: the token one is far below the
# 120,000-token per-request cap above, so a batch sized only against that cap
# is rejected outright. Discovered on the first real `era ingest`, which failed
# with RateLimitError before writing a single chunk.
#
# These are defaults, not constants: an account with a payment method (the free
# token allowance still applies) gets standard limits, and raising them is a
# constructor argument rather than an edit.
FREE_TIER_REQUESTS_PER_MINUTE = 3
FREE_TIER_TOKENS_PER_MINUTE = 10_000

RATE_LIMIT_WINDOW_SECONDS = 60.0
# Retries for a 429 that slips through anyway -- the limiter tracks what *this*
# process sent, and another run against the same key would not be visible to it.
RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_BACKOFF_SECONDS = 20.0


def _estimated_tokens(texts: list[str]) -> int:
    return sum(len(text) for text in texts) // _CHARS_PER_TOKEN_ESTIMATE + 1


class _RateLimiter:
    """Holds requests under a per-minute request *and* token ceiling.

    A sliding window rather than a fixed one: Voyage measures the last 60
    seconds continuously, so resetting a counter on the minute would let a
    burst straddle the boundary and still get rejected.

    `sleep` and `now` are injected so tests can prove the pacing without
    actually waiting a minute.
    """

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._requests_per_minute = requests_per_minute
        self._tokens_per_minute = tokens_per_minute
        self._sleep = sleep
        self._now = now
        self._window: deque[tuple[float, int]] = deque()
        # The graph fans out five sections in parallel and each embeds its own
        # query, so five threads reach this at once. Without the lock they all
        # read the window, all see room, and all fire -- straight through a
        # 3-per-minute ceiling. Observed as "rate limit not cleared after 5
        # attempts" on the first multi-ticker eval run. EdgarClient's throttle
        # needed exactly this fix for exactly this reason.
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        while self._window and now - self._window[0][0] >= RATE_LIMIT_WINDOW_SECONDS:
            self._window.popleft()

    def acquire(self, tokens: int) -> None:
        # Held across the sleep on purpose. Releasing it to wait would let the
        # other threads re-check, find the same lack of room, and pile up --
        # serialising the wait is what keeps the pacing honest under fan-out.
        with self._lock:
            self._acquire_locked(tokens)

    def _acquire_locked(self, tokens: int) -> None:
        while True:
            now = self._now()
            self._evict(now)
            spent = sum(count for _, count in self._window)
            room = (
                len(self._window) < self._requests_per_minute
                and spent + tokens <= self._tokens_per_minute
            )
            # A single request larger than the whole per-minute budget can
            # never fit alongside anything else. Sending it alone is the only
            # way to make progress; Voyage may still reject it, which surfaces
            # as a real error rather than an infinite wait here.
            if room or not self._window:
                self._window.append((now, tokens))
                return
            oldest = self._window[0][0]
            self._sleep(max(oldest + RATE_LIMIT_WINDOW_SECONDS - now, 0.05))


def _batch_texts(texts: list[str], max_chars: int = MAX_CHARS_PER_REQUEST) -> list[list[str]]:
    """Group texts into requests honoring both the count cap and the
    estimated-token budget above, without ever splitting a caller's list
    out of order.

    A single text longer than the whole budget (never produced by
    era.index.chunking's default settings, but not something this function
    can rule out for an arbitrary caller) still gets its own batch rather
    than being dropped or truncated -- if Voyage rejects it, that surfaces
    as a real API error, not a silent content loss.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        at_count_limit = len(current) >= MAX_BATCH_SIZE
        at_char_limit = bool(current) and current_chars + len(text) > max_chars
        if at_count_limit or at_char_limit:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += len(text)
    if current:
        batches.append(current)
    return batches


class Embedder(Protocol):
    """The one door onto an embedding API.

    Everything downstream (chunk indexing, query-time retrieval) depends on
    this Protocol, never on VoyageEmbedder directly, so tests can swap in
    tests.fakes.FakeEmbedder and the suite never spends the project's $7
    OpenAI budget or Voyage's free-tier quota.
    """

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str,
        requests_per_minute: int = FREE_TIER_REQUESTS_PER_MINUTE,
        tokens_per_minute: int = FREE_TIER_TOKENS_PER_MINUTE,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limiter = _RateLimiter(requests_per_minute, tokens_per_minute, sleep, now)
        self._sleep = sleep
        # Batches are sized against the per-minute token budget, not just the
        # per-request cap: on the free tier the minute budget is the tighter of
        # the two by more than an order of magnitude, and a batch built for the
        # request cap is rejected before the limiter's pacing ever helps.
        self._max_chars = min(MAX_CHARS_PER_REQUEST, tokens_per_minute * _CHARS_PER_TOKEN_ESTIMATE)
        # The installed client defaults to timeout=None, max_retries=0: a
        # hung request blocks ingest forever, and a single transient error
        # (rather than SEC's own retryable 429/503s, see era.edgar.client)
        # kills the whole run. 30s/3 retries gives it the same kind of
        # margin EdgarClient gives SEC requests.
        self._client = Client(api_key=api_key, timeout=30.0, max_retries=3)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """One request, paced by the limiter and retried on a 429.

        The limiter only knows what this process sent. A concurrent run against
        the same key, or Voyage counting differently at the edge, can still
        produce a rate-limit error -- so the retry exists as well as the pacing,
        not instead of it.
        """
        from voyageai.error import RateLimitError

        last_error: Exception | None = None
        for attempt in range(RATE_LIMIT_MAX_ATTEMPTS):
            self._limiter.acquire(_estimated_tokens(batch))
            try:
                result = self._client.embed(batch, model=MODEL, input_type="document")
            except RateLimitError as exc:
                last_error = exc
                self._sleep(RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1))
                continue
            # voyageai types this as list[list[float]] | list[list[int]]
            # because an all-integer response parses that way; the caller
            # coerces to float, so hand back the raw rows unchanged.
            rows: list[list[float]] = [list(row) for row in result.embeddings]
            return rows
        raise RuntimeError(
            f"voyage rate limit not cleared after {RATE_LIMIT_MAX_ATTEMPTS} attempts"
        ) from last_error

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batch_texts(texts, self._max_chars):
            embeddings = self._embed_batch(batch)
            # A short response (fewer embeddings than documents sent) must
            # be caught here, at the embedder boundary, not left to surface
            # later as store.upsert's zip(strict=True) ValueError -- which
            # would point a debugger at the store when the real fault is a
            # malformed Voyage response.
            if len(embeddings) != len(batch):
                raise ValueError(
                    f"voyage returned {len(embeddings)} embeddings "
                    f"for a batch of {len(batch)} documents"
                )
            # voyageai types this as list[list[float]] | list[list[int]]
            # because an all-integer response would parse that way; coerce
            # so callers (and PgVectorStore's vector column) get a
            # consistent float vector.
            vectors.extend([float(x) for x in vector] for vector in embeddings)
        return vectors
