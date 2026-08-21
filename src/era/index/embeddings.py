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


def _batch_texts(texts: list[str]) -> list[list[str]]:
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
        at_char_limit = bool(current) and current_chars + len(text) > MAX_CHARS_PER_REQUEST
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
    def __init__(self, api_key: str) -> None:
        # The installed client defaults to timeout=None, max_retries=0: a
        # hung request blocks ingest forever, and a single transient error
        # (rather than SEC's own retryable 429/503s, see era.edgar.client)
        # kills the whole run. 30s/3 retries gives it the same kind of
        # margin EdgarClient gives SEC requests.
        self._client = Client(api_key=api_key, timeout=30.0, max_retries=3)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batch_texts(texts):
            result = self._client.embed(batch, model=MODEL, input_type="document")
            # A short response (fewer embeddings than documents sent) must
            # be caught here, at the embedder boundary, not left to surface
            # later as store.upsert's zip(strict=True) ValueError -- which
            # would point a debugger at the store when the real fault is a
            # malformed Voyage response.
            if len(result.embeddings) != len(batch):
                raise ValueError(
                    f"voyage returned {len(result.embeddings)} embeddings "
                    f"for a batch of {len(batch)} documents"
                )
            # voyageai types this as list[list[float]] | list[list[int]]
            # because an all-integer response would parse that way; coerce
            # so callers (and PgVectorStore's vector column) get a
            # consistent float vector.
            vectors.extend([float(x) for x in vector] for vector in result.embeddings)
        return vectors
