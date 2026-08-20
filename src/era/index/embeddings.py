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
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            result = self._client.embed(batch, model=MODEL, input_type="document")
            # voyageai types this as list[list[float]] | list[list[int]]
            # because an all-integer response would parse that way; coerce
            # so callers (and PgVectorStore's vector column) get a
            # consistent float vector.
            vectors.extend([float(x) for x in vector] for vector in result.embeddings)
        return vectors
