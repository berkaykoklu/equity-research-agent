from typing import Protocol

# voyageai's package __init__ resolves some attributes through a module-level
# __getattr__, which defeats mypy's static export check on `voyageai.Client`.
# Importing straight from the submodule sidesteps that without an ignore comment.
from voyageai.client import Client

MODEL = "voyage-finance-2"
# voyage-finance-2's native output width. era.index.store.SCHEMA declares
# the embedding column as vector(1024) to match -- changing this without
# migrating the column would make every insert fail dimension checks, or
# worse, silently truncate.
DIMENSIONS = 1024


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
        self._client = Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=MODEL, input_type="document")
        # voyageai types this as list[list[float]] | list[list[int]] because
        # an all-integer response would parse that way; coerce so callers
        # (and PgVectorStore's vector column) get a consistent float vector.
        return [[float(x) for x in vector] for vector in result.embeddings]
