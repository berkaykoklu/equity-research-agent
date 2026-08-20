"""Offline stand-ins for every external-dependency seam in this project.

Same pattern throughout: the real implementation sits behind a Protocol
(BoundarySelector, Embedder, ChunkStore) so tests can swap in something
deterministic and free instead of calling a model, an embedding API, or a
database.
"""

import math

from era.edgar.boundaries import Boundary, HeadingCandidate
from era.index.chunking import Chunk
from era.index.store import StoredChunk


class FirstMatchSelector:
    """Picks the first candidate per item and runs to the next candidate.

    Deliberately naive -- it reproduces the pre-Task-19 regex behaviour
    (first occurrence wins, no judgment about what kind of line it is), so
    tests can assert what the real selector must do better, and can assert
    that era.edgar.sections' verification step catches this selector's
    mistakes rather than trusting them.
    """

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        chosen: dict[str, Boundary] = {}
        for candidate in candidates:
            if candidate.item in chosen:
                continue
            following = next((c.index for c in candidates if c.index > candidate.index), None)
            chosen[candidate.item] = Boundary(candidate.index, following)
        return chosen


class ScriptedSelector:
    """Returns boundaries a test dictates, so slicing and guards can be tested alone."""

    def __init__(self, boundaries: dict[str, Boundary]) -> None:
        self._boundaries = boundaries

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        return self._boundaries


_VOCAB = ["smartphones", "wearables", "supply", "chain", "risk", "revenue", "services"]


class FakeEmbedder:
    """Bag-of-words vectors -- deterministic and dependency-free.

    Not a semantic embedding; it only exists so query/store tests can
    assert nearest-neighbour behaviour on inputs that share or don't share
    vocabulary, without ever calling the real Voyage API.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append([float(lowered.count(word)) for word in _VOCAB])
        return vectors


class InMemoryChunkStore:
    """Faithful stand-in for PgVectorStore, including its delete-before-write upsert.

    A fake that behaves differently from the real store is worse than no
    fake: it would let a test pass here while the same code silently
    corrupts data against the real database. In particular, upsert deletes
    every existing row for each accession being written before inserting
    the new set -- see era.index.store.ChunkStore.upsert for why chunk_id
    is not a stable identity across chunking runs.
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[str, int], tuple[StoredChunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        accessions = {chunk.accession for chunk in chunks}
        for key in [key for key in self._rows if key[0] in accessions]:
            del self._rows[key]
        for chunk, vector in zip(chunks, vectors, strict=True):
            stored = StoredChunk(
                chunk_id=chunk.chunk_id,
                accession=chunk.accession,
                item=chunk.item,
                start=chunk.start,
                text=chunk.text,
            )
            self._rows[(chunk.accession, chunk.chunk_id)] = (stored, vector)

    def query(self, vector: list[float], item_filter: str | None, k: int) -> list[StoredChunk]:
        scored: list[tuple[float, StoredChunk]] = []
        for stored, stored_vector in self._rows.values():
            if item_filter is not None and stored.item != item_filter:
                continue
            scored.append((_cosine(vector, stored_vector), stored))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [stored for _, stored in scored[:k]]

    def get(self, accession: str, chunk_id: int) -> StoredChunk | None:
        row = self._rows.get((accession, chunk_id))
        return row[0] if row else None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
