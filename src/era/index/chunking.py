from dataclasses import dataclass

DEFAULT_MAX_CHARS = 4000
DEFAULT_OVERLAP = 400


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    accession: str
    item: str
    text: str


def chunk_items(
    accession: str,
    items: dict[str, str],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split each item's text into overlapping passages for embedding.

    Each item is chunked independently, so a chunk can never span two
    items -- a passage straddling the end of one item and the start of the
    next would make it ambiguous which item a citation into it came from.
    chunk_id increments across the whole call, so it is a stable, unique
    address within this filing; store.get(accession, chunk_id) is how a
    later verification step resolves a citation back to its source text.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[Chunk] = []
    next_id = 0
    stride = max_chars - overlap

    # Iterate in the caller's own key order rather than sorting: for the
    # canonical item set ("1", "1A", "7"), ParsedFiling.items is built in
    # filing order (see era.edgar.sections), and a lexicographic sort would
    # actually misorder a wider set (e.g. "10" would sort before "1A").
    for item in items:
        for window in _windows(items[item], max_chars, stride):
            chunks.append(Chunk(chunk_id=next_id, accession=accession, item=item, text=window))
            next_id += 1

    return chunks


def _windows(text: str, max_chars: int, stride: int) -> list[str]:
    """Slide a max_chars-wide window across text, advancing by stride.

    Consecutive windows share `max_chars - stride` characters (the
    overlap) so a sentence that lands on a cut point still appears whole in
    at least one window, rather than being split across two and
    unretrievable from either.
    """
    if not text.strip():
        return []

    windows: list[str] = []
    start = 0
    while True:
        end = start + max_chars
        window = text[start:end]
        # A window can be blank even when the item as a whole is not -- a
        # long run of whitespace between paragraphs, landing exactly on a
        # stride boundary. Skip it rather than emit an empty citation
        # target, but keep advancing on `end`, not on whether we skipped,
        # so a blank window never changes when the scan reaches the end.
        # `.strip()` only decides whether to keep the window -- the text
        # itself is stored as the exact `text[start:end]` slice, whitespace
        # included, so a citation's offset always matches the source item.
        if window.strip():
            windows.append(window)
        if end >= len(text):
            break
        start += stride
    return windows
