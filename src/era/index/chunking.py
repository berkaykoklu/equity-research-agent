from dataclasses import dataclass

DEFAULT_MAX_CHARS = 4000
DEFAULT_OVERLAP = 400


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    accession: str
    item: str
    start: int
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

    chunk_id increments across this call and is unique within it, but it is
    only a valid address for the run that produced it, not across runs: the
    boundary selector in era.edgar.sections is a model, so two parses of
    the same accession can legitimately produce different item sets or
    different body text for the same item, which shifts every chunk_id
    after the difference. store.get(accession, chunk_id) only resolves a
    citation correctly against the exact chunk set it was minted from.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    if overlap > max_chars // 2:
        # Task 9 embeds every chunk this function produces, on a project
        # with a hard cost budget. Stride shrinks toward zero as overlap
        # approaches max_chars, so an overlap close to max_chars turns one
        # item into thousands of near-duplicate chunks -- an unbounded
        # embedding cost reachable from a single config value. Capping
        # overlap at half of max_chars keeps stride, and so chunk count,
        # bounded by item length regardless of how overlap is configured.
        raise ValueError("overlap must not exceed half of max_chars")

    chunks: list[Chunk] = []
    next_id = 0
    stride = max_chars - overlap

    # Iterate in the caller's own key order rather than sorting: for the
    # canonical item set ("1", "1A", "7"), ParsedFiling.items is built in
    # filing order (see era.edgar.sections) and that order is part of its
    # contract. A lexicographic sort would actually misorder a wider set
    # (e.g. "10" would sort before "1A").
    for item in items:
        for start, window in _windows(items[item], max_chars, stride):
            chunks.append(
                Chunk(chunk_id=next_id, accession=accession, item=item, start=start, text=window)
            )
            next_id += 1

    return chunks


def _windows(text: str, max_chars: int, stride: int) -> list[tuple[int, str]]:
    """Slide a max_chars-wide window across text, advancing by stride.

    Consecutive windows share `max_chars - stride` characters (the
    overlap) so a sentence that lands on a cut point still appears whole in
    at least one window, rather than being split across two and
    unretrievable from either. Returns (start_offset, window_text) pairs;
    start_offset is recorded on the resulting Chunk so a caller can tell
    exactly where in the item's text a chunk came from.
    """
    if not text.strip():
        return []

    windows: list[tuple[int, str]] = []
    start = 0
    while True:
        end = start + max_chars
        window = text[start:end]
        # A window can be blank even when the item as a whole is not -- a
        # long run of whitespace between paragraphs, landing exactly on a
        # stride boundary. Skip it rather than emit an empty citation
        # target, but keep advancing on `end`, not on whether we skipped,
        # so a blank window never changes when the scan reaches the end.
        # This can leave a gap: characters inside a skipped blank window
        # are never stored in any chunk. That is accepted, not accidental
        # -- pure whitespace has no citable content -- but it does mean a
        # chunk's `start` cannot always be derived as index * stride.
        # `.strip()` only decides whether to keep the window -- the text
        # itself is stored as the exact `text[start:end]` slice, whitespace
        # included, so a citation's offset always matches the source item.
        if window.strip():
            windows.append((start, window))
        if end >= len(text):
            break
        start += stride
    return windows
