import re
from dataclasses import dataclass

WANTED_ITEMS = ("1", "1A", "7")

_TAGS = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_HEADING = re.compile(r"item\s+(?P<item>\d{1,2}[A-Z]?)\s*[.:\-–]", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ParsedFiling:
    items: dict[str, str]
    missing: tuple[str, ...]


def _to_text(html: str) -> str:
    without_tags = _TAGS.sub(" ", html)
    return _WHITESPACE.sub(" ", without_tags).strip()


def parse_items(html: str) -> ParsedFiling:
    """Split a filing into the items we care about.

    Every 10-K lists its items twice: once in the table of contents and once
    as the actual sections. Both listings match the same heading pattern, so
    a naive first-match would usually capture a TOC page-number line ("Risk
    Factors ..... 15") instead of the risk factors themselves. A TOC entry is
    a few words; a real item is paragraphs. Keeping the longest body found
    for each item -- rather than the first -- is what picks the real section.
    """
    text = _to_text(html)
    matches = list(_HEADING.finditer(text))

    bodies: dict[str, str] = {}
    for index, match in enumerate(matches):
        item = match.group("item").upper()
        if item not in WANTED_ITEMS:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if len(body) > len(bodies.get(item, "")):
            bodies[item] = body

    missing = tuple(item for item in WANTED_ITEMS if item not in bodies)
    return ParsedFiling(items=bodies, missing=missing)
