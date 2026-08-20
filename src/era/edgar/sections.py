import re
from dataclasses import dataclass
from html import unescape

WANTED_ITEMS = ("1", "1A", "7")

# 10-Ks number Item 8 (Financial Statements) directly after Items 1, 1A and
# 7, in that fixed order, every time. Its first line-anchored heading (see
# _HEADING) marks the end of the region where those three items can
# legitimately occur, so a later section -- exhibit index, signatures --
# that happens to repeat one of their headings verbatim is excluded from
# candidacy outright rather than merely losing a length contest it could
# still win by being long (see _back_matter_boundary and I3). This is a
# heuristic, not a guarantee: a filing where Item 8's heading itself goes
# undetected loses the guard and falls back to unrestricted longest-wins.
_BOUNDARY_ITEM = "8"

_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1>", flags=re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_TAG_NAME = re.compile(r"^<\s*/?\s*([a-zA-Z][a-zA-Z0-9]*)")
_INLINE_WHITESPACE = re.compile(r"[^\S\n]+")

# Tags whose open/close boundary marks a real break between blocks of text,
# as opposed to formatting inline within a run of text. Real item headings
# always start a block; heading-shaped text that appears mid-sentence (a
# cross-reference like "...as discussed in Item 7 above") never does. This
# distinction only matters because _HEADING is anchored to line starts:
# giving block tags a newline and every other tag nothing at all is what
# lets that anchor tell a genuine heading apart from prose that merely
# mentions one.
#
# Inline tags must be zero-width, not a space: some filers (Berkshire's
# filing is a confirmed real example) split a single word across sibling
# inline tags with no text-node space between them --
# `<span>Item 1. Busines</span><span>s Description</span>` -- because the
# split falls wherever the document's rendering pipeline happened to wrap,
# not at a word boundary. Substituting a space at every inline-tag boundary
# would land a space inside that word ("Busines s Description"); deleting
# the tag and inserting nothing lets the two text fragments rejoin exactly
# as they were written ("Business Description"). A genuine missing space
# between two inline elements is comparatively rare and, when it happens,
# produces two run-together words rather than a corrupted one -- a smaller
# and more honest failure than silently injecting whitespace that was never
# there.
#
# Anything not in this set -- including tags this parser doesn't recognize
# at all -- defaults to zero-width rather than a newline, which is the more
# conservative choice: it can only ever cost a false-negative heading match,
# never manufacture a false-positive one.
_BLOCK_TAGS = frozenset(
    {
        "html",
        "body",
        "p",
        "div",
        "br",
        "hr",
        "tr",
        "td",
        "th",
        "li",
        "table",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "header",
        "footer",
        "ul",
        "ol",
    }
)

_HEADING = re.compile(
    r"^item\s+(?P<item>\d{1,2}[A-Z]?)(?:\s*[.:\-–—‒])?",
    flags=re.IGNORECASE | re.MULTILINE,
)

# A table of contents lists every item alongside a page number, using the
# same "Item N. Title" text a real heading uses, so it produces a second,
# competing match for every wanted item. A TOC entry's body is a title
# followed by dot leaders and/or a bare page number; a real section's body
# is paragraphs. Rejecting anything with that shape -- before bodies are
# ever compared by length -- is what stops a long, descriptive TOC line
# from beating a genuinely short real section (an Item 1A that just says
# "None.", for instance, where length alone would pick the TOC line). The
# length cap keeps this from misfiring on real content that happens to end
# in a number.
_TOC_ENTRY_TAIL = re.compile(r"(\.{2,}\s*\d{1,4}|\s\d{1,4})\s*$")
_TOC_ENTRY_MAX_LENGTH = 200


@dataclass(frozen=True)
class ParsedFiling:
    items: dict[str, str]
    missing: tuple[str, ...]


def _strip_scripts_and_styles(html: str) -> str:
    # <script> and <style> bodies are code and CSS, not filing text, but the
    # generic tag stripper below only removes tags -- it would leave their
    # contents behind. Worse, an unescaped "<" inside either (a JS
    # comparison, a CSS child-combinator) breaks that stripper outright:
    # `<[^>]+>` treats it as the start of a tag and consumes everything up
    # to the next literal ">", wherever in the document that happens to be.
    # Removing these blocks whole -- tag and contents together -- before any
    # other processing avoids both problems at once.
    return _SCRIPT_OR_STYLE.sub("\n", html)


def _tag_replacement(match: re.Match[str]) -> str:
    name_match = _TAG_NAME.match(match.group(0))
    name = name_match.group(1).lower() if name_match else ""
    return "\n" if name in _BLOCK_TAGS else ""


def _to_text(html: str) -> str:
    without_scripts = _strip_scripts_and_styles(html)
    without_tags = _TAG.sub(_tag_replacement, without_scripts)
    # EDGAR filings commonly space out headings with "&nbsp;" ("Item&nbsp;1A.")
    # instead of a literal space character. Unescaping before whitespace
    # normalization turns that into U+00A0, which \s already matches, so
    # _HEADING needs no separate handling for it.
    unescaped = unescape(without_tags)
    single_spaced = _INLINE_WHITESPACE.sub(" ", unescaped)
    lines = (line.strip() for line in single_spaced.split("\n"))
    return "\n".join(line for line in lines if line)


def _looks_like_a_toc_entry(body: str) -> bool:
    return len(body) <= _TOC_ENTRY_MAX_LENGTH and bool(_TOC_ENTRY_TAIL.search(body))


def _back_matter_boundary(text: str, matches: list[re.Match[str]]) -> int:
    # The TOC mentions Item 8 too, ahead of every real section, so the
    # first match for it is unusable as a boundary -- it would exclude the
    # real Items 1, 1A and 7 that follow. The last match is the one to use:
    # in a well-formed filing, nothing legitimately says "Item 8" again
    # after the real Item 8 section itself begins.
    boundary = len(text)
    for match in matches:
        if match.group("item").upper() == _BOUNDARY_ITEM:
            boundary = match.start()
    return boundary


def parse_items(html: str) -> ParsedFiling:
    """Split a filing into the items we care about.

    Every 10-K lists its items twice: once in the table of contents and once
    as the actual sections, and both match the same heading pattern. Filers
    also cross-reference items by name in running prose ("...as discussed in
    Item 7 above"), which matches too if nothing stops it.

    _HEADING is anchored to the start of a line, so a heading-shaped
    cross-reference buried mid-sentence never counts as a match at all --
    only text that starts its own block does. What survives that is
    filtered again: _looks_like_a_toc_entry drops table-of-contents lines by
    shape, and _back_matter_boundary drops anything past the start of Item 8,
    since exhibits and signatures always come after it. Only once a
    candidate has passed both filters does the longest surviving body for
    each item win.
    """
    text = _to_text(html)
    matches = list(_HEADING.finditer(text))
    boundary = _back_matter_boundary(text, matches)

    bodies: dict[str, str] = {}
    for index, match in enumerate(matches):
        item = match.group("item").upper()
        if item not in WANTED_ITEMS:
            continue
        if match.start() >= boundary:
            continue

        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if _looks_like_a_toc_entry(body):
            continue

        # Longest body wins: a TOC line that slipped past the shape filter
        # above would still be short, while the real section -- even a
        # terse one like "None." -- is already sitting in `bodies` as a
        # valid candidate, so comparing lengths here is what finishes
        # telling them apart.
        if len(body) > len(bodies.get(item, "")):
            bodies[item] = body

    missing = tuple(item for item in WANTED_ITEMS if item not in bodies)
    return ParsedFiling(items=bodies, missing=missing)
