import re
from dataclasses import dataclass
from html import unescape

from era.edgar.boundaries import (
    WANTED_ITEMS,
    Boundary,
    BoundarySelector,
    HeadingCandidate,
    build_candidates,
)

_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1>", flags=re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_TAG_NAME = re.compile(r"^<\s*/?\s*([a-zA-Z][a-zA-Z0-9]*)")
_INLINE_WHITESPACE = re.compile(r"[^\S\n]+")

# Tags whose open/close boundary marks a real break between blocks of text,
# as opposed to formatting inline within a run of text. Real item headings
# always start a block; heading-shaped text that appears mid-sentence (a
# cross-reference like "...as discussed in Item 7 above") never does. This
# distinction only matters because era.edgar.boundaries' candidate pattern is
# anchored to line starts: giving block tags a newline and every other tag
# nothing at all is what lets that anchor tell a genuine heading apart from
# prose that merely mentions one.
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

# A correctly chosen body opens with its own title -- "Business", "Risk
# Factors", "Management's Discussion...". Anything else means the selector
# picked a table-of-contents line, a running page header, a back-of-document
# index entry, or a cross-reference stub, none of which open that way. Task
# 18 measured this exact check across twenty real filings (see
# docs/parser-validation.md); it is the only guard that caught every failure
# the old pure-regex parser missed, which is what makes it safe to hand
# boundary selection to a model that can be wrong: a wrong pick fails this
# check and the item is reported `missing` rather than returned as if it
# were correct.
EXPECTED_OPENING = {
    "1": ("business",),
    "1A": ("risk factor",),
    "7": ("management", "discussion"),
}
OPENING_WINDOW = 120

# A table-of-contents line, a back-of-document index entry ("Business
# 4-7, 9-10, 71-73"), and an incorporated-by-reference stub ("...which
# appear on pages 165-314") are all short -- Task 18's twenty-filing sample
# found no genuine item body under a few thousand characters, and no decoy
# over a few hundred. 500 sits well inside that gap. The cost is a
# deliberate trade: a legitimately terse real item (a small filer's Item 1A
# that just says "None.") would also fail this check and be reported
# missing rather than returned -- accepted because no such item ever turned
# up in the real sample, and a false "missing" is a far smaller problem than
# a false, silently wrong body.
MIN_BODY_CHARS = 500

# Neither check above catches a selector that picks a table-of-contents
# line as an item's *start* and a later real heading as its *end*: the
# slice runs from "Item 1. Business ..... 3" through every other TOC entry
# up to the real Item 1A heading, opens with the word "Business" same as
# the genuine section would, and is easily long enough to clear
# MIN_BODY_CHARS -- it just happens to be front matter, not content. A run
# of five or more dots is the one shape a genuine section body never has
# (dot leaders are a table-of-contents-specific typesetting convention) but
# a TOC-shaped body reliably does, so it is a direct, cheap test for
# exactly this failure regardless of length or opening words.
_DOT_LEADER = re.compile(r"\.{5,}")


@dataclass(frozen=True)
class ParsedFiling:
    # `items`' iteration order is part of this dataclass's contract, not an
    # implementation detail: parse_items below builds it by inserting in
    # WANTED_ITEMS (filing) order, and era.index.chunking.chunk_items relies
    # on that order rather than re-sorting keys itself. A caller that
    # rebuilds this dict from another source (e.g. `json.loads`) must
    # preserve filing order or downstream chunk ordering silently changes.
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
    # era.edgar.boundaries' candidate pattern needs no separate handling for it.
    unescaped = unescape(without_tags)
    single_spaced = _INLINE_WHITESPACE.sub(" ", unescaped)
    lines = (line.strip() for line in single_spaced.split("\n"))
    return "\n".join(line for line in lines if line)


def _resolve_offset(candidates: list[HeadingCandidate], index: int) -> int | None:
    # A selector's index is untrusted input, whether it came from a model or
    # a bug in a future selector implementation. Out-of-range means the
    # choice cannot be honoured at all; the caller treats that exactly like
    # any other verification failure -- dropped, never guessed at.
    if 0 <= index < len(candidates):
        return candidates[index].offset
    return None


def _verified_body(
    item: str,
    text: str,
    candidates: list[HeadingCandidate],
    boundary: Boundary,
    minimum_start_offset: int | None,
) -> tuple[str, int] | None:
    start_offset = _resolve_offset(candidates, boundary.start_index)
    if start_offset is None:
        return None

    # Items are always filed in the fixed order 1, 1A, 7 -- WANTED_ITEMS'
    # own order, which parse_items iterates in below. A start offset that
    # doesn't come strictly after the previous accepted item's start means
    # the selector picked something out of order (pointed two items at the
    # same heading, or went backward), which is never correct in a
    # well-formed filing and is worth rejecting on its own, independent of
    # the dot-leader check below.
    if minimum_start_offset is not None and start_offset <= minimum_start_offset:
        return None

    end_offset: int
    if boundary.end_index is None:
        end_offset = len(text)
    else:
        resolved_end = _resolve_offset(candidates, boundary.end_index)
        if resolved_end is None:
            return None
        end_offset = resolved_end

    if end_offset <= start_offset:
        return None

    body = text[start_offset:end_offset].strip()
    if len(body) < MIN_BODY_CHARS:
        return None

    if _DOT_LEADER.search(body):
        return None

    opening = body[:OPENING_WINDOW].lower()
    if not any(word in opening for word in EXPECTED_OPENING[item]):
        return None

    return body, start_offset


def parse_items(html: str, selector: BoundarySelector) -> ParsedFiling:
    """Split a filing into the items we care about.

    Choosing which "Item N" occurrence is a genuine section start is a
    judgment call a regex cannot make reliably (see era.edgar.boundaries for
    why); this function owns everything downstream of that choice instead.
    build_candidates finds every place that could be a heading, `selector`
    -- a model, in production; a fake, in every test -- decides which
    candidates are real boundaries, and this function slices the raw text at
    those offsets and verifies the result before trusting it.

    Nothing here ever raises: a selector call itself can fail (a timeout, a
    malformed model response -- see era.edgar.boundaries.BoundarySelectionError)
    exactly as easily as a selector can simply choose wrong, and both
    degrade the same way -- the affected item is reported `missing`, never a
    crash and never a silently wrong body. That is what makes it safe to
    plug an unreliable selector into this function in the first place.
    """
    text = _to_text(html)
    candidates = build_candidates(text)
    try:
        chosen = selector.select(candidates)
    except Exception:  # noqa: BLE001 -- deliberately broad, see the docstring above
        chosen = {}

    bodies: dict[str, str] = {}
    previous_start_offset: int | None = None
    for item in WANTED_ITEMS:
        boundary = chosen.get(item)
        if boundary is None:
            continue
        result = _verified_body(item, text, candidates, boundary, previous_start_offset)
        if result is None:
            continue
        body, start_offset = result
        bodies[item] = body
        previous_start_offset = start_offset

    missing = tuple(item for item in WANTED_ITEMS if item not in bodies)
    return ParsedFiling(items=bodies, missing=missing)
