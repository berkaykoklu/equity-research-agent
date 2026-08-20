"""Choosing where a 10-K item starts is judgment, not pattern matching.

Twenty real filings (see docs/parser-validation.md) proved that no regex can
reliably tell a genuine "Item 1" section start apart from the same text
appearing in a table of contents, a running page header repeated a dozen
times, or a back-of-document cross-reference index. Those all *look* like a
heading to a pattern matcher; telling them apart takes reading the document
the way a person would.

So the work is split three ways:
  - Code finds every place that could plausibly be a heading (this module's
    `build_candidates`) -- deliberately permissive, because a missed real
    heading can never be recovered downstream, while an extra decoy
    candidate costs nothing but a few tokens.
  - A model picks which candidates are real section boundaries (`select`
    below). It never sees the document's substantive prose and never
    returns text -- only integers referring back to the candidate list --
    so it cannot paraphrase, summarise, or drop a paragraph even if it
    wanted to.
  - Code (era.edgar.sections) does the actual cutting and verifies the
    result before trusting it.
"""

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

WANTED_ITEMS = ("1", "1A", "7")

# Item 8 (Financial Statements) is not one of the three items this parser
# extracts, but its heading is still worth finding: a 10-K numbers Item 8
# directly after Items 1, 1A and 7, in that fixed order, every single time.
# Without some later candidate to point at, Item 7 -- always the last of the
# three wanted items -- could only ever be bounded by another 1/1A/7 mention
# (rare) or run to the end of the document, swallowing exhibits, signatures
# and financial-statement footnotes into what should have been a clean
# section. Surfacing Item 8 gives the model a real anchor to end Item 7 at.
# It is never a candidate the model can *start* a section at, because the
# response schema below has no field for it -- it only ever gets used as an
# end_index.
_BOUNDARY_ANCHOR_ITEM = "8"
_CANDIDATE_ITEMS = frozenset(WANTED_ITEMS) | {_BOUNDARY_ANCHOR_ITEM}

CONTEXT_CHARS = 140

# Deliberately permissive: a delimiter is optional and the line need not be
# short. Precision is the model's job -- this only has to avoid missing a
# real heading. Anchored to the start of a line (see era.edgar.sections'
# _to_text: block tags become newlines, everything else does not), so a
# heading-shaped phrase buried mid-sentence -- "...as discussed in Item 7
# above" -- never counts as a candidate at all, no model judgment needed.
_CANDIDATE = re.compile(
    r"^[ \t]*item[ \t ]+(?P<item>\d{1,2}[A-Za-z]?)[ \t]*[.:\-‒–—]?",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class HeadingCandidate:
    index: int  # position in the candidate list -- what the model refers to
    item: str  # "1", "1A", "7", or "8" (boundary-anchor only, never selectable as a start)
    offset: int  # character offset of the match start in the full text
    context: str  # a short window so the model can judge what this line is


@dataclass(frozen=True)
class Boundary:
    start_index: int  # candidate index where the item's body begins
    end_index: int | None  # candidate index where it ends; None means end of document


def build_candidates(text: str) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []
    for match in _CANDIDATE.finditer(text):
        item = match.group("item").upper()
        if item not in _CANDIDATE_ITEMS:
            continue
        window = text[match.start() : match.start() + CONTEXT_CHARS]
        candidates.append(
            HeadingCandidate(
                index=len(candidates),
                item=item,
                offset=match.start(),
                context=" ".join(window.split()),
            )
        )
    return candidates


class BoundarySelector(Protocol):
    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]: ...


# --- model-backed selector -------------------------------------------------


class _ItemBoundary(BaseModel):
    start_index: int = Field(description="candidate index where this item's real body begins")
    end_index: int | None = Field(
        default=None,
        description="candidate index where this item's body ends; null means end of document",
    )


class _BoundaryResponse(BaseModel):
    """One field per wanted item. A null field means: no real section exists.

    Field names can't be "1" or "1A" (not valid Python identifiers), hence
    the item_-prefixed spelling; _FIELD_BY_ITEM maps them back.
    """

    item_1: _ItemBoundary | None = Field(
        default=None,
        description="Item 1 (Business), or null if this document has no real Item 1 section",
    )
    item_1a: _ItemBoundary | None = Field(
        default=None,
        description="Item 1A (Risk Factors), or null if this document has no real Item 1A section",
    )
    item_7: _ItemBoundary | None = Field(
        default=None,
        description="Item 7 (MD&A), or null if this document has no real Item 7 section",
    )


_FIELD_BY_ITEM = {"1": "item_1", "1A": "item_1a", "7": "item_7"}

_SYSTEM_PROMPT = """You choose where SEC 10-K item sections genuinely begin and end.

You are given every line in the document that matched a permissive "Item N" \
pattern, each as "index | item | context" -- a short snippet of the text \
around that line, not the section itself. Most candidates are NOT real \
section starts. Real filings contain several kinds of decoy, confirmed by \
auditing real filings from Microsoft, GE, JPMorgan, Chevron and others:

- A table of contents near the front lists every item once, next to a page
  number or a line of dots leading to one.
- Some filers repeat the item text as a running header on every single
  printed page, so the same item can legitimately appear a dozen or more
  times in a row. When that happens, the section's real body runs from the
  FIRST such repeat to the LAST -- not just one of them, and not the one
  with the most text around it.
- A cross-reference index near the back of the document lists items next to
  a page number or a page range (e.g. "24-31", "4-7, 9-10").
- Some items are only a pointer stub ("see pages 165-314", "incorporated by
  reference") with the real discussion elsewhere in the document, never
  under its own "Item N" heading anywhere you can point to.
- A few candidates are labelled item "8" (Financial Statements). You can
  never choose one of these as a section's start -- they exist only so you
  can use one as Item 7's end_index, since Item 8 always follows Item 7
  immediately in a 10-K and its heading is the correct place for Item 7 to
  stop.

For each of Item 1, Item 1A and Item 7, choose the candidate index where its
real body begins and the candidate index where it ends, or null for "runs to
the end of the document". If the document does not contain a real section for
an item -- only a table-of-contents entry, an index, or a stub -- return null
for that item entirely. Returning null is correct and expected when that is
the truth about the document, and is strongly preferred over guessing."""


def _format_candidates(candidates: list[HeadingCandidate]) -> str:
    lines = [f"{c.index} | {c.item} | {c.context}" for c in candidates]
    return "\n".join(lines)


class LlmBoundarySelector:
    """Asks a model which candidates are real section boundaries.

    The model sees only the candidate list built by `build_candidates` --
    never the document's actual prose, and never anything it could
    paraphrase or reproduce. It answers in integers, which is what makes an
    unreliable model safe to use here: era.edgar.sections still verifies
    every body those integers produce before trusting it.
    """

    def __init__(self, model: object) -> None:
        # `model` is a BaseChatModel; typed loosely here (see
        # era.graph.models.build_model) so this module doesn't need
        # langchain_core.language_models imported just for a type hint.
        self._structured = model.with_structured_output(_BoundaryResponse)  # type: ignore[attr-defined]

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        if not candidates:
            # Nothing to ask about -- and nothing worth spending a model
            # call on.
            return {}

        response = self._structured.invoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _format_candidates(candidates)),
            ]
        )
        if not isinstance(response, _BoundaryResponse):
            # with_structured_output can be configured to return a raw dict
            # instead of the pydantic model; this project never does that,
            # but if it ever changed, failing closed (nothing chosen, every
            # item ends up `missing`) is safer than guessing at a shape.
            return {}

        chosen: dict[str, Boundary] = {}
        for item, field in _FIELD_BY_ITEM.items():
            raw = getattr(response, field)
            if raw is None:
                continue
            chosen[item] = Boundary(start_index=raw.start_index, end_index=raw.end_index)
        return chosen


# --- caching wrapper ---------------------------------------------------


class CachedBoundarySelector:
    """Wraps another selector so a filing's boundary decision is made once.

    A filed document never changes, so re-deciding its boundaries on every
    run would spend a model call to get the same answer back -- and, worse,
    risks a *different* answer if the model is ever swapped or isn't
    perfectly deterministic. The candidate list (each candidate's item and
    offset) is a stable proxy for "this exact document": it is derived
    deterministically from the raw filing text, so hashing it is equivalent
    to keying on the filing's accession number without this layer needing to
    know what an accession number is.

    Same atomic-write discipline as EdgarClient: write to a uniquely-named
    temp file, then os.replace onto the real path, so a crash mid-write can
    only ever leave the temp file damaged -- never a truncated cache entry
    that a later run mistakes for a complete, valid decision.
    """

    def __init__(self, inner: BoundarySelector, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, candidates: list[HeadingCandidate]) -> Path:
        key = json.dumps([[c.item, c.offset] for c in candidates], separators=(",", ":"))
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._cache_dir / f"{digest}.json"

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        path = self._cache_path(candidates)
        if path.exists():
            return _boundaries_from_json(json.loads(path.read_text(encoding="utf-8")))

        chosen = self._inner.select(candidates)
        self._write_cache(path, chosen)
        return chosen

    def _write_cache(self, path: Path, chosen: dict[str, Boundary]) -> None:
        payload = {
            item: {"start_index": boundary.start_index, "end_index": boundary.end_index}
            for item, boundary in chosen.items()
        }
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


def _boundaries_from_json(payload: dict[str, dict[str, int | None]]) -> dict[str, Boundary]:
    chosen: dict[str, Boundary] = {}
    for item, raw in payload.items():
        start = raw["start_index"]
        if start is None:  # pragma: no cover -- defensive; _write_cache never writes this
            continue
        end = raw["end_index"]
        chosen[item] = Boundary(start_index=start, end_index=end)
    return chosen
