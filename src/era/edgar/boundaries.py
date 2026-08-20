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
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from era.graph.models import CHEAP_MODEL, build_model

WANTED_ITEMS = ("1", "1A", "7")

CONTEXT_CHARS = 140

# A pathological or malformed document could in principle repeat "item" an
# unbounded number of times; without a cap, that would translate directly
# into an unbounded prompt. 400 is far beyond anything a real 10-K produces
# (the twenty-filing sample topped out at a few dozen, even counting every
# item number and every running-header repeat) -- this exists purely as a
# safety valve, not a tuning knob.
MAX_CANDIDATES = 400

# Deliberately permissive: a delimiter is optional and the line need not be
# short, and every item number is captured, not just the three this parser
# extracts (see build_candidates' own comment below for why). Precision is
# the model's job -- this only has to avoid missing a real heading. Anchored
# to the start of a line (see era.edgar.sections' _to_text: block tags
# become newlines, everything else does not), so a heading-shaped phrase
# buried mid-sentence -- "...as discussed in Item 7 above" -- never counts
# as a candidate at all, no model judgment needed. The character class
# between "item" and the number is `[ \t]`, plain space or tab -- _to_text
# already folds &nbsp; into a plain space before this pattern ever runs, so
# nothing extra is needed here for that.
_CANDIDATE = re.compile(
    r"^[ \t]*item[ \t]+(?P<item>\d{1,2}[A-Za-z]?)[ \t]*[.:\-‒–—]?",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class HeadingCandidate:
    index: int  # position in the candidate list -- what the model refers to
    item: str  # the item number as printed: "1", "1A", "2", "7A", "8", etc.
    offset: int  # character offset of the match start in the full text
    context: str  # a short window so the model can judge what this line is


@dataclass(frozen=True)
class Boundary:
    start_index: int  # candidate index where the item's body begins
    end_index: int | None  # candidate index where it ends; None means end of document


def build_candidates(text: str) -> list[HeadingCandidate]:
    # Every item number is a candidate, not just 1, 1A and 7 -- including
    # the ones this parser never extracts (2, 3, 1B, 7A, and so on). A 10-K
    # numbers its items in one unbroken sequence, and a real Item 1A section
    # ends wherever the next real item heading begins, whichever number that
    # happens to be: Properties, Legal Proceedings and Mine Safety routinely
    # sit between Risk Factors and MD&A, and Item 7A sits between Item 7 and
    # Item 8. An earlier version of this function filtered candidates down
    # to {1, 1A, 7, 8} on the theory that only Item 8 was needed as an end
    # anchor for Item 7; that missed that 1A and 7 both need an end anchor
    # too, and any item in between it filtered out was invisible to the
    # model as anywhere to stop -- so a selector had no way to end Item 1A
    # anywhere except at the real Item 7 heading, silently absorbing every
    # item in between into what should have been a short section. The
    # response schema below still only ever lets the model *start* a
    # section at Item 1, 1A or 7 -- every other item number is only ever
    # usable as an end_index.
    candidates: list[HeadingCandidate] = []
    for match in _CANDIDATE.finditer(text):
        if len(candidates) >= MAX_CANDIDATES:
            break
        item = match.group("item").upper()
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


class BoundarySelectionError(RuntimeError):
    """A selector could not produce a decision at all -- distinct from a
    selector correctly deciding "no real section exists" (an empty dict).

    That distinction is what CachedBoundarySelector relies on to know
    whether a result is safe to persist forever: a transient API error,
    a timeout, or a malformed response is not a fact about the document
    and must never be cached as though it were one. Only exceptions of
    this type propagate past era.edgar.sections.parse_items' own guard --
    see the try/except there -- so a selector failure degrades one filing
    to `missing`, not a crash that aborts an entire batch run.
    """


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
  times in a row. When that happens, choose the first candidate belonging to
  the NEXT item as its end -- not one of the repeats themselves, which would
  cut off everything printed after that page.
- A cross-reference index near the back of the document lists items next to
  a page number or a page range (e.g. "24-31", "4-7, 9-10").
- Some items are only a pointer stub ("see pages 165-314", "incorporated by
  reference") with the real discussion elsewhere in the document, never
  under its own "Item N" heading anywhere you can point to.
- Many candidates are for item numbers other than 1, 1A and 7 -- Properties,
  Legal Proceedings, Item 1B, Item 7A, Item 8, and so on. You can never
  choose one of these as a section's start -- only Items 1, 1A and 7 are
  ever real answers here -- but the nearest one after an item's real start
  is usually the right end_index, since a 10-K numbers its items in one
  fixed, unbroken sequence and nothing legitimately sits between one item's
  real content and the next item's heading.
- When several candidates exist for the SAME item -- a running header
  repeated across pages, or a running header plus the real section start --
  strongly prefer the candidate whose context contains that item's own
  title: "Business" for Item 1, "Risk Factors" for Item 1A, "Management's
  Discussion" for Item 7. A running page header repeats only the bare item
  number ("Item 1"), never the title, because it exists to tell a reader
  which Part/Item a printed page falls under, not to restate the section
  name. The real section heading is the one place that carries both the
  number and the title together. Do not default to whichever repeat happens
  to appear first in the candidate list -- a running header can start a page
  or two before the real section heading, and picking that earlier repeat
  will slice off the section's own opening and everything the reader would
  recognize as its title. This preference only ever chooses among genuine
  section-start candidates -- it never overrides the table-of-contents or
  cross-reference-index rules above, both of which can also contain the
  item's title (a TOC line reads "Item 1. Business ..... 4"); a candidate
  that looks like a TOC or index entry is never a real section start no
  matter how strongly its context matches the title. If NONE of an item's
  candidates contain the title -- the real heading was missed by the
  permissive pattern, or its title falls outside the short context window --
  fall back to the first repeat as the start rather than leaving the choice
  unmade.

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

    def __init__(self, model: BaseChatModel | None = None) -> None:
        # Defaulting to the cheap model here -- not build_model's own
        # default, which is the expensive drafting model -- is deliberate.
        # Boundary selection is an integers-only task with no reason to
        # ever need the expensive model; this is the one call site where
        # forgetting to override the default would silently multiply the
        # cost of every filing's boundary decision by 10-50x.
        resolved_model = model if model is not None else build_model(model_name=CHEAP_MODEL)
        self._structured = resolved_model.with_structured_output(_BoundaryResponse)

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        if not candidates:
            # Nothing to ask about -- and nothing worth spending a model
            # call on. This is a genuine, deterministic fact about the
            # document (it has no "item"-shaped text anywhere), not a
            # failure, so returning {} here is safe to cache.
            return {}

        try:
            response = self._structured.invoke(
                [
                    ("system", _SYSTEM_PROMPT),
                    ("human", _format_candidates(candidates)),
                ]
            )
        except Exception as exc:
            # A timeout, a rate limit, an auth failure surviving
            # build_model's retries, or with_structured_output raising on
            # a response it couldn't parse into JSON at all -- none of
            # these are a fact about the document, so they must never be
            # allowed to look like one. Raising a distinct type here (see
            # BoundarySelectionError's docstring) is what lets
            # CachedBoundarySelector tell "the model decided nothing" apart
            # from "the model never answered" and refuse to cache the
            # latter.
            raise BoundarySelectionError(f"boundary selection call failed: {exc}") from exc

        if not isinstance(response, _BoundaryResponse):
            # with_structured_output can be configured to return a raw dict
            # instead of the pydantic model; this project never does that,
            # but if it ever changed, this is an integration bug, not a
            # decision about the document -- same reasoning as the except
            # block above, so it gets the same treatment.
            raise BoundarySelectionError(
                f"expected a _BoundaryResponse, got {type(response).__name__}"
            )

        chosen: dict[str, Boundary] = {}
        for item, field in _FIELD_BY_ITEM.items():
            raw = getattr(response, field)
            if raw is None:
                continue
            chosen[item] = Boundary(start_index=raw.start_index, end_index=raw.end_index)
        return chosen


# --- caching wrapper ---------------------------------------------------

# Bump this whenever _SYSTEM_PROMPT, the candidate schema, or the model
# changes in a way that could change a decision. Without it, re-running
# validation against a warm cache after improving the prompt would silently
# keep measuring the *old* prompt's decisions and publish that as evidence
# the change worked.
#
# "2": added the title-preference bullet to _SYSTEM_PROMPT (prefer the
# candidate whose context contains the item's own title -- "Business",
# "Risk Factors", "Management's Discussion" -- over an earlier running-header
# repeat that carries only the bare item number). This is the first real bump
# of this constant: Task 19 measured MSFT's Item 1 landing on a running-header
# repeat one page before the real heading under version "1", which this
# prompt change exists to fix. Live-validated: MSFT recovered, no regression
# on the other nineteen filings.
#
# "3": clarified the same bullet without changing the tested behaviour --
# made explicit that the title preference never overrides the table-of-
# contents/index rules (a TOC line also contains the title, e.g. "Item 1.
# Business ..... 4", and must still be rejected as a TOC entry, not chosen
# for matching the title), and restored an explicit fallback ("first repeat
# wins") for the case where no candidate's context contains the title at
# all -- version "2" left that case unspecified, which a code-reviewer pass
# flagged as a real gap even though it never triggered in the 20-filing
# sample. Not re-run live: this is a clarification of intent for an
# untriggered edge case, not a change aimed at any observed failure, and the
# project's live-test budget is spent deliberately, not on speculative
# re-verification (see docs/parser-validation.md).
_DECISION_VERSION = "3"


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

    Only a successful inner decision is ever written: if `inner.select`
    raises (see BoundarySelectionError), that exception propagates from this
    method too, without touching the cache -- there is nothing to cache when
    the model never actually answered. A genuine decision, including a
    genuine "no real section here" (an empty dict), is cached and trusted
    from then on.

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
        key = json.dumps(
            {
                "version": _DECISION_VERSION,
                "candidates": [[c.item, c.offset] for c in candidates],
            },
            separators=(",", ":"),
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._cache_dir / f"{digest}.json"

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        path = self._cache_path(candidates)
        cached = self._read_cache(path)
        if cached is not None:
            return cached

        # If this raises, it propagates as-is -- no write below runs, so a
        # transient failure is never mistaken for a decision worth keeping.
        chosen = self._inner.select(candidates)
        self._write_cache(path, chosen)
        return chosen

    def _read_cache(self, path: Path) -> dict[str, Boundary] | None:
        # Any problem reading or parsing an existing cache file -- it was
        # truncated by a crash before this class existed, hand-edited,
        # written by some future incompatible version of this format -- is
        # treated as a plain cache miss: recompute via the inner selector,
        # then overwrite the bad file with a fresh, valid one. A corrupt
        # cache entry must never be trusted (it would otherwise resurface
        # as a crash deep inside era.edgar.sections' offset resolution,
        # nowhere near where the real problem is) and must never abort a
        # run either -- it degrades exactly like a selector failure does.
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return _boundaries_from_json(raw)

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


def _boundaries_from_json(payload: Any) -> dict[str, Boundary] | None:
    # `payload` is genuinely Any -- it came from json.loads on a file this
    # process didn't necessarily write in this exact shape -- so every
    # level here is a real runtime check, not a type-checker-only
    # annotation. Any shape mismatch at all returns None, which the caller
    # treats as a cache miss.
    if not isinstance(payload, dict):
        return None
    chosen: dict[str, Boundary] = {}
    for item, raw in payload.items():
        if not isinstance(item, str) or not isinstance(raw, dict):
            return None
        # _write_cache always writes both keys explicitly, so a genuine
        # cache entry never has one missing -- treating an absent key as
        # "default to None" here would let a truncated or hand-edited file
        # silently pass as valid instead of being recomputed.
        if "start_index" not in raw or "end_index" not in raw:
            return None
        start = raw["start_index"]
        if not isinstance(start, int) or isinstance(start, bool):
            return None
        end = raw.get("end_index")
        if end is not None and (not isinstance(end, int) or isinstance(end, bool)):
            return None
        chosen[item] = Boundary(start_index=start, end_index=end)
    return chosen
