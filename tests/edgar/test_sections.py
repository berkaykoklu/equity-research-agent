from pathlib import Path

from tests.fakes import FirstMatchSelector, ScriptedSelector

from era.edgar.boundaries import Boundary, HeadingCandidate, build_candidates
from era.edgar.sections import _to_text, parse_items

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _candidates(fixture: str) -> list[HeadingCandidate]:
    html = (FIXTURES / fixture).read_text()
    return build_candidates(_to_text(html))


def _nth_index(candidates: list[HeadingCandidate], item: str, occurrence: int) -> int:
    """occurrence=0 is the first candidate for `item`, 1 is the second, etc."""
    matches = [c.index for c in candidates if c.item == item]
    return matches[occurrence]


def test_a_word_split_across_inline_tags_stays_whole() -> None:
    # Berkshire's real filing does exactly this: an inline <span> boundary
    # falls in the middle of a word with no text-node space to rejoin it.
    html = "<p><span>Item 1. Busines</span><span>s Description</span></p><p>We sell things.</p>"

    assert "Business Description" in _to_text(html)
    assert "Busines s" not in _to_text(html)


def test_extracts_each_item_body() -> None:
    parsed = parse_items((FIXTURES / "filing_simple.html").read_text(), FirstMatchSelector())

    assert "smartphones" in parsed.items["1"]
    assert "supply chain concentration" in parsed.items["1A"]
    assert "services growth" in parsed.items["7"]
    assert parsed.missing == ()


def test_item_bodies_stop_at_the_next_heading() -> None:
    parsed = parse_items((FIXTURES / "filing_simple.html").read_text(), FirstMatchSelector())

    assert "supply chain" not in parsed.items["1"]


def test_reports_a_missing_item_rather_than_inventing_one() -> None:
    parsed = parse_items((FIXTURES / "filing_no_item_1a.html").read_text(), FirstMatchSelector())

    assert "1A" not in parsed.items
    assert parsed.missing == ("1A",)


def test_a_selector_that_skips_the_toc_extracts_the_real_section() -> None:
    # Every real 10-K lists its items twice -- once in the TOC, once as
    # content -- and build_candidates surfaces both as candidates, permissive
    # by design. Telling them apart is the selector's job now, not code's:
    # given the *real* section's indices (the second occurrence of each
    # item), slicing and verification produce the real content even though
    # the TOC line for the same item sits right there in the candidate list.
    fixture = "filing_with_toc.html"
    candidates = _candidates(fixture)
    boundaries = {
        "1": Boundary(_nth_index(candidates, "1", 1), _nth_index(candidates, "1A", 1)),
        "1A": Boundary(_nth_index(candidates, "1A", 1), _nth_index(candidates, "7", 1)),
        "7": Boundary(_nth_index(candidates, "7", 1), _nth_index(candidates, "8", 1)),
    }
    parsed = parse_items((FIXTURES / fixture).read_text(), ScriptedSelector(boundaries))

    assert "supply chain concentration" in parsed.items["1A"]
    assert "smartphones" in parsed.items["1"]
    assert "services growth" in parsed.items["7"]
    assert "....." not in parsed.items["1A"]
    assert parsed.missing == ()


def test_a_naive_selector_on_a_toc_only_document_is_caught_by_verification() -> None:
    # The flip side of the test above: FirstMatchSelector has no judgment at
    # all, so on a TOC-first document it picks the TOC line -- a dot-leader
    # and a page number, a few dozen characters. MIN_BODY_CHARS is what
    # keeps that from being returned as if it were the real section.
    parsed = parse_items((FIXTURES / "filing_with_toc.html").read_text(), FirstMatchSelector())

    assert parsed.missing == ("1", "1A", "7")


def test_a_late_cross_reference_does_not_swallow_the_rest_of_the_filing() -> None:
    # "...as discussed in Item 7. Management's Discussion..." inside the
    # exhibit index/signature back matter is mid-sentence, so it was never a
    # candidate in the first place (build_candidates is line-anchored). What
    # actually bounds Item 7 here is the Item 8 heading right after it --
    # build_candidates surfaces Item 8 purely as an anchor for exactly this
    # -- so even the naive first-match selector keeps exhibits and
    # signatures out.
    parsed = parse_items(
        (FIXTURES / "filing_late_cross_reference.html").read_text(), FirstMatchSelector()
    )

    assert "services growth" in parsed.items["7"]
    assert "SIGNATURES" not in parsed.items["7"]
    assert "exhibit" not in parsed.items["7"].lower()
    assert parsed.missing == ()


def test_a_mid_body_cross_reference_does_not_truncate_its_own_item() -> None:
    # Item 1's own paragraph mentions "Item 1A. Risk Factors" in passing,
    # mid-sentence. build_candidates never turns that into a candidate at
    # all (it isn't at the start of a line), so there is nothing for any
    # selector to mistakenly pick -- Item 1's body runs intact to the next
    # real heading.
    parsed = parse_items(
        (FIXTURES / "filing_mid_body_cross_reference.html").read_text(), FirstMatchSelector()
    )

    assert "smartphones" in parsed.items["1"]
    assert "retail footprint spans many countries" in parsed.items["1"]
    assert "supply chain concentration" in parsed.items["1A"]
    assert parsed.missing == ()


def test_a_body_below_the_minimum_length_is_dropped_even_when_correctly_chosen() -> None:
    # Smaller filers sometimes write a genuinely one-line Item 1A ("None.").
    # Even pointing a selector at exactly the right candidate can't save it:
    # MIN_BODY_CHARS is a deliberate trade documented in sections.py -- no
    # real item in Task 18's twenty-filing sample was ever this short, only
    # decoys were, so a body this size is treated as suspicious regardless
    # of which candidate produced it.
    fixture = "filing_short_real_item.html"
    candidates = _candidates(fixture)
    boundaries = {
        "1": Boundary(_nth_index(candidates, "1", 1), _nth_index(candidates, "1A", 1)),
        "1A": Boundary(_nth_index(candidates, "1A", 1), _nth_index(candidates, "7", 1)),
        "7": Boundary(_nth_index(candidates, "7", 1), _nth_index(candidates, "8", 0)),
    }
    parsed = parse_items((FIXTURES / fixture).read_text(), ScriptedSelector(boundaries))

    assert "1" in parsed.items
    assert "7" in parsed.items
    assert "1A" not in parsed.items
    assert parsed.missing == ("1A",)


def test_nbsp_in_a_heading_is_still_recognized() -> None:
    # EDGAR filings commonly write "Item&nbsp;1A." instead of a literal
    # space. Without unescaping entities first, the candidate pattern never
    # matches -- every wanted item in this fixture uses &nbsp; and none of
    # them are plain text elsewhere, so no candidate would exist at all.
    parsed = parse_items((FIXTURES / "filing_nbsp_headings.html").read_text(), FirstMatchSelector())

    assert "smartphones" in parsed.items["1"]
    assert "supply chain concentration" in parsed.items["1A"]
    assert "services growth" in parsed.items["7"]
    assert parsed.missing == ()


def test_em_dash_heading_delimiter_is_recognized() -> None:
    # "Item 1A — Risk Factors" (em dash) is common in modern filings.
    parsed = parse_items(
        (FIXTURES / "filing_em_dash_headings.html").read_text(), FirstMatchSelector()
    )

    assert "smartphones" in parsed.items["1"]
    assert "supply chain concentration" in parsed.items["1A"]
    assert "services growth" in parsed.items["7"]
    assert parsed.missing == ()


def test_a_heading_with_no_delimiter_does_not_contaminate_the_previous_item() -> None:
    # "ITEM 1A RISK FACTORS" has no punctuation after the item number, which
    # the candidate pattern's optional delimiter must still match as its own
    # heading rather than letting its content run on as part of Item 1.
    parsed = parse_items(
        (FIXTURES / "filing_heading_without_delimiter.html").read_text(), FirstMatchSelector()
    )

    assert "supply chain concentration" in parsed.items["1A"]
    assert "supply chain concentration" not in parsed.items["1"]
    assert parsed.missing == ()


def test_script_and_style_content_does_not_leak_into_an_item_body() -> None:
    # The tag stripper only removes tags, not <script>/<style> contents, and
    # an unescaped "<" inside either (e.g. "if (a < b)") would make a naive
    # stripper consume everything up to the next literal ">" found anywhere
    # later in the document -- corrupting whatever real content that span
    # crossed, if _strip_scripts_and_styles didn't remove these blocks whole
    # first.
    parsed = parse_items(
        (FIXTURES / "filing_script_and_style.html").read_text(), FirstMatchSelector()
    )

    assert "services growth" in parsed.items["7"]
    assert "var x" not in parsed.items["7"]
    assert "color: red" not in parsed.items["7"]


def test_an_exhibit_repeating_a_heading_does_not_beat_the_real_item() -> None:
    # An exhibit can legitimately start its own block with "Item 1. Business"
    # deep in the back matter while incorporating unrelated boilerplate by
    # reference, and that repeat is considerably longer than the genuine,
    # terse Item 1 section above it. The old parser's longest-body-wins rule
    # would have picked the exhibit; first-occurrence-wins, which this
    # architecture uses even in its naive fallback selector, cannot be
    # fooled by a later repeat being long, by construction -- it never looks
    # past the first candidate for an item it has already chosen.
    parsed = parse_items(
        (FIXTURES / "filing_exhibit_repeats_heading.html").read_text(), FirstMatchSelector()
    )

    assert "small holding company" in parsed.items["1"]
    assert "incorporated by reference" not in parsed.items["1"]


def test_a_running_page_header_body_spans_the_whole_section_not_one_page() -> None:
    # Microsoft's real filing repeats a bare "Item 1" as a page-break marker
    # roughly a dozen times per item. Given the correct boundary -- from the
    # FIRST repeat to the start of the next real item -- the sliced body
    # must contain every page's worth of content in between, not just
    # whichever page happens to be longest.
    fixture = "filing_running_page_header.html"
    candidates = _candidates(fixture)
    assert len([c for c in candidates if c.item == "1"]) > 1, "fixture must repeat Item 1"
    boundaries = {
        "1": Boundary(_nth_index(candidates, "1", 0), _nth_index(candidates, "1A", 0)),
    }
    parsed = parse_items((FIXTURES / fixture).read_text(), ScriptedSelector(boundaries))

    assert "business" in parsed.items["1"][:120].lower()
    assert "server products" in parsed.items["1"]
    assert "heart of our entertainment ecosystem" in parsed.items["1"]
    assert "productivity and business process" in parsed.items["1"]


def test_a_running_page_header_defeats_the_naive_selector_but_verification_catches_it() -> None:
    # FirstMatchSelector has no way to know the repeats are the same
    # section -- it locks onto the very first "Item 1" occurrence and stops
    # at the very next candidate, which here is the second page-header
    # repeat a couple hundred characters later. That fragment is too short
    # to clear MIN_BODY_CHARS, so it is dropped rather than returned as a
    # truncated "Item 1".
    parsed = parse_items(
        (FIXTURES / "filing_running_page_header.html").read_text(), FirstMatchSelector()
    )

    assert "1" not in parsed.items


def test_a_back_matter_index_with_page_ranges_is_reported_missing_not_captured() -> None:
    # GE's real filing's only "Item 1"/"Item 1A"/"Item 7" text anywhere is a
    # cross-reference index near the end, mapping each item to a page range
    # ("4-7, 9-10, 71-73"). The genuine Business/Risk Factors/MD&A content
    # exists in this fixture too, but under plain topic titles with no
    # "Item" prefix -- invisible to a heading-anchored parser by
    # construction, not something any selector choice can recover. Even the
    # naive selector -- which happily locks onto the index lines, since
    # they're the only candidates that exist -- gets caught by
    # MIN_BODY_CHARS: an index entry is a title and a page range, nowhere
    # near 500 characters.
    parsed = parse_items(
        (FIXTURES / "filing_back_matter_index.html").read_text(), FirstMatchSelector()
    )

    assert parsed.missing == ("1", "1A", "7")
    assert "4-7, 9-10, 71-73" not in "".join(parsed.items.values())


def test_a_pointer_stub_is_reported_missing_not_returned_as_the_real_section() -> None:
    # JPMorgan and Chevron's real Item 7 headings are genuine, but the body
    # underneath is a short pointer stub ("...which appear on pages
    # 165-314") with the real MD&A prose elsewhere in the document, never
    # under an "Item 7" heading. The stub's own heading text happens to
    # contain "management" and "discussion", so EXPECTED_OPENING alone
    # would pass it -- MIN_BODY_CHARS is what actually catches this case.
    parsed = parse_items((FIXTURES / "filing_pointer_stub.html").read_text(), FirstMatchSelector())

    assert parsed.missing == ("7",)
    assert "1" in parsed.items
    assert "1A" in parsed.items


def test_a_word_split_heading_survives_the_full_parse_path() -> None:
    # Berkshire's real filing splits "Business" across sibling <span> tags
    # with no text-node space between them. _to_text's fix (see
    # test_a_word_split_across_inline_tags_stays_whole) has to hold up
    # through the rest of the pipeline too: build_candidates must still
    # recognize the rejoined heading, and the verified body must open with
    # the word intact.
    parsed = parse_items(
        (FIXTURES / "filing_word_split_heading.html").read_text(), FirstMatchSelector()
    )

    assert "Business Description" in parsed.items["1"]
    assert "Busines s" not in parsed.items["1"]
    assert "Risk Factors" in parsed.items["1A"]
    assert "Ris k" not in parsed.items["1A"]


def test_an_out_of_range_index_is_rejected_without_raising() -> None:
    # A selector's chosen index is untrusted input -- it could come from a
    # model hallucinating a number that was never in the candidate list.
    # This is the case step 7 of the brief calls out as mattering most: it
    # proves the design stays safe even when the selector is simply wrong.
    fixture = "filing_simple.html"
    candidates = _candidates(fixture)
    boundaries = {"1": Boundary(start_index=len(candidates) + 5, end_index=None)}
    parsed = parse_items((FIXTURES / fixture).read_text(), ScriptedSelector(boundaries))

    assert "1" not in parsed.items
    assert "1" in parsed.missing


def test_a_wrong_but_in_range_index_is_rejected_by_the_opening_check() -> None:
    # Here the selector's index is valid but wrong -- it points Item 1 at
    # Item 7's real heading instead. The sliced body opens with "Item 7.
    # Management's Discussion...", which doesn't contain "business" in the
    # opening window, so EXPECTED_OPENING rejects it even though nothing
    # about the slice itself was invalid. (Item 1A's body was deliberately
    # not used for this: its own filler text starts "Our business is
    # subject to...", which would accidentally satisfy Item 1's check.)
    fixture = "filing_simple.html"
    candidates = _candidates(fixture)
    boundaries = {
        "1": Boundary(
            _nth_index(candidates, "7", 0),
            _nth_index(candidates, "8", 0),
        ),
    }
    parsed = parse_items((FIXTURES / fixture).read_text(), ScriptedSelector(boundaries))

    assert "1" not in parsed.items
    assert "1" in parsed.missing
