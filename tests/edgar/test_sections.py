from pathlib import Path

from era.edgar.sections import _to_text, parse_items

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_a_word_split_across_inline_tags_stays_whole() -> None:
    # Berkshire's real filing does exactly this: an inline <span> boundary
    # falls in the middle of a word with no text-node space to rejoin it.
    html = "<p><span>Item 1. Busines</span><span>s Description</span></p><p>We sell things.</p>"

    assert "Business Description" in _to_text(html)
    assert "Busines s" not in _to_text(html)


def test_extracts_each_item_body() -> None:
    parsed = parse_items((FIXTURES / "filing_simple.html").read_text())

    assert "smartphones" in parsed.items["1"]
    assert "supply chain concentration" in parsed.items["1A"]
    assert "services growth" in parsed.items["7"]
    assert parsed.missing == ()


def test_item_bodies_stop_at_the_next_heading() -> None:
    parsed = parse_items((FIXTURES / "filing_simple.html").read_text())

    assert "supply chain" not in parsed.items["1"]


def test_reports_a_missing_item_rather_than_inventing_one() -> None:
    parsed = parse_items((FIXTURES / "filing_no_item_1a.html").read_text())

    assert "1A" not in parsed.items
    assert parsed.missing == ("1A",)


def test_prefers_the_real_section_over_the_table_of_contents_entry() -> None:
    # Every real 10-K lists its items twice — once in the TOC, once as content.
    # Taking the first match would capture "Risk Factors ..... 15" instead of
    # the risk factors themselves, and no other test in this file would notice.
    parsed = parse_items((FIXTURES / "filing_with_toc.html").read_text())

    assert "supply chain concentration" in parsed.items["1A"]
    assert "smartphones" in parsed.items["1"]
    assert "services growth" in parsed.items["7"]
    assert parsed.missing == ()


def test_a_late_cross_reference_does_not_swallow_the_rest_of_the_filing() -> None:
    # C1: "...as discussed in Item 7. Management's Discussion..." inside the
    # exhibit index/signature back matter matches the same heading pattern as
    # the real Item 7 heading. For that match, being the last one found, the
    # old unanchored parser ran its body to end-of-document and then let
    # longest-body-wins prefer that huge, wrong body over the real one.
    parsed = parse_items((FIXTURES / "filing_late_cross_reference.html").read_text())

    assert "services growth" in parsed.items["7"]
    assert "SIGNATURES" not in parsed.items["7"]
    assert "exhibit" not in parsed.items["7"].lower()
    assert parsed.missing == ()


def test_a_mid_body_cross_reference_does_not_truncate_its_own_item() -> None:
    # C2: Item 1's own paragraph mentions "Item 1A. Risk Factors" in passing.
    # The old unanchored parser matched that mention as a heading, cutting
    # Item 1's body off mid-sentence right before it and silently discarding
    # everything that followed the mention.
    parsed = parse_items((FIXTURES / "filing_mid_body_cross_reference.html").read_text())

    assert "smartphones" in parsed.items["1"]
    assert "retail footprint spans many countries" in parsed.items["1"]
    assert "supply chain concentration" in parsed.items["1A"]
    assert parsed.missing == ()


def test_a_short_real_item_is_not_beaten_by_a_descriptive_toc_line() -> None:
    # C3: smaller filers often write a one-line Item 1A ("None."). Against a
    # long, descriptive TOC entry for the same item, plain longest-body-wins
    # picks the TOC line -- length alone isn't enough to tell them apart.
    parsed = parse_items((FIXTURES / "filing_short_real_item.html").read_text())

    assert "None." in parsed.items["1A"]
    assert "cybersecurity" not in parsed.items["1A"]
    assert parsed.missing == ()


def test_nbsp_in_a_heading_is_still_recognized() -> None:
    # C4: EDGAR filings commonly write "Item&nbsp;1A." instead of a literal
    # space. Without unescaping entities first, "item\s+" never matches --
    # every wanted item in this fixture uses &nbsp; and none of them are
    # plain text elsewhere, so the old parser found nothing at all.
    parsed = parse_items((FIXTURES / "filing_nbsp_headings.html").read_text())

    assert "smartphones" in parsed.items["1"]
    assert "supply chain concentration" in parsed.items["1A"]
    assert "services growth" in parsed.items["7"]
    assert parsed.missing == ()


def test_em_dash_heading_delimiter_is_recognized() -> None:
    # I1: "Item 1A — Risk Factors" (em dash) is common in modern filings.
    # The old delimiter class had hyphen and en dash but not em dash, so
    # none of this fixture's headings matched at all.
    parsed = parse_items((FIXTURES / "filing_em_dash_headings.html").read_text())

    assert "smartphones" in parsed.items["1"]
    assert "supply chain concentration" in parsed.items["1A"]
    assert "services growth" in parsed.items["7"]
    assert parsed.missing == ()


def test_a_heading_with_no_delimiter_does_not_contaminate_the_previous_item() -> None:
    # I2: "ITEM 1A RISK FACTORS" has no punctuation after the item number.
    # The old parser required a delimiter unconditionally, so this heading
    # never matched and its content -- including the phrase this test looks
    # for -- landed inside Item 1's body instead, reported as Business.
    parsed = parse_items((FIXTURES / "filing_heading_without_delimiter.html").read_text())

    assert "supply chain concentration" in parsed.items["1A"]
    assert "supply chain concentration" not in parsed.items["1"]
    assert parsed.missing == ()


def test_script_and_style_content_does_not_leak_into_an_item_body() -> None:
    # I4: the tag stripper only removes tags, not <script>/<style> contents,
    # and an unescaped "<" inside either (e.g. "if (a < b)") makes it consume
    # everything up to the next literal ">" found anywhere later in the
    # document -- corrupting whatever real content that span crossed.
    parsed = parse_items((FIXTURES / "filing_script_and_style.html").read_text())

    assert "services growth" in parsed.items["7"]
    assert "var x" not in parsed.items["7"]
    assert "color: red" not in parsed.items["7"]


def test_an_exhibit_repeating_a_heading_does_not_beat_the_real_item() -> None:
    # I3 (partial mitigation, documented trade-off -- see _back_matter_boundary):
    # an exhibit can legitimately start its own block with "Item 1. Business"
    # while incorporating unrelated boilerplate by reference. That heading is
    # not a mid-sentence cross-reference (C1/C2's fix doesn't touch it) and
    # doesn't look like a TOC line (C3's fix doesn't touch it either), so a
    # long exhibit can still beat a short but genuine Item 1 on length alone.
    parsed = parse_items((FIXTURES / "filing_exhibit_repeats_heading.html").read_text())

    assert "small holding company" in parsed.items["1"]
    assert "incorporated by reference" not in parsed.items["1"]
