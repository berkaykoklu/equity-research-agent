from pathlib import Path

from era.edgar.sections import parse_items

FIXTURES = Path(__file__).parent.parent / "fixtures"


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
