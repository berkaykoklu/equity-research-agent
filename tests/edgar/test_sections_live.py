"""Validate the item parser against real filings.

Skipped unless ERA_LIVE_EDGAR=1. Everything else in the suite runs offline;
this one deliberately hits EDGAR, because synthetic fixtures only ever encode
the assumptions of whoever wrote them.
"""

import os
from pathlib import Path

import pytest

from era.edgar.boundaries import CachedBoundarySelector, LlmBoundarySelector
from era.edgar.client import EdgarClient
from era.edgar.filings import latest_filings, resolve_cik
from era.edgar.sections import parse_items
from era.graph.models import CHEAP_MODEL, build_model

USER_AGENT = os.environ.get("EDGAR_USER_AGENT", "")

pytestmark = pytest.mark.skipif(
    os.environ.get("ERA_LIVE_EDGAR") != "1" or "@" not in USER_AGENT,
    reason="set ERA_LIVE_EDGAR=1 and a contactable EDGAR_USER_AGENT to run this",
)

CACHE_DIR = Path(".cache/edgar-live")
BOUNDARY_CACHE_DIR = Path(".cache/boundaries")

# A correctly extracted body opens with its own title. Anything else means the
# parser captured a table-of-contents line or ran past a cross-reference.
EXPECTED_OPENING = {
    "1": ("business",),
    "1A": ("risk factor",),
    "7": ("management", "discussion"),
}

# Large caps whose filings genuinely contain all three items, plus deliberate
# awkwardness: BRK-B files unconventionally, and the banks and energy names use
# very different document generators from the tech names.
TICKERS = [
    "AAPL",
    "MSFT",
    "BRK-B",
    "JPM",
    "XOM",
    "KO",
    "JNJ",
    "PG",
    "WMT",
    "CVX",
    "MRK",
    "T",
    "VZ",
    "PFE",
    "INTC",
    "CSCO",
    "BA",
    "CAT",
    "GE",
    "DIS",
]

OPENING_WINDOW = 120


def _annual_report_html(client: EdgarClient, ticker: str) -> str:
    cik = resolve_cik(client, ticker)
    filings = latest_filings(client, cik)
    annual = next(f for f in filings if f.form == "10-K")
    return client.get_text(annual.primary_document_url)


def _opening_matches(item: str, body: str) -> bool:
    opening = body[:OPENING_WINDOW].lower()
    return any(word in opening for word in EXPECTED_OPENING[item])


@pytest.fixture(scope="module")
def fetch_failures() -> dict[str, str]:
    """Populated by the `results` fixture with tickers that never reached the parser."""
    return {}


@pytest.fixture(scope="module")
def results(fetch_failures: dict[str, str]) -> dict[str, dict[str, str]]:
    """Parse every ticker once; the client's disk cache makes reruns cheap.

    A ticker can fail before parse_items ever sees it -- an unresolvable
    symbol, a missing 10-K, a network error -- and that is a fact about
    EDGAR's data, not about the parser. Recording it in fetch_failures and
    moving on keeps one bad ticker from erasing the other nineteen results.

    The selector is the real production one -- a cheap model wrapped in the
    per-accession cache -- built here rather than at module scope, so
    constructing it (which needs OPENAI_API_KEY) only ever happens when this
    module's tests actually run, never on collection of the offline suite.
    """
    client = EdgarClient(user_agent=USER_AGENT, cache_dir=CACHE_DIR)
    selector = CachedBoundarySelector(
        LlmBoundarySelector(build_model(model_name=CHEAP_MODEL)), BOUNDARY_CACHE_DIR
    )
    collected: dict[str, dict[str, str]] = {}
    for ticker in TICKERS:
        try:
            html = _annual_report_html(client, ticker)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
            fetch_failures[ticker] = f"{type(exc).__name__}: {exc}"
            continue
        parsed = parse_items(html, selector)
        collected[ticker] = parsed.items
    return collected


def test_every_ticker_was_fetched(fetch_failures: dict[str, str]) -> None:
    # A resolution or fetch failure is worth surfacing on its own: it means a
    # ticker never reached the parser at all, which the per-item assertions
    # below can't detect (they only ever see tickers that made it into
    # `results`).
    assert not fetch_failures, "\n".join(
        f"{ticker}: {reason}" for ticker, reason in sorted(fetch_failures.items())
    )


def test_every_extracted_item_opens_with_its_own_title(results) -> None:
    failures: list[str] = []
    for ticker, items in results.items():
        for item, body in items.items():
            if not _opening_matches(item, body):
                failures.append(f"{ticker} item {item}: opens with {body[:80]!r}")

    assert not failures, "\n".join(failures)


# Recorded baseline, measured across twenty real filings. Each entry is the
# parser being correct rather than failing: a filing that never labels its
# sections, or an Item 7 that is a pointer to unlabelled prose elsewhere in
# the document. A gap outside this set is a real regression.
KNOWN_UNAVAILABLE = {
    "GE": {"1", "1A", "7"},  # only "Item N" text is a back-of-document index
    "INTC": {"1", "1A", "7"},  # same shape as GE
    "JPM": {"7"},  # Item 7 reads "...pages 165-314"
    "CVX": {"7"},  # same incorporation-by-reference shape
}


def test_no_new_gaps_appear(results) -> None:
    # A ticker missing an item that isn't in KNOWN_UNAVAILABLE is a genuine
    # parser regression, not a known structural limitation -- fail loudly
    # rather than let a new gap blend in with the recorded ones.
    unexpected = {
        ticker: sorted(missing)
        for ticker, items in results.items()
        if (missing := (set(EXPECTED_OPENING) - set(items)) - KNOWN_UNAVAILABLE.get(ticker, set()))
    }

    assert not unexpected, f"unexpected gaps: {unexpected}"


def test_the_baseline_does_not_hide_a_recovered_item(results) -> None:
    # The inverse check matters as much as the one above: if a filing that
    # used to be missing an item now parses it, KNOWN_UNAVAILABLE is stale
    # and must be tightened -- a baseline that only ever grows is not a
    # baseline, it's blanket permission to fail.
    stale = {
        ticker: sorted(recovered)
        for ticker, expected_missing in KNOWN_UNAVAILABLE.items()
        if ticker in results and (recovered := expected_missing & set(results[ticker]))
    }

    assert not stale, f"baseline entries that no longer reproduce: {stale}"


def test_no_item_body_is_implausibly_short(results) -> None:
    # Item 1A on a company this size is always substantial. A body of a few
    # dozen characters means a table-of-contents line won the length contest.
    too_short = [
        f"{ticker} item {item}: {len(body)} chars"
        for ticker, items in results.items()
        for item, body in items.items()
        if len(body) < 500
    ]

    assert not too_short, "\n".join(too_short)
