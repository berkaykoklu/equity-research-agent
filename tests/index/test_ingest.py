import httpx
import pytest
import respx
from tests.fakes import FakeEmbedder, FirstMatchSelector, InMemoryChunkStore

from era.edgar.client import EdgarClient
from era.edgar.filings import MissingFilingError
from era.index.ingest import ingest_ticker

UA = "Berkay Koklu kokluberkay@gmail.com"
CIK = "0000320193"
ACCESSION = "0000320193-24-000123"
TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

# era.edgar.sections.parse_items rejects any item body under 500 characters
# (MIN_BODY_CHARS) and one whose opening 120 characters don't contain the
# item's own title -- both guards against a selector picking a table-of-
# contents line rather than the real section. The brief's original fixture
# ("We design smartphones.") predates both checks and would fail every item,
# not just the intentionally absent one, so these bodies are built long
# enough and with the right opening words to clear real verification.
BUSINESS_HEADING = "Item 1. Business"
BUSINESS_BODY = "We design, manufacture and market smartphones and related services. " * 12
RISK_HEADING = "Item 1A. Risk Factors"
RISK_BODY = "Risk factors affecting our business include supply chain concentration. " * 12
MDA_HEADING = "Item 7. Management's Discussion and Analysis"
MDA_BODY = "Our results of operations reflect continued growth across all segments. " * 12


def _filing_html(*, include_item_7: bool) -> str:
    sections = [
        f"<p>{BUSINESS_HEADING}</p><p>{BUSINESS_BODY}</p>",
        f"<p>{RISK_HEADING}</p><p>{RISK_BODY}</p>",
    ]
    if include_item_7:
        sections.append(f"<p>{MDA_HEADING}</p><p>{MDA_BODY}</p>")
    return "".join(sections)


def _mock_edgar(*, forms: list[str], html: str | None = None) -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    respx.get(f"https://data.sec.gov/submissions/CIK{CIK}.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "accessionNumber": [ACCESSION] * len(forms),
                        "form": forms,
                        "filingDate": ["2024-11-01"] * len(forms),
                        "primaryDocument": ["aapl.htm"] * len(forms),
                    }
                }
            },
        )
    )
    if html is not None:
        respx.get(
            "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl.htm"
        ).mock(return_value=httpx.Response(200, text=html))


@respx.mock
def test_ingest_writes_chunks_and_reports_a_missing_item() -> None:
    _mock_edgar(forms=["10-K"], html=_filing_html(include_item_7=False))
    store = InMemoryChunkStore()

    result = ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,
    )

    assert result.cik == CIK
    assert result.accessions == (ACCESSION,)
    assert result.chunks_written > 0
    assert result.items_missing == {ACCESSION: ("7",)}


@respx.mock
def test_ingest_reports_no_missing_items_when_every_item_is_found() -> None:
    _mock_edgar(forms=["10-K"], html=_filing_html(include_item_7=True))
    store = InMemoryChunkStore()

    result = ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,
    )

    # A success must not read as failure and a failure must not read as
    # success: with every item present, items_missing has to be empty, not
    # just "missing 7" as in the sibling test above.
    assert result.items_missing == {}


@respx.mock
def test_ingest_actually_writes_chunks_to_the_store() -> None:
    # chunks_written being nonzero isn't enough to prove real work happened
    # -- it would still pass if the count were computed but the store call
    # were skipped. Assert against the store directly.
    _mock_edgar(forms=["10-K"], html=_filing_html(include_item_7=True))
    store = InMemoryChunkStore()

    result = ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,
    )

    hits = store.query([0.0] * len(FakeEmbedder().embed(["x"])[0]), None, None, k=100)
    assert len(hits) == result.chunks_written
    assert {h.item for h in hits} == {"1", "1A", "7"}


@respx.mock
def test_ingest_attributes_each_chunk_to_the_item_its_text_actually_came_from() -> None:
    # A set-of-items check (above) would still pass if two chunks' text got
    # swapped between items -- the set of item labels present is unchanged,
    # only which text sits under which label. This checks content, not just
    # labels, to catch that class of bug.
    _mock_edgar(forms=["10-K"], html=_filing_html(include_item_7=True))
    store = InMemoryChunkStore()

    ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,
    )

    hits = store.query([0.0] * len(FakeEmbedder().embed(["x"])[0]), None, None, k=100)
    text_by_item = {h.item: h.text for h in hits}
    assert "smartphones" in text_by_item["1"]
    assert "supply chain" in text_by_item["1A"]
    assert "results of operations" in text_by_item["7"]


@respx.mock
def test_ingest_is_idempotent() -> None:
    _mock_edgar(forms=["10-K"], html=_filing_html(include_item_7=True))
    store = InMemoryChunkStore()
    client = EdgarClient(user_agent=UA, cache_dir=None)

    first = ingest_ticker("AAPL", client, FirstMatchSelector(), FakeEmbedder(), store)
    second = ingest_ticker("AAPL", client, FirstMatchSelector(), FakeEmbedder(), store)

    hits = store.query([0.0] * len(FakeEmbedder().embed(["x"])[0]), None, None, k=100)
    assert first.chunks_written == second.chunks_written
    assert len(hits) == first.chunks_written


@respx.mock
def test_ingest_lets_a_missing_10k_propagate_rather_than_swallowing_it() -> None:
    # No 10-K on file at all: latest_filings raises MissingFilingError, and
    # ingest_ticker must not catch it -- silently returning an empty
    # IngestResult would look like a company with no filings to research,
    # not like the real problem (no annual report exists to ingest).
    _mock_edgar(forms=["10-Q"])
    store = InMemoryChunkStore()

    with pytest.raises(MissingFilingError):
        ingest_ticker(
            "AAPL",
            EdgarClient(user_agent=UA, cache_dir=None),
            FirstMatchSelector(),
            FakeEmbedder(),
            store,
        )
