import httpx
import pytest
import respx

from era.edgar.client import EdgarClient
from era.edgar.filings import UnknownTickerError, latest_filings, resolve_cik

UA = "Berkay Koklu kokluberkay@gmail.com"

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

SUBMISSIONS = {
    "cik": "0000320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000123", "0000320193-24-000070"],
            "form": ["10-K", "10-Q"],
            "filingDate": ["2024-11-01", "2024-08-02"],
            "primaryDocument": ["aapl-20240928.htm", "aapl-20240629.htm"],
        }
    },
}


@respx.mock
def test_resolves_a_ticker_to_a_zero_padded_cik() -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    assert resolve_cik(client, "aapl") == "0000320193"


@respx.mock
def test_unknown_ticker_fails_loudly() -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    with pytest.raises(UnknownTickerError, match="ZZZZ"):
        resolve_cik(client, "ZZZZ")


@respx.mock
def test_returns_the_latest_10k_and_10q_with_document_urls() -> None:
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=SUBMISSIONS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    filings = latest_filings(client, "0000320193")

    assert [f.form for f in filings] == ["10-K", "10-Q"]
    assert filings[0].accession == "0000320193-24-000123"
    assert filings[0].primary_document_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )
