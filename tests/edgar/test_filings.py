import os
import time

import httpx
import pytest
import respx

from era.edgar.client import EdgarClient
from era.edgar.filings import (
    MissingFilingError,
    UnknownTickerError,
    latest_filings,
    resolve_cik,
)

UA = "Berkay Koklu kokluberkay@gmail.com"

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

BRK_TICKERS = {
    "0": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway Inc."},
}

MALFORMED_TICKERS = {
    "0": {"cik_str": 1, "title": "Row missing a ticker field entirely"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
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

EMPTY_DOCUMENT_SUBMISSIONS = {
    "cik": "0000320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000200", "0000320193-24-000123"],
            "form": ["10-K", "10-K"],
            "filingDate": ["2024-12-01", "2024-11-01"],
            "primaryDocument": ["", "aapl-20240928.htm"],
        }
    },
}

OUT_OF_ORDER_SUBMISSIONS = {
    "cik": "0000320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-23-000050", "0000320193-24-000123"],
            "form": ["10-K", "10-K"],
            "filingDate": ["2023-11-03", "2024-11-01"],
            "primaryDocument": ["aapl-20230930.htm", "aapl-20240928.htm"],
        }
    },
}

ONLY_10Q_SUBMISSIONS = {
    "cik": "0000320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000070"],
            "form": ["10-Q"],
            "filingDate": ["2024-08-02"],
            "primaryDocument": ["aapl-20240629.htm"],
        }
    },
}

ONLY_10K_SUBMISSIONS = {
    "cik": "0000320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000123"],
            "form": ["10-K"],
            "filingDate": ["2024-11-01"],
            "primaryDocument": ["aapl-20240928.htm"],
        }
    },
}

WITH_AMENDMENT_SUBMISSIONS = {
    "cik": "0000320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000150", "0000320193-24-000123"],
            "form": ["10-K/A", "10-K"],
            "filingDate": ["2024-12-15", "2024-11-01"],
            "primaryDocument": ["aapl-20240928a.htm", "aapl-20240928.htm"],
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
def test_resolves_a_dotted_ticker_to_the_indexs_dashed_form() -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=BRK_TICKERS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    assert resolve_cik(client, " brk.b ") == "0001067983"


@respx.mock
def test_skips_a_malformed_index_entry_and_keeps_resolving() -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=MALFORMED_TICKERS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    assert resolve_cik(client, "AAPL") == "0000320193"


@respx.mock
def test_ticker_index_is_treated_as_a_24_hour_cache_not_forever(tmp_path) -> None:
    route = respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=tmp_path)

    resolve_cik(client, "AAPL")
    cache_file = next(tmp_path.glob("*.cache"))
    thirty_six_hours_ago = time.time() - (36 * 60 * 60)
    os.utime(cache_file, (thirty_six_hours_ago, thirty_six_hours_ago))

    resolve_cik(client, "AAPL")

    assert route.call_count == 2


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


@respx.mock
def test_skips_a_filing_with_an_empty_primary_document() -> None:
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=EMPTY_DOCUMENT_SUBMISSIONS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    filings = latest_filings(client, "0000320193")

    assert [f.form for f in filings] == ["10-K"]
    assert filings[0].accession == "0000320193-24-000123"


@respx.mock
def test_picks_the_most_recent_filing_regardless_of_list_order() -> None:
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=OUT_OF_ORDER_SUBMISSIONS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    filings = latest_filings(client, "0000320193")

    assert [f.form for f in filings] == ["10-K"]
    assert filings[0].accession == "0000320193-24-000123"


@respx.mock
def test_missing_10k_raises_naming_the_cik_and_form() -> None:
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=ONLY_10Q_SUBMISSIONS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    with pytest.raises(MissingFilingError, match="0000320193.*10-K"):
        latest_filings(client, "0000320193")


@respx.mock
def test_returns_10k_alone_when_no_10q_is_present() -> None:
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=ONLY_10K_SUBMISSIONS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    filings = latest_filings(client, "0000320193")

    assert [f.form for f in filings] == ["10-K"]


@respx.mock
def test_prefers_the_original_10k_over_a_later_amendment() -> None:
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=WITH_AMENDMENT_SUBMISSIONS)
    )
    client = EdgarClient(user_agent=UA, cache_dir=None)

    filings = latest_filings(client, "0000320193")

    assert [f.form for f in filings] == ["10-K"]
    assert filings[0].accession == "0000320193-24-000123"
