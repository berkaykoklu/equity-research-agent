import httpx
import pytest
import respx
from tests.fakes import FakeEmbedder, FirstMatchSelector, InMemoryChunkStore

from era.edgar.client import EdgarClient
from era.edgar.filings import Filing, MissingFilingError, UnknownTickerError
from era.index.chunking import DEFAULT_MAX_CHARS
from era.index.ingest import ingest_ticker
from era.index.store import StoredChunk

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
#
# BUSINESS_BODY is deliberately longer than DEFAULT_MAX_CHARS: every other
# item in this fixture fits in one chunk, which would leave era.index.
# chunking's windowing (and the `start` offset it records) never exercised
# through ingest_ticker at all.
BUSINESS_HEADING = "Item 1. Business"
_BUSINESS_UNIT = "We design, manufacture and market smartphones and related services. "
BUSINESS_BODY = _BUSINESS_UNIT * (DEFAULT_MAX_CHARS // len(_BUSINESS_UNIT) + 5)
RISK_HEADING = "Item 1A. Risk Factors"
RISK_BODY = "Risk factors affecting our business include supply chain concentration. " * 12
MDA_HEADING = "Item 7. Management's Discussion and Analysis"
MDA_BODY = "Our results of operations reflect continued revenue growth across all segments. " * 12


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
    assert result.failures == {}


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
    # were skipped. Assert against the store directly. A zero query vector
    # is fine here: this test only checks *how many* rows exist, not which
    # text landed under which item (see the query-based test below for that).
    _mock_edgar(forms=["10-K"], html=_filing_html(include_item_7=True))
    store = InMemoryChunkStore()

    result = ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,
    )

    hits = store.query([0.0] * len(FakeEmbedder().embed(["x"])[0]), None, None, k=1000)
    assert len(hits) == result.chunks_written
    assert {h.item for h in hits} == {"1", "1A", "7"}


@respx.mock
def test_ingest_retrieves_the_right_item_for_content_distinctive_to_it() -> None:
    # A zero query vector (used above for plain count checks) scores every
    # row identically via FakeEmbedder's cosine fallback, so it can never
    # catch content ending up under the wrong item -- e.g. a mutant that
    # embeds chunk.item instead of chunk.text would destroy retrieval in
    # production while still passing a zero-vector-only suite. Querying
    # with real, item-distinctive text and checking which item comes back
    # is what actually exercises "does retrieval work."
    _mock_edgar(forms=["10-K"], html=_filing_html(include_item_7=True))
    store = InMemoryChunkStore()

    ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,
    )

    embedder = FakeEmbedder()
    business_hit = store.query(embedder.embed(["smartphones"])[0], None, None, k=1)[0]
    assert business_hit.item == "1"

    risk_hit = store.query(embedder.embed(["supply chain risk"])[0], None, None, k=1)[0]
    assert risk_hit.item == "1A"

    mda_hit = store.query(embedder.embed(["revenue"])[0], None, None, k=1)[0]
    assert mda_hit.item == "7"


@respx.mock
def test_ingest_is_idempotent() -> None:
    _mock_edgar(forms=["10-K"], html=_filing_html(include_item_7=True))
    store = InMemoryChunkStore()
    client = EdgarClient(user_agent=UA, cache_dir=None)

    first = ingest_ticker("AAPL", client, FirstMatchSelector(), FakeEmbedder(), store)
    second = ingest_ticker("AAPL", client, FirstMatchSelector(), FakeEmbedder(), store)

    hits = store.query([0.0] * len(FakeEmbedder().embed(["x"])[0]), None, None, k=1000)
    assert first.chunks_written == second.chunks_written
    assert len(hits) == first.chunks_written


@respx.mock
def test_ingest_only_processes_the_10k_even_when_a_10q_is_also_on_file() -> None:
    # A 10-Q genuinely on file alongside the 10-K must not get ingested:
    # a 10-Q's Item 1 is titled "Financial Statements" (never matches
    # EXPECTED_OPENING["1"]), it has no Item 7 at all, and the section that
    # matters -- Item 2 -- isn't selectable. See the comment in
    # ingest_ticker. Both document URLs are mocked with valid, parseable
    # HTML so this test actually exercises the filter rather than relying
    # on an unmocked-request error to mask a missing one.
    tenk_accession = ACCESSION
    tenq_accession = "0000320193-24-000999"
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    respx.get(f"https://data.sec.gov/submissions/CIK{CIK}.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "accessionNumber": [tenk_accession, tenq_accession],
                        "form": ["10-K", "10-Q"],
                        "filingDate": ["2024-11-01", "2024-08-01"],
                        "primaryDocument": ["aapl-10k.htm", "aapl-10q.htm"],
                    }
                }
            },
        )
    )
    respx.get(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-10k.htm"
    ).mock(return_value=httpx.Response(200, text=_filing_html(include_item_7=True)))
    respx.get(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000999/aapl-10q.htm"
    ).mock(return_value=httpx.Response(200, text=_filing_html(include_item_7=True)))
    store = InMemoryChunkStore()

    result = ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,
    )

    assert result.accessions == (tenk_accession,)
    assert store.get(tenq_accession, 0) is None


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


@respx.mock
def test_ingest_lets_an_unknown_ticker_propagate() -> None:
    # resolve_cik raises UnknownTickerError for a ticker SEC doesn't
    # recognize -- same reasoning as the missing-10-K case above, and the
    # same requirement not to swallow it into an empty, misleading result.
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    store = InMemoryChunkStore()

    with pytest.raises(UnknownTickerError):
        ingest_ticker(
            "NOTATICKER",
            EdgarClient(user_agent=UA, cache_dir=None),
            FirstMatchSelector(),
            FakeEmbedder(),
            store,
        )


def _two_10k_filings() -> tuple[Filing, Filing]:
    # latest_filings can never actually return two 10-Ks for one ticker --
    # era.edgar.filings.latest_filings keeps at most one Filing per form.
    # But ingest_ticker's per-filing loop has to stay correct regardless of
    # how many 10-Ks it's ever handed, and today's 10-K-only filter (see the
    # comment in ingest_ticker) is exactly the kind of change that could
    # silently break the loop underneath it without a test like this one.
    return (
        Filing(
            accession="0000320193-24-000123",
            form="10-K",
            filing_date="2024-11-01",
            primary_document_url="https://www.sec.gov/Archives/edgar/data/320193/a/doc.htm",
        ),
        Filing(
            accession="0000320193-23-000456",
            form="10-K",
            filing_date="2023-11-01",
            primary_document_url="https://www.sec.gov/Archives/edgar/data/320193/b/doc.htm",
        ),
    )


@respx.mock
def test_ingest_processes_every_filing_without_dropping_or_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import era.index.ingest as ingest_module

    filing_a, filing_b = _two_10k_filings()
    monkeypatch.setattr(ingest_module, "resolve_cik", lambda client, ticker: CIK)
    monkeypatch.setattr(ingest_module, "latest_filings", lambda client, cik: [filing_a, filing_b])
    respx.get(filing_a.primary_document_url).mock(
        return_value=httpx.Response(200, text=_filing_html(include_item_7=False))
    )
    respx.get(filing_b.primary_document_url).mock(
        return_value=httpx.Response(200, text=_filing_html(include_item_7=True))
    )
    store = InMemoryChunkStore()

    result = ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,
    )

    assert result.accessions == (filing_a.accession, filing_b.accession)
    # Each filing keeps its own missing items -- not just the last one's.
    assert result.items_missing == {filing_a.accession: ("7",)}
    # Both filings' rows must coexist: an upsert scoped to the wrong
    # accession, or a second filing's delete reaching the first filing's
    # rows, would make one of these None.
    assert store.get(filing_a.accession, 0) is not None
    assert store.get(filing_b.accession, 0) is not None


class _RaisesForOneAccession:
    """Wraps InMemoryChunkStore, failing only when asked to store one accession.

    Simulates a store outage partway through a multi-filing run without
    needing a live database -- InMemoryChunkStore does the real work for
    every accession except the one under test.
    """

    def __init__(self, fail_accession: str) -> None:
        self._inner = InMemoryChunkStore()
        self._fail_accession = fail_accession

    def upsert(self, chunks: object, vectors: object) -> None:
        if any(c.accession == self._fail_accession for c in chunks):  # type: ignore[attr-defined]
            raise RuntimeError("simulated store outage")
        self._inner.upsert(chunks, vectors)  # type: ignore[arg-type]

    def query(
        self, vector: list[float], item_filter: str | None, accession_filter: str | None, k: int
    ) -> list[StoredChunk]:
        return self._inner.query(vector, item_filter, accession_filter, k)

    def get(self, accession: str, chunk_id: int) -> StoredChunk | None:
        return self._inner.get(accession, chunk_id)


@respx.mock
def test_ingest_records_a_per_filing_failure_and_keeps_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With a store that raises on the second filing, the first filing's
    # rows must stay committed and show up in the result -- not be lost
    # because the run as a whole "failed." ingest_ticker must never let one
    # filing's exception erase or block a filing that already succeeded.
    import era.index.ingest as ingest_module

    filing_a, filing_b = _two_10k_filings()
    monkeypatch.setattr(ingest_module, "resolve_cik", lambda client, ticker: CIK)
    monkeypatch.setattr(ingest_module, "latest_filings", lambda client, cik: [filing_a, filing_b])
    respx.get(filing_a.primary_document_url).mock(
        return_value=httpx.Response(200, text=_filing_html(include_item_7=True))
    )
    respx.get(filing_b.primary_document_url).mock(
        return_value=httpx.Response(200, text=_filing_html(include_item_7=True))
    )
    store = _RaisesForOneAccession(fail_accession=filing_b.accession)

    result = ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,  # type: ignore[arg-type]
    )

    assert result.accessions == (filing_a.accession,)
    # Pinned to an exact count, not just `> 0`: a surviving mutant moves
    # `chunks_written += len(chunks)` above the `if chunks:`/store.upsert
    # block, so a filing whose store write raised would still have its
    # chunk count added to the total -- chunks_written would report work
    # that never reached the store. Comparing against the store's own row
    # count for the surviving filing is what catches that: if filing_b's
    # (never-written) chunks leaked into the total, this equality breaks.
    hits = store.query([0.0] * len(FakeEmbedder().embed(["x"])[0]), None, None, k=1000)
    assert result.chunks_written == len(hits) > 0
    assert filing_b.accession in result.failures
    assert "simulated store outage" in result.failures[filing_b.accession]


@respx.mock
def test_ingest_prints_the_per_filing_traceback_to_stderr_not_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ingest_ticker's per-filing except block writes a debug traceback via
    # traceback.print_exc(file=sys.stderr) -- deliberately not stdout, since
    # era.cli is the project's stated only module that prints or formats,
    # and stdout is where a caller's report/parsing lives. Nothing asserted
    # the destination: swapping file=sys.stderr for file=sys.stdout would
    # still leave every other test in this module passing, since none of
    # them inspect either stream. Pin it directly with capsys.
    import era.index.ingest as ingest_module

    filing_a, filing_b = _two_10k_filings()
    monkeypatch.setattr(ingest_module, "resolve_cik", lambda client, ticker: CIK)
    monkeypatch.setattr(ingest_module, "latest_filings", lambda client, cik: [filing_a, filing_b])
    respx.get(filing_a.primary_document_url).mock(
        return_value=httpx.Response(200, text=_filing_html(include_item_7=True))
    )
    respx.get(filing_b.primary_document_url).mock(
        return_value=httpx.Response(200, text=_filing_html(include_item_7=True))
    )
    store = _RaisesForOneAccession(fail_accession=filing_b.accession)

    ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,  # type: ignore[arg-type]
    )

    captured = capsys.readouterr()
    assert "simulated store outage" in captured.err
    assert "Traceback (most recent call last)" in captured.err
    assert "simulated store outage" not in captured.out
    assert "Traceback" not in captured.out


@respx.mock
def test_ingest_a_zero_chunk_reingest_keeps_the_previous_runs_rows() -> None:
    # Deliberate, not a bug -- see the comment in ingest_ticker above the
    # `if chunks:` block. store.upsert (and the delete inside it) only ever
    # runs when a filing produces new chunks; a re-ingest whose document
    # fails verification for every item must leave a previous good run's
    # rows in place rather than deleting them and writing nothing. Pinning
    # this means a future "always upsert, even with zero chunks" change
    # shows up here as a real, visible behaviour change instead of quietly
    # passing as a cleanup.
    good_html = _filing_html(include_item_7=True)
    # Headings present (so candidates exist for the selector to pick) but
    # every body under era.edgar.sections.MIN_BODY_CHARS (500), so
    # verification rejects all three items regardless of selector.
    bad_html = (
        "<p>Item 1. Business</p><p>Too short.</p>"
        "<p>Item 1A. Risk Factors</p><p>Too short.</p>"
        "<p>Item 7. Management's Discussion and Analysis</p><p>Too short.</p>"
    )
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=TICKERS)
    )
    respx.get(f"https://data.sec.gov/submissions/CIK{CIK}.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "accessionNumber": [ACCESSION],
                        "form": ["10-K"],
                        "filingDate": ["2024-11-01"],
                        "primaryDocument": ["aapl.htm"],
                    }
                }
            },
        )
    )
    doc_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl.htm"
    respx.get(doc_url).mock(
        side_effect=[
            httpx.Response(200, text=good_html),
            httpx.Response(200, text=bad_html),
        ]
    )
    store = InMemoryChunkStore()
    client = EdgarClient(user_agent=UA, cache_dir=None)

    first = ingest_ticker("AAPL", client, FirstMatchSelector(), FakeEmbedder(), store)
    assert first.chunks_written > 0

    second = ingest_ticker("AAPL", client, FirstMatchSelector(), FakeEmbedder(), store)

    assert second.chunks_written == 0
    assert second.items_missing == {ACCESSION: ("1", "1A", "7")}
    hits = store.query([0.0] * len(FakeEmbedder().embed(["x"])[0]), None, None, k=1000)
    assert len(hits) == first.chunks_written


class _CountingEmbedder:
    """Wraps FakeEmbedder, counting how many times embed() itself is called.

    Distinguishes "one batched call per filing" from "one call per chunk" --
    both produce the same vectors and the same chunks_written, so nothing
    else in this suite can tell them apart. Against a metered embedding
    API, the difference is one request per filing versus one request per
    chunk (a 10-K routinely has hundreds).
    """

    def __init__(self) -> None:
        self.calls = 0
        self._inner = FakeEmbedder()

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return self._inner.embed(texts)


@respx.mock
def test_ingest_makes_exactly_one_embed_call_per_filing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pins "ingest cannot bypass batching": replacing the single
    # `embedder.embed([...])` call with a per-chunk loop would still produce
    # correct chunks, correct vectors and a correct chunks_written count --
    # nothing else in this suite would notice -- while turning one request
    # per filing into one request per chunk against a metered API.
    import era.index.ingest as ingest_module

    filing_a, filing_b = _two_10k_filings()
    monkeypatch.setattr(ingest_module, "resolve_cik", lambda client, ticker: CIK)
    monkeypatch.setattr(ingest_module, "latest_filings", lambda client, cik: [filing_a, filing_b])
    respx.get(filing_a.primary_document_url).mock(
        return_value=httpx.Response(200, text=_filing_html(include_item_7=True))
    )
    respx.get(filing_b.primary_document_url).mock(
        return_value=httpx.Response(200, text=_filing_html(include_item_7=True))
    )
    store = InMemoryChunkStore()
    counting_embedder = _CountingEmbedder()

    ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        counting_embedder,  # type: ignore[arg-type]
        store,
    )

    assert counting_embedder.calls == 2


@respx.mock
def test_ingest_result_items_missing_and_failures_are_immutable() -> None:
    # The comment on IngestResult argues MappingProxyType makes these
    # fields immutable; nothing verified that claim against what
    # ingest_ticker actually returns. A version of this test that builds an
    # IngestResult by hand with MappingProxyType({}) passed in directly only
    # proves MappingProxyType's own contract -- it would keep passing even
    # if `return IngestResult(...)` inside ingest_ticker were mutated to
    # pass the plain, unwrapped dicts it builds internally. Call
    # ingest_ticker for real and assert the *returned* object's fields
    # reject mutation.
    _mock_edgar(forms=["10-K"], html=_filing_html(include_item_7=False))
    store = InMemoryChunkStore()

    result = ingest_ticker(
        "AAPL",
        EdgarClient(user_agent=UA, cache_dir=None),
        FirstMatchSelector(),
        FakeEmbedder(),
        store,
    )

    with pytest.raises(TypeError):
        result.items_missing["7"] = ("oops",)  # type: ignore[index]

    with pytest.raises(TypeError):
        result.failures[ACCESSION] = "oops"  # type: ignore[index]
