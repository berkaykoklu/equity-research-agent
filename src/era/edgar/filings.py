from dataclasses import dataclass

from era.edgar.client import EdgarClient

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
WANTED_FORMS = ("10-K", "10-Q")


class UnknownTickerError(LookupError):
    """The ticker is not present in SEC's company index."""


@dataclass(frozen=True)
class Filing:
    accession: str
    form: str
    filing_date: str
    primary_document_url: str


def resolve_cik(client: EdgarClient, ticker: str) -> str:
    index = client.get_json(TICKERS_URL)
    wanted = ticker.upper()
    for entry in index.values():
        if entry["ticker"].upper() == wanted:
            return str(entry["cik_str"]).zfill(10)
    raise UnknownTickerError(f"{wanted} is not a US-listed issuer in SEC's index")


def latest_filings(client: EdgarClient, cik: str) -> list[Filing]:
    data = client.get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = data["filings"]["recent"]

    # SEC's archive URL layout mixes two different CIK/accession spellings.
    # The folder segment is the accession number with its dashes stripped,
    # but the CIK segment drops the submissions endpoint's zero-padding
    # entirely. `cik` arrives here zero-padded to 10 digits because that is
    # what data.sec.gov/submissions requires; www.sec.gov/Archives wants the
    # bare, un-padded form instead.
    bare_cik = cik.lstrip("0")

    found: list[Filing] = []
    seen: set[str] = set()
    for accession, form, date, document in zip(
        recent["accessionNumber"],
        recent["form"],
        recent["filingDate"],
        recent["primaryDocument"],
        strict=True,
    ):
        if form not in WANTED_FORMS or form in seen:
            continue
        seen.add(form)
        folder = accession.replace("-", "")
        found.append(
            Filing(
                accession=accession,
                form=form,
                filing_date=date,
                primary_document_url=(
                    f"https://www.sec.gov/Archives/edgar/data/{bare_cik}/{folder}/{document}"
                ),
            )
        )
        if len(seen) == len(WANTED_FORMS):
            break
    return found
