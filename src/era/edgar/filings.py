from dataclasses import dataclass

from era.edgar.client import DATA_API_MAX_AGE_SECONDS, EdgarClient

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# company_tickers.json lives on www.sec.gov, which EdgarClient caches
# indefinitely by default under the assumption that www.sec.gov serves
# immutable, already-published documents. This file breaks that
# assumption: it's a live index SEC updates as issuers list, delist, and
# recycle tickers. A forever-cached copy would mean a newly listed ticker
# never resolves and a recycled one resolves to the wrong CIK -- silently
# pointing every downstream accession at the wrong issuer. So it gets the
# same 24-hour treatment as the data.sec.gov endpoints instead of the
# host's default.
TICKERS_MAX_AGE_SECONDS = DATA_API_MAX_AGE_SECONDS

WANTED_FORMS = ("10-K", "10-Q")

# The 10-K is the only filing required for a research note: Items 1, 1A
# and 7 all come from it. Without one there is nothing to research, so its
# absence is an error rather than an empty result. The 10-Q is supporting
# material and stays optional.
REQUIRED_FORMS = ("10-K",)


class UnknownTickerError(LookupError):
    """The ticker is not present in SEC's company index."""


class MissingFilingError(LookupError):
    """A required filing form is absent from the issuer's submission history."""


@dataclass(frozen=True)
class Filing:
    accession: str
    form: str
    filing_date: str
    primary_document_url: str


def _normalize_ticker(ticker: str) -> str:
    # SEC's index spells share classes with a dash ("BRK-B"), but users
    # (and the CLI, which takes raw user input) commonly type the dot
    # form ("BRK.B") instead. Normalize whitespace, case, and the dot
    # convention to the index's own spelling before comparing.
    return ticker.strip().upper().replace(".", "-")


def resolve_cik(client: EdgarClient, ticker: str) -> str:
    index = client.get_json(TICKERS_URL, max_age=TICKERS_MAX_AGE_SECONDS)
    wanted = _normalize_ticker(ticker)
    for entry in index.values():
        # The index is SEC-maintained but not schema-guaranteed per entry.
        # A single malformed row (missing or non-string "ticker") must not
        # abort resolution for every ticker that would have matched later
        # in the file, so we skip it rather than let a KeyError propagate.
        if not isinstance(entry, dict):
            continue
        entry_ticker = entry.get("ticker")
        if not isinstance(entry_ticker, str):
            continue
        if entry_ticker.upper() == wanted:
            return str(entry["cik_str"]).zfill(10)
    raise UnknownTickerError(f"{wanted} is not a US-listed issuer in SEC's index")


def latest_filings(client: EdgarClient, cik: str) -> list[Filing]:
    data = client.get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = data["filings"]["recent"]

    # SEC's archive URL layout mixes two different CIK/accession spellings.
    # The folder segment is the accession number with its dashes stripped.
    # The CIK segment conventionally drops the submissions endpoint's
    # zero-padding too -- www.sec.gov/Archives also resolves the padded
    # form, so this isn't a hard requirement, just the form EDGAR itself
    # uses when it links to a filing.
    bare_cik = cik.lstrip("0")

    best_by_form: dict[str, Filing] = {}
    for accession, form, date, document in zip(
        recent["accessionNumber"],
        recent["form"],
        recent["filingDate"],
        recent["primaryDocument"],
        strict=True,
    ):
        # Exact match only. This deliberately excludes amendments and
        # transition reports (10-K/A, 10-KT, 10-Q/A, 10-QT): we want the
        # original, not-yet-superseded periodic report. A company whose
        # only annual filing is a 10-K/A will surface as "missing 10-K"
        # (see REQUIRED_FORMS below) rather than silently substituting
        # the amendment for it.
        if form not in WANTED_FORMS:
            continue

        # SEC emits an empty primaryDocument on some entries (observed on
        # older and paper-derived filings). The would-be URL is just the
        # accession's folder with nothing after it, which resolves to a
        # real, HTTP 200 EDGAR directory-listing page rather than a 404 --
        # so this must be rejected here, not left for the downstream
        # fetch to fail on.
        if not document:
            continue

        # `recent`'s arrays are reverse-chronological in practice, but
        # that ordering is not documented by SEC, so treating "first
        # entry wins" as reliable would be correct by convention rather
        # than by construction. Comparing filing_date explicitly (ISO
        # "YYYY-MM-DD", so string comparison is chronological) is correct
        # regardless of the input order.
        current_best = best_by_form.get(form)
        if current_best is None or date > current_best.filing_date:
            folder = accession.replace("-", "")
            best_by_form[form] = Filing(
                accession=accession,
                form=form,
                filing_date=date,
                primary_document_url=(
                    f"https://www.sec.gov/Archives/edgar/data/{bare_cik}/{folder}/{document}"
                ),
            )

    for required_form in REQUIRED_FORMS:
        if required_form not in best_by_form:
            raise MissingFilingError(f"CIK {cik} has no {required_form} in its recent filings")

    return [best_by_form[form] for form in WANTED_FORMS if form in best_by_form]
