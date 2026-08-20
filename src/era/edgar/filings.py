from dataclasses import dataclass
from typing import Any

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

# A restructuring can transfer a well-known ticker to a newly formed holding
# company (SEC's tell is form 8-K12B, the successor-issuer report) that has
# filed quarterlies but has not yet reached its first annual report. That is
# a legitimate, if confusing, state -- not a bug -- so the error just needs
# to name the entity and show what it does file rather than try to trace the
# succession back to a predecessor, which would be speculative.
MAX_FORMS_IN_ERROR_MESSAGE = 8


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
            raise MissingFilingError(
                _missing_filing_message(data, cik, required_form, recent["form"])
            )

    return [best_by_form[form] for form in WANTED_FORMS if form in best_by_form]


def _missing_filing_message(
    data: dict[str, Any], cik: str, required_form: str, forms: list[Any]
) -> str:
    name = data.get("name")
    entity = f"{name} (CIK {cik})" if isinstance(name, str) and name else f"CIK {cik}"

    # `forms` comes straight out of a JSON payload typed dict[str, Any], so
    # nothing guarantees its elements are strings. This function only runs on
    # the already-confusing "no annual report" path -- crashing here would
    # replace a clear message with a traceback at exactly the moment the
    # reader is relying on the message to explain what went wrong.
    distinct_forms = sorted({form for form in forms if isinstance(form, str)})
    forms_on_file = ", ".join(distinct_forms[:MAX_FORMS_IN_ERROR_MESSAGE]) or "none"

    return f"{entity} has filed no {required_form}. Forms on file: {forms_on_file}."
