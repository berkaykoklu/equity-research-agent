from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

# us-gaap tags for the same concept differ across filers and across years
# for the same filer (e.g. Apple's revenue tag changed with ASC 606
# adoption). Each metric lists its tags in priority order; every tag is
# resolved and the candidate with the newest reported period wins, with
# list order only breaking a genuine tie between two tags reporting the
# same period (see _resolve_metric) -- a higher-priority tag that has gone
# stale must not shadow a lower-priority tag with live, current data.
# Which tag actually won is recorded on the MetricValue so a downstream
# reader can see it. An unmapped filer loses coverage on that metric
# visibly (via NormalizedFacts.missing) instead of resolving to a wrong or
# stale number.
PeriodKind = Literal["duration", "instant"]


@dataclass(frozen=True)
class MetricSpec:
    # "duration" facts (revenue, income, cash flow) cover a span and carry
    # both start and end; "instant" facts (assets, debt) are a snapshot
    # and carry only end. Selecting the right entry requires knowing which
    # kind a metric is -- see _select_duration / _select_instant.
    period: PeriodKind
    tags: tuple[str, ...]


METRICS: dict[str, MetricSpec] = {
    "revenue": MetricSpec(
        period="duration",
        tags=(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ),
    ),
    "net_income": MetricSpec(period="duration", tags=("NetIncomeLoss", "ProfitLoss")),
    "operating_income": MetricSpec(period="duration", tags=("OperatingIncomeLoss",)),
    "total_assets": MetricSpec(period="instant", tags=("Assets",)),
    "total_debt": MetricSpec(
        period="instant",
        tags=("DebtLongtermAndShorttermCombinedAmount", "LongTermDebtNoncurrent"),
    ),
    "operating_cash_flow": MetricSpec(
        period="duration", tags=("NetCashProvidedByUsedInOperatingActivities",)
    ),
}


@dataclass(frozen=True)
class MetricValue:
    tag: str
    fiscal_period: str
    value: float
    accession: str


@dataclass(frozen=True)
class NormalizedFacts:
    metrics: dict[str, MetricValue]
    missing: tuple[str, ...]


def _usable(entry: dict[str, Any]) -> bool:
    """True if entry carries the minimum fields needed to build a MetricValue.

    SEC's XBRL data occasionally has a partial row -- an accession without
    a tagged value, or a JSON `null` where a number is expected. One bad
    row must not abort every metric in the payload: it gets skipped here
    so the metric can resolve from a different entry, or -- if there is no
    other candidate -- fall through to `missing`, which is exactly the
    visible-degradation behaviour this module exists to guarantee.
    """
    accn = entry.get("accn")
    val = entry.get("val")
    return isinstance(accn, str) and bool(accn) and isinstance(val, int | float)


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _is_annual_span(entry: dict[str, Any]) -> bool:
    """True if start->end covers roughly a year (350-380 days).

    fy/fp/form on a companyfacts entry describe the *filing* the fact
    appeared in, not the period it covers: a single FY2024 10-K emits its
    whole three-year comparative income statement, so every one of those
    entries carries fy=2024, fp="FY", form="10-K" -- indistinguishable
    without looking at start/end. This is the actual signal for "this
    entry is the full fiscal year," not the label fields.
    """
    start, end = _parse_date(entry.get("start")), _parse_date(entry.get("end"))
    if start is None or end is None:
        return False
    return 350 <= (end - start).days <= 380


def _sort_key(entry: dict[str, Any]) -> tuple[str, str]:
    # ISO "YYYY-MM-DD" strings compare chronologically; filed breaks a tie
    # on identical end (e.g. a same-period restatement) so the later,
    # corrected filing wins over the original.
    return (str(entry.get("end", "")), str(entry.get("filed", "")))


def _select_duration(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    # form == "10-K" only, matching the filing-selection policy in
    # era.edgar.filings: an amendment (10-K/A) is deliberately excluded
    # rather than treated as an update to the original annual figure.
    candidates = [e for e in entries if _usable(e) and e.get("form") == "10-K"]
    annual = [e for e in candidates if _is_annual_span(e)]
    if not annual:
        # A payload that never carries start/end can't be verified by date
        # math (this happens with minimal or malformed input). Fall back
        # to the filing's own FY label so a single, unambiguous entry can
        # still resolve, at the cost of losing the duration-span guard.
        annual = [e for e in candidates if e.get("fp") == "FY"]
    if not annual:
        return None
    return max(annual, key=_sort_key)


def _select_instant(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [e for e in entries if _usable(e) and e.get("form") == "10-K"]
    instants = [e for e in candidates if e.get("start") is None and _parse_date(e.get("end"))]
    if not instants:
        instants = [e for e in candidates if e.get("fp") == "FY"]
    if not instants:
        return None
    return max(instants, key=_sort_key)


def _fiscal_period(entry: dict[str, Any]) -> str:
    end = _parse_date(entry.get("end"))
    if end is not None:
        return f"FY{end.year}"
    fy = entry.get("fy")
    return f"FY{fy}" if isinstance(fy, int) else "FY?"


def _resolve_metric(us_gaap: dict[str, Any], spec: MetricSpec) -> MetricValue | None:
    select = _select_duration if spec.period == "duration" else _select_instant
    candidates: list[tuple[str, dict[str, Any]]] = []

    for tag in spec.tags:
        units = us_gaap.get(tag, {}).get("units", {}).get("USD")
        if not units:
            continue
        entry = select(units)
        if entry is not None:
            candidates.append((tag, entry))

    if not candidates:
        return None

    def rank(candidate: tuple[str, dict[str, Any]]) -> tuple[str, str, int]:
        tag, entry = candidate
        end, filed = _sort_key(entry)
        # Negative index: an earlier (higher-priority) tag wins only when
        # end and filed are identical -- a genuine tie, not a preference.
        return (end, filed, -spec.tags.index(tag))

    best_tag, best_entry = max(candidates, key=rank)
    return MetricValue(
        tag=best_tag,
        fiscal_period=_fiscal_period(best_entry),
        # Raw absolute USD value, intentionally left unscaled: the
        # verifier compares a claimed figure against this value at
        # several scales (e.g. a filing stating "391,035" in millions
        # against this field's 391035000000), and that comparison assumes
        # an unscaled figure here.
        value=float(best_entry["val"]),
        accession=str(best_entry["accn"]),
    )


def normalize_facts(companyfacts: dict[str, Any]) -> NormalizedFacts:
    # Some us-gaap tags report both USD and shares (or other units) under
    # the same concept. Taking only the USD unit is what makes a metric
    # unambiguous downstream: the citation carries tag, period, and
    # accession but no unit field, so a non-USD value here would silently
    # corrupt a claimed figure.
    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    metrics: dict[str, MetricValue] = {}

    for name, spec in METRICS.items():
        value = _resolve_metric(us_gaap, spec)
        if value is not None:
            metrics[name] = value

    missing = tuple(name for name in METRICS if name not in metrics)
    return NormalizedFacts(metrics=metrics, missing=missing)


def facts_card(facts: NormalizedFacts) -> str:
    """A few hundred tokens of headline figures, shared with every section.

    Injected into every section prompt (Task 12), so its size multiplies
    across parallel LLM calls -- kept to a header, one line per metric,
    and one summary line for what's missing.
    """
    lines = ["Financial facts (from SEC XBRL filings):"]
    lines += [
        f"{name}: {value.value:,.0f} USD ({value.fiscal_period}, tag {value.tag})"
        for name, value in sorted(facts.metrics.items())
    ]
    if facts.missing:
        lines.append(f"unavailable: {', '.join(sorted(facts.missing))}")
    return "\n".join(lines)
