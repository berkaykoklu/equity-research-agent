from dataclasses import dataclass
from typing import Any

# us-gaap tags for the same concept differ across filers and across years
# for the same filer (e.g. Apple's revenue tag changed with ASC 606
# adoption). Each metric lists its tags in priority order; the first tag
# present in the filing wins, and which one resolved is recorded on the
# MetricValue so a downstream reader can see it. An unmapped filer loses
# coverage on that metric visibly (via NormalizedFacts.missing) instead of
# resolving to a wrong or stale number.
TAG_PRIORITY: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_income": ("OperatingIncomeLoss",),
    "total_assets": ("Assets",),
    "total_debt": ("DebtLongtermAndShorttermCombinedAmount", "LongTermDebtNoncurrent"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
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


def _latest_annual(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    # form == "10-K" only, matching the filing-selection policy in
    # era.edgar.filings: an amendment (10-K/A) is deliberately excluded
    # rather than treated as an update to the original annual figure.
    annual = [e for e in entries if e.get("fp") == "FY" and e.get("form") == "10-K"]
    if not annual:
        return None
    return max(annual, key=lambda e: e["fy"])


def normalize_facts(companyfacts: dict[str, Any]) -> NormalizedFacts:
    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    metrics: dict[str, MetricValue] = {}

    for metric, tags in TAG_PRIORITY.items():
        for tag in tags:
            # Some us-gaap tags report both USD and shares (or other
            # units) under the same concept. Taking only the USD unit is
            # what makes a metric unambiguous downstream: the citation
            # carries tag, period, and accession but no unit field, so a
            # non-USD value here would silently corrupt a claimed figure.
            units = us_gaap.get(tag, {}).get("units", {}).get("USD")
            if not units:
                continue
            entry = _latest_annual(units)
            if entry is None:
                continue
            metrics[metric] = MetricValue(
                tag=tag,
                fiscal_period=f"FY{entry['fy']}",
                # Raw absolute USD value, intentionally left unscaled: the
                # verifier compares a claimed figure against this value at
                # several scales (e.g. a filing stating "391,035" in
                # millions against this field's 391035000000), and that
                # comparison assumes an unscaled figure here.
                value=float(entry["val"]),
                accession=entry["accn"],
            )
            break

    missing = tuple(metric for metric in TAG_PRIORITY if metric not in metrics)
    return NormalizedFacts(metrics=metrics, missing=missing)


def facts_card(facts: NormalizedFacts) -> str:
    """A few hundred tokens of headline figures, shared with every section.

    Injected into every section prompt (Task 12), so its size multiplies
    across parallel LLM calls -- kept to one line per metric plus one
    summary line for what's missing.
    """
    lines = [
        f"{name}: {value.value:,.0f} USD ({value.fiscal_period}, tag {value.tag})"
        for name, value in sorted(facts.metrics.items())
    ]
    if facts.missing:
        lines.append(f"unavailable: {', '.join(sorted(facts.missing))}")
    return "\n".join(lines)
