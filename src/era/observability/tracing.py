"""Measure what a run cost and how long it took.

The README states cost and latency per run. Those must be *measured* numbers,
not estimates copied from a pricing page -- an estimate that drifts from
reality is worse than no figure at all, because a reader has no way to tell.

Token usage is collected with LangChain's own usage callback rather than by
inspecting responses: the section subgraphs use structured output, so the
model returns a Pydantic object and the `AIMessage` carrying `usage_metadata`
never reaches the caller. The callback sees every model call in the graph,
including retries, which is exactly the total worth reporting.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# USD per million tokens. Checked against OpenAI's published rates on
# 2026-08-09; a stale entry here silently misreports the README's headline
# cost figure, so it is worth re-checking when the model changes.
PRICING: dict[str, dict[str, float]] = {
    "gpt-5.6-terra": {"input": 2.0, "cached": 0.2, "output": 12.0},
    "gpt-5.6-luna": {"input": 0.2, "cached": 0.02, "output": 1.2},
}


def cost_from_usage(usage: dict[str, int], model: str) -> float:
    """Price one model's token usage.

    An unknown model returns 0.0 rather than guessing. Reporting a plausible
    but invented number is the failure mode worth avoiding: a zero is visibly
    missing, a wrong figure is not.
    """
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    cached = usage.get("cache_read_tokens", 0)
    # Cached tokens are billed at their own rate and are *included* in
    # input_tokens by the provider, so charging both would double-count them.
    uncached_input = max(usage.get("input_tokens", 0) - cached, 0)
    return (
        uncached_input / 1e6 * rates["input"]
        + cached / 1e6 * rates["cached"]
        + usage.get("output_tokens", 0) / 1e6 * rates["output"]
    )


@dataclass
class RunRecord:
    ticker: str
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    trace_id: str | None = None


def _flatten(by_model: dict[str, Any]) -> dict[str, int]:
    """Total token counts across every model the run touched."""
    totals: dict[str, int] = {}
    for usage in by_model.values():
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def _cost_across_models(by_model: dict[str, Any]) -> float:
    """Price each model's usage at its own rates, then sum.

    A run can touch more than one model -- the cheap model chooses filing
    boundaries while the drafting model writes sections -- and charging both
    at one rate would misreport by an order of magnitude either way.
    """
    total = 0.0
    for model, usage in by_model.items():
        counts = {key: value for key, value in usage.items() if isinstance(value, int)}
        total += cost_from_usage(counts, model)
    return total


@contextmanager
def measured(ticker: str) -> Iterator[tuple[RunRecord, list[Any]]]:
    """Time a run and collect its token usage.

    Yields the record it will fill in and the callback list to hand to the
    graph. The record is only complete once the block exits.
    """
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    handler = UsageMetadataCallbackHandler()
    record = RunRecord(ticker=ticker)
    started = time.monotonic()
    try:
        yield record, [handler]
    finally:
        record.latency_seconds = time.monotonic() - started
        by_model = dict(handler.usage_metadata)
        record.tokens = _flatten(by_model)
        record.cost_usd = _cost_across_models(by_model)
