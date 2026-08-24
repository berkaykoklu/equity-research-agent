"""Offline tests for cost and latency measurement. No API calls."""

from era.observability.tracing import (
    RunRecord,
    _cost_across_models,
    _flatten,
    cost_from_usage,
    measured,
)


def test_costs_input_and_output_tokens_at_their_own_rates() -> None:
    cost = cost_from_usage({"input_tokens": 6_300, "output_tokens": 800}, model="gpt-5.6-terra")

    assert round(cost, 6) == round(6_300 / 1e6 * 2.0 + 800 / 1e6 * 12.0, 6)


def test_cached_tokens_are_not_charged_twice() -> None:
    # Providers report cached tokens *inside* input_tokens. Charging the full
    # input count and the cached count separately would overstate every run
    # that benefits from caching -- exactly the runs meant to look cheaper.
    cost = cost_from_usage(
        {"input_tokens": 6_000, "cache_read_tokens": 5_000, "output_tokens": 0},
        model="gpt-5.6-terra",
    )

    assert round(cost, 6) == round(1_000 / 1e6 * 2.0 + 5_000 / 1e6 * 0.2, 6)


def test_a_cache_count_larger_than_the_input_count_does_not_go_negative() -> None:
    cost = cost_from_usage({"input_tokens": 100, "cache_read_tokens": 5_000}, model="gpt-5.6-terra")

    assert cost >= 0.0


def test_unknown_model_costs_nothing_rather_than_guessing() -> None:
    # A zero is visibly missing; an invented figure is not.
    assert cost_from_usage({"input_tokens": 100}, model="mystery") == 0.0


def test_each_model_is_priced_at_its_own_rates() -> None:
    # A run touches the cheap model for boundary selection and the drafting
    # model for sections. Charging both at one rate misreports by ~10x.
    by_model = {
        "gpt-5.6-terra": {"input_tokens": 1_000, "output_tokens": 1_000},
        "gpt-5.6-luna": {"input_tokens": 1_000, "output_tokens": 1_000},
    }

    total = _cost_across_models(by_model)

    expected = (1_000 / 1e6 * 2.0 + 1_000 / 1e6 * 12.0) + (1_000 / 1e6 * 0.2 + 1_000 / 1e6 * 1.2)
    assert round(total, 8) == round(expected, 8)


def test_token_totals_sum_across_models() -> None:
    totals = _flatten(
        {
            "a": {"input_tokens": 10, "output_tokens": 1},
            "b": {"input_tokens": 5, "output_tokens": 2},
        }
    )

    assert totals == {"input_tokens": 15, "output_tokens": 3}


def test_non_integer_usage_fields_are_ignored() -> None:
    # Providers include nested breakdowns alongside the counts; summing those
    # would raise rather than report.
    totals = _flatten({"a": {"input_tokens": 10, "input_token_details": {"cache_read": 5}}})

    assert totals == {"input_tokens": 10}


def test_measured_records_latency_even_when_the_block_raises() -> None:
    # A failed run still cost time and money; losing the measurement because
    # the graph raised would hide exactly the runs worth investigating.
    record: RunRecord | None = None
    try:
        with measured("AAPL") as (rec, _callbacks):
            record = rec
            raise RuntimeError("graph blew up")
    except RuntimeError:
        pass

    assert record is not None
    assert record.ticker == "AAPL"
    assert record.latency_seconds >= 0.0


def test_measured_yields_a_callback_list_for_the_graph() -> None:
    with measured("AAPL") as (record, callbacks):
        assert callbacks, "no usage callback was provided"

    assert record.cost_usd == 0.0  # nothing ran, so nothing was spent
    assert record.tokens == {}
