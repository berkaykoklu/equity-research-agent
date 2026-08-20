from pathlib import Path

import pytest

from era.edgar.boundaries import (
    Boundary,
    CachedBoundarySelector,
    HeadingCandidate,
    LlmBoundarySelector,
    _BoundaryResponse,
    _ItemBoundary,
    build_candidates,
)


def test_build_candidates_finds_every_wanted_item_and_the_item_8_anchor() -> None:
    text = (
        "Item 1. Business\n"
        "We design things.\n"
        "Item 1A. Risk Factors\n"
        "Stuff.\n"
        "Item 7. MD&A\n"
        "More stuff.\n"
        "Item 8. Financial Statements\n"
        "Done."
    )

    candidates = build_candidates(text)

    assert [c.item for c in candidates] == ["1", "1A", "7", "8"]
    assert [c.index for c in candidates] == [0, 1, 2, 3]


def test_build_candidates_excludes_items_that_are_neither_wanted_nor_the_anchor() -> None:
    # Item 8 is kept as a boundary anchor for Item 7 (see boundaries.py), but
    # nothing else needs it -- Item 2 has no bearing on any of the three
    # sections this parser extracts, so it should never even reach the model.
    text = "Item 1. Business\nStuff.\nItem 2. Properties\nMore.\nItem 7. MD&A\nDone."

    candidates = build_candidates(text)

    assert [c.item for c in candidates] == ["1", "7"]


def test_build_candidates_ignores_a_heading_shaped_mid_sentence_mention() -> None:
    # Anchored to the start of a line: "...as discussed in Item 7 above"
    # sitting mid-paragraph must never produce a candidate at all.
    text = "Item 1. Business\nFor detail, see Item 7 above for our discussion."

    candidates = build_candidates(text)

    assert [c.item for c in candidates] == ["1"]


def test_build_candidates_accepts_a_heading_with_no_delimiter() -> None:
    text = "ITEM 1A RISK FACTORS\nSome text."

    candidates = build_candidates(text)

    assert [c.item for c in candidates] == ["1A"]


def test_build_candidates_context_is_a_short_single_line_window() -> None:
    long_line = "Item 1. Business\n" + ("word " * 100)

    candidates = build_candidates(long_line)

    assert len(candidates[0].context) <= 140
    assert "\n" not in candidates[0].context


def test_cached_boundary_selector_only_calls_the_inner_selector_once(tmp_path: Path) -> None:
    candidates = build_candidates("Item 1. Business\nStuff.\nItem 8. Financial Statements\nDone.")
    inner = _CountingSelector({"1": Boundary(0, None)})
    cached = CachedBoundarySelector(inner, tmp_path)

    first = cached.select(candidates)
    second = cached.select(candidates)

    assert inner.calls == 1
    assert first == second == {"1": Boundary(0, None)}


def test_cached_boundary_selector_recomputes_for_a_different_document(tmp_path: Path) -> None:
    # The cache key is item+offset pairs, not the surrounding text, so the
    # two documents must actually produce different candidate lists (not
    # just different prose around the same single offset) to prove this.
    inner = _CountingSelector({"1": Boundary(0, None)})
    cached = CachedBoundarySelector(inner, tmp_path)

    cached.select(build_candidates("Item 1. Business\nStuff."))
    cached.select(build_candidates("Item 1. Business\nStuff.\nItem 1A. Risk Factors\nMore."))

    assert inner.calls == 2


def test_cached_boundary_selector_persists_across_instances(tmp_path: Path) -> None:
    candidates = build_candidates("Item 1. Business\nStuff.")
    inner = _CountingSelector({"1": Boundary(0, None)})

    CachedBoundarySelector(inner, tmp_path).select(candidates)
    # A fresh instance pointed at the same directory should read the cache
    # file rather than re-asking the (fresh, zero-call-count) inner selector.
    second_inner = _CountingSelector({"1": Boundary(0, None)})
    result = CachedBoundarySelector(second_inner, tmp_path).select(candidates)

    assert second_inner.calls == 0
    assert result == {"1": Boundary(0, None)}


def test_cached_boundary_selector_round_trips_a_null_end_index(tmp_path: Path) -> None:
    candidates = build_candidates("Item 1. Business\nStuff.")
    inner = _CountingSelector({"1": Boundary(start_index=0, end_index=None)})
    cached = CachedBoundarySelector(inner, tmp_path)

    cached.select(candidates)
    result = CachedBoundarySelector(_CountingSelector({}), tmp_path).select(candidates)

    assert result == {"1": Boundary(start_index=0, end_index=None)}


class _CountingSelector:
    """Records how many times select() actually ran the inner decision."""

    def __init__(self, result: dict[str, Boundary]) -> None:
        self._result = result
        self.calls = 0

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        self.calls += 1
        return self._result


class _FakeStructuredRunnable:
    def __init__(self, response: object) -> None:
        self._response = response
        self.invocations: list[object] = []

    def invoke(self, prompt: object) -> object:
        self.invocations.append(prompt)
        return self._response


class _FakeChatModel:
    """Stands in for a BaseChatModel: offline, no network, no API key needed."""

    def __init__(self, response: object) -> None:
        self._runnable = _FakeStructuredRunnable(response)

    def with_structured_output(self, schema: object) -> _FakeStructuredRunnable:
        return self._runnable


def test_llm_boundary_selector_maps_the_structured_response_to_boundaries() -> None:
    response = _BoundaryResponse(
        item_1=_ItemBoundary(start_index=0, end_index=3),
        item_1a=None,
        item_7=_ItemBoundary(start_index=5, end_index=None),
    )
    model = _FakeChatModel(response)
    selector = LlmBoundarySelector(model)
    candidates = build_candidates("Item 1. Business\nStuff.")

    chosen = selector.select(candidates)

    assert chosen == {
        "1": Boundary(start_index=0, end_index=3),
        "7": Boundary(start_index=5, end_index=None),
    }
    assert "1A" not in chosen


def test_llm_boundary_selector_returns_empty_without_calling_the_model_for_no_candidates() -> None:
    model = _FakeChatModel(response=_BoundaryResponse())
    selector = LlmBoundarySelector(model)

    chosen = selector.select([])

    assert chosen == {}
    assert model._runnable.invocations == []


def test_llm_boundary_selector_fails_closed_on_an_unexpected_response_shape() -> None:
    # with_structured_output can be configured to hand back a raw dict
    # instead of the pydantic model; this project never does that, but if a
    # future change did, every item should end up missing, not crash.
    model = _FakeChatModel(response={"item_1": {"start_index": 0}})
    selector = LlmBoundarySelector(model)
    candidates = build_candidates("Item 1. Business\nStuff.")

    chosen = selector.select(candidates)

    assert chosen == {}


@pytest.mark.parametrize("provider", ["openai"])
def test_build_model_default_returns_a_chat_model(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    # Construction alone must never touch the network -- only .invoke() does
    # -- so a fake key is enough to prove build_model wires up the right
    # class without spending anything or requiring a real credential.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    from langchain_openai import ChatOpenAI

    from era.graph.models import CHEAP_MODEL, REQUEST_TIMEOUT_SECONDS, build_model

    model = build_model(provider=provider, model_name=CHEAP_MODEL)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == CHEAP_MODEL
    assert model.request_timeout == REQUEST_TIMEOUT_SECONDS


def test_build_model_rejects_an_unknown_provider() -> None:
    from era.graph.models import build_model

    with pytest.raises(ValueError, match="unknown provider"):
        build_model(provider="not-a-real-provider")
