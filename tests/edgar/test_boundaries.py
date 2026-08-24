from pathlib import Path

import pytest

from era.edgar.boundaries import (
    MAX_CANDIDATES,
    Boundary,
    BoundarySelectionError,
    CachedBoundarySelector,
    HeadingCandidate,
    LlmBoundarySelector,
    _BoundaryResponse,
    _ItemBoundary,
    build_candidates,
)


def test_build_candidates_finds_every_item_number_not_just_the_wanted_three() -> None:
    # Every item number is a candidate, including ones this parser never
    # extracts -- Item 1A's real end anchor is often Item 1B, 2 or 3, not
    # Item 7 directly, and a filter that dropped those would leave the
    # selector with no way to bound 1A anywhere except at Item 7 itself,
    # silently absorbing everything in between (the bug this test guards).
    text = (
        "Item 1. Business\n"
        "We design things.\n"
        "Item 1A. Risk Factors\n"
        "Stuff.\n"
        "Item 1B. Unresolved Staff Comments\n"
        "None.\n"
        "Item 2. Properties\n"
        "More.\n"
        "Item 7. MD&A\n"
        "More stuff.\n"
        "Item 7A. Market Risk\n"
        "Some.\n"
        "Item 8. Financial Statements\n"
        "Done."
    )

    candidates = build_candidates(text)

    assert [c.item for c in candidates] == ["1", "1A", "1B", "2", "7", "7A", "8"]
    assert [c.index for c in candidates] == list(range(7))


def test_build_candidates_caps_at_max_candidates() -> None:
    # A safety valve, not a tuning knob: a pathological document repeating
    # "item" without bound must not turn into an unbounded prompt.
    text = "Item 1. Something\n" * (MAX_CANDIDATES + 50)

    candidates = build_candidates(text)

    assert len(candidates) == MAX_CANDIDATES
    assert [c.index for c in candidates] == list(range(MAX_CANDIDATES))


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


@pytest.mark.parametrize(
    "contents",
    [
        "not json at all {{{",
        '{"1": {"start_index": 0}}',  # missing end_index key
        '{"1": {"start_index": "zero", "end_index": null}}',  # wrong type
        '{"1": "not even an object"}',
        "[]",  # valid JSON, wrong top-level shape
    ],
)
def test_cached_boundary_selector_treats_a_corrupt_cache_file_as_a_miss(
    tmp_path: Path, contents: str
) -> None:
    # A cache file can be damaged by anything -- a crash mid-write from a
    # version of this code that didn't yet have the atomic-write discipline,
    # a hand edit, a future incompatible format. None of that should ever
    # reach _resolve_offset as a live TypeError; it should look exactly like
    # a cache miss and get recomputed.
    candidates = build_candidates("Item 1. Business\nStuff.")
    cached = CachedBoundarySelector(_CountingSelector({}), tmp_path)
    path = cached._cache_path(candidates)
    path.write_text(contents, encoding="utf-8")

    inner = _CountingSelector({"1": Boundary(start_index=0, end_index=None)})
    result = CachedBoundarySelector(inner, tmp_path).select(candidates)

    assert inner.calls == 1
    assert result == {"1": Boundary(start_index=0, end_index=None)}
    # The bad file is overwritten with a fresh, valid one, so a third read
    # doesn't need to recompute again.
    third_inner = _CountingSelector({})
    CachedBoundarySelector(third_inner, tmp_path).select(candidates)
    assert third_inner.calls == 0


def test_cached_boundary_selector_never_persists_a_selector_failure(tmp_path: Path) -> None:
    candidates = build_candidates("Item 1. Business\nStuff.")
    failing = _RaisingSelector()
    cached = CachedBoundarySelector(failing, tmp_path)

    with pytest.raises(BoundarySelectionError):
        cached.select(candidates)

    # No cache file was written for a failed decision...
    assert cached._cache_path(candidates).exists() is False
    # ...so a second attempt asks the inner selector again rather than
    # trusting a cached failure forever.
    with pytest.raises(BoundarySelectionError):
        cached.select(candidates)
    assert failing.calls == 2


class _RaisingSelector:
    def __init__(self) -> None:
        self.calls = 0

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        self.calls += 1
        raise BoundarySelectionError("simulated transient failure")


class _CountingSelector:
    """Records how many times select() actually ran the inner decision."""

    def __init__(self, result: dict[str, Boundary]) -> None:
        self._result = result
        self.calls = 0

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        self.calls += 1
        return self._result


class _FakeStructuredRunnable:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.invocations: list[object] = []

    def invoke(self, prompt: object) -> object:
        self.invocations.append(prompt)
        if self._error is not None:
            raise self._error
        return self._response


class _FakeChatModel:
    """Stands in for a BaseChatModel: offline, no network, no API key needed."""

    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self._runnable = _FakeStructuredRunnable(response, error)

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


def test_llm_boundary_selector_raises_on_an_unexpected_response_shape() -> None:
    # with_structured_output can be configured to hand back a raw dict
    # instead of the pydantic model; this project never does that, but if a
    # future change did, that's an integration bug, not a fact about the
    # document -- it must raise, not silently look like a genuine "no real
    # section" decision that CachedBoundarySelector would then cache forever.
    model = _FakeChatModel(response={"item_1": {"start_index": 0}})
    selector = LlmBoundarySelector(model)
    candidates = build_candidates("Item 1. Business\nStuff.")

    with pytest.raises(BoundarySelectionError):
        selector.select(candidates)


def test_llm_boundary_selector_raises_when_the_model_call_itself_fails() -> None:
    # A timeout, a rate limit, an auth failure surviving build_model's
    # retries -- none of these are a fact about the document. This is the
    # case CachedBoundarySelector relies on never being swallowed into {}.
    model = _FakeChatModel(error=TimeoutError("simulated network timeout"))
    selector = LlmBoundarySelector(model)
    candidates = build_candidates("Item 1. Business\nStuff.")

    with pytest.raises(BoundarySelectionError):
        selector.select(candidates)


def test_llm_boundary_selector_defaults_to_the_cheap_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Boundary selection is an integers-only task -- there is never a reason
    # for it to default to build_model's own default, which is the
    # expensive drafting model. Omitting the model argument entirely must
    # not silently multiply the cost of every filing's decision.
    import era.edgar.boundaries as boundaries_module
    from era.graph.models import CHEAP_MODEL

    calls: list[str] = []

    def _fake_build_model(provider: str = "openai", model_name: str = "", **_: object) -> object:
        calls.append(model_name)
        return _FakeChatModel(response=_BoundaryResponse())

    monkeypatch.setattr(boundaries_module, "build_model", _fake_build_model)

    LlmBoundarySelector()

    assert calls == [CHEAP_MODEL]


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


# --- guards the final review found untested --------------------------------


def test_the_candidate_list_is_capped() -> None:
    # MAX_CANDIDATES is the module's defence against an unbounded prompt: a
    # pathological document could otherwise send thousands of candidate lines
    # to the model. Raising the cap survived the whole suite -- and made it
    # 7x slower, which is the cost this guard exists to prevent.
    from era.edgar.boundaries import build_candidates

    text = "\n".join(f"Item {n}. Heading" for n in range(5_000))

    # A hard literal, not MAX_CANDIDATES. Asserting against the constant under
    # test moves the goalposts with it: raising the cap to ten million would
    # satisfy `<= MAX_CANDIDATES` and the guard would be gone with every test
    # still green.
    assert len(build_candidates(text)) <= 1_000


def test_the_cache_key_changes_when_the_decision_version_changes(tmp_path: Path) -> None:
    # _DECISION_VERSION exists so that improving the prompt invalidates cached
    # decisions. Without it in the key, re-running a validation against a warm
    # cache measures the OLD prompt and publishes the result as evidence the
    # change worked -- which nearly happened once already.
    from era.edgar import boundaries

    candidates = build_candidates("Item 1. Business\nsome body text here\n")
    selector = boundaries.CachedBoundarySelector(_CountingSelector({}), tmp_path)

    first = selector._cache_path(candidates)
    original = boundaries._DECISION_VERSION
    try:
        boundaries._DECISION_VERSION = original + "-next"
        second = selector._cache_path(candidates)
    finally:
        boundaries._DECISION_VERSION = original

    assert first != second, "a prompt-version bump must not reuse cached decisions"
