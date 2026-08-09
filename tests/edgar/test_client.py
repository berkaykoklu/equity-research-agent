import os
import time

import httpx
import pytest
import respx

from era.edgar.client import EdgarClient, MissingUserAgentError, UnexpectedJsonShapeError

CONTACT_USER_AGENT = "Berkay Koklu kokluberkay@gmail.com"


def test_rejects_a_user_agent_without_contact_details() -> None:
    with pytest.raises(MissingUserAgentError):
        EdgarClient(user_agent="era-bot", cache_dir=None)


@respx.mock
def test_sends_the_declared_user_agent() -> None:
    route = respx.get("https://data.sec.gov/thing.json").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=None)

    assert client.get_json("https://data.sec.gov/thing.json") == {"ok": True}
    assert route.calls.last.request.headers["user-agent"] == CONTACT_USER_AGENT


@respx.mock
def test_caches_responses_on_disk(tmp_path) -> None:
    route = respx.get("https://data.sec.gov/thing.json").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=tmp_path)

    first = client.get_json("https://data.sec.gov/thing.json")
    second = client.get_json("https://data.sec.gov/thing.json")

    assert first == {"ok": True}
    assert second == {"ok": True}
    assert route.call_count == 1


@respx.mock
def test_a_failed_cache_write_never_leaves_a_corrupt_cache_file(tmp_path, monkeypatch) -> None:
    route = respx.get("https://data.sec.gov/thing.json").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=tmp_path)

    def failing_replace(self: object, target: object) -> None:
        raise OSError("simulated crash while committing the cache write")

    monkeypatch.setattr("pathlib.Path.replace", failing_replace)

    with pytest.raises(OSError):
        client.get_json("https://data.sec.gov/thing.json")

    # The failed commit must not leave any file behind: no short/corrupt
    # cache entry, and no leaked temp file either.
    assert list(tmp_path.glob("*.cache")) == []
    assert list(tmp_path.glob("*.tmp")) == []

    monkeypatch.undo()

    # Because nothing was ever committed, the next call is a clean cache
    # miss — it re-fetches rather than being wedged on a corrupt entry.
    assert client.get_json("https://data.sec.gov/thing.json") == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_a_stale_data_sec_gov_entry_triggers_a_refetch(tmp_path) -> None:
    route = respx.get("https://data.sec.gov/thing.json").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=tmp_path)

    client.get_json("https://data.sec.gov/thing.json")
    cache_file = next(tmp_path.glob("*.cache"))
    thirty_six_hours_ago = time.time() - (36 * 60 * 60)
    os.utime(cache_file, (thirty_six_hours_ago, thirty_six_hours_ago))

    client.get_json("https://data.sec.gov/thing.json")

    assert route.call_count == 2


@respx.mock
def test_a_stale_archives_entry_does_not_trigger_a_refetch(tmp_path) -> None:
    route = respx.get("https://www.sec.gov/Archives/edgar/data/thing.htm").mock(
        return_value=httpx.Response(200, text="filing body")
    )
    client = EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=tmp_path)

    client.get_text("https://www.sec.gov/Archives/edgar/data/thing.htm")
    cache_file = next(tmp_path.glob("*.cache"))
    ten_years_ago = time.time() - (10 * 365 * 24 * 60 * 60)
    os.utime(cache_file, (ten_years_ago, ten_years_ago))

    result = client.get_text("https://www.sec.gov/Archives/edgar/data/thing.htm")

    assert result == "filing body"
    assert route.call_count == 1


@respx.mock
def test_max_age_override_forces_a_refetch(tmp_path) -> None:
    route = respx.get("https://www.sec.gov/Archives/edgar/data/thing.htm").mock(
        return_value=httpx.Response(200, text="filing body")
    )
    client = EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=tmp_path)

    client.get_text("https://www.sec.gov/Archives/edgar/data/thing.htm")
    client.get_text("https://www.sec.gov/Archives/edgar/data/thing.htm", max_age=0)

    assert route.call_count == 2


@respx.mock
def test_get_json_rejects_a_non_object_json_body() -> None:
    respx.get("https://data.sec.gov/list.json").mock(
        return_value=httpx.Response(200, json=[1, 2, 3])
    )
    client = EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=None)

    with pytest.raises(UnexpectedJsonShapeError):
        client.get_json("https://data.sec.gov/list.json")


@respx.mock
def test_retries_a_429_and_honours_retry_after(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    respx.get("https://data.sec.gov/thing.json").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "throttled"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=None)

    assert client.get_json("https://data.sec.gov/thing.json") == {"ok": True}
    assert 1.0 in sleeps


@respx.mock
def test_exhausts_retries_and_raises_on_persistent_429() -> None:
    route = respx.get("https://data.sec.gov/thing.json").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "0"}, json={})
    )
    client = EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=None)

    with pytest.raises(httpx.HTTPStatusError):
        client.get_json("https://data.sec.gov/thing.json")

    assert route.call_count == 3


def test_context_manager_closes_the_underlying_http_client() -> None:
    with EdgarClient(user_agent=CONTACT_USER_AGENT, cache_dir=None) as client:
        pass

    with pytest.raises(RuntimeError):
        client.get_text("https://data.sec.gov/thing.json")
