import httpx
import pytest
import respx

from era.edgar.client import EdgarClient, MissingUserAgentError


def test_rejects_a_user_agent_without_contact_details() -> None:
    with pytest.raises(MissingUserAgentError):
        EdgarClient(user_agent="era-bot", cache_dir=None)


@respx.mock
def test_sends_the_declared_user_agent() -> None:
    route = respx.get("https://data.sec.gov/thing.json").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = EdgarClient(user_agent="Berkay Koklu kokluberkay@gmail.com", cache_dir=None)

    assert client.get_json("https://data.sec.gov/thing.json") == {"ok": True}
    assert route.calls.last.request.headers["user-agent"] == ("Berkay Koklu kokluberkay@gmail.com")


@respx.mock
def test_caches_responses_on_disk(tmp_path) -> None:
    route = respx.get("https://data.sec.gov/thing.json").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = EdgarClient(user_agent="Berkay Koklu kokluberkay@gmail.com", cache_dir=tmp_path)

    client.get_json("https://data.sec.gov/thing.json")
    client.get_json("https://data.sec.gov/thing.json")

    assert route.call_count == 1
