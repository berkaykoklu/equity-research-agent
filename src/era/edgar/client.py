import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

# SEC's fair-access policy caps clients at 10 requests/second and bans IPs
# that exceed it. 0.11s keeps us just under that ceiling with margin for
# clock jitter.
MIN_INTERVAL_SECONDS = 0.11


class MissingUserAgentError(ValueError):
    """SEC blocks requests whose User-Agent lacks contact details."""


class EdgarClient:
    """The single door SEC filing data enters through.

    Enforces the two constraints every caller would otherwise have to
    remember: a contact-bearing User-Agent (SEC rejects anonymous bots)
    and the 10 req/s rate limit. An optional on-disk cache avoids repeat
    round trips for the same URL across runs.
    """

    def __init__(self, user_agent: str, cache_dir: Path | None) -> None:
        # SEC's own guidance asks for "name email@domain" style headers and
        # blocks requests that look like an anonymous bot. Checking for "@"
        # is deliberately crude: it catches the actual mistake (no contact
        # info at all) without pretending to validate an email address.
        if "@" not in user_agent:
            raise MissingUserAgentError(
                "SEC requires a User-Agent containing a contact email address"
            )
        self._client = httpx.Client(headers={"User-Agent": user_agent}, timeout=30.0)
        self._cache_dir = cache_dir
        self._last_request_at = 0.0

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self._cache_dir / f"{digest}.cache"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_INTERVAL_SECONDS:
            time.sleep(MIN_INTERVAL_SECONDS - elapsed)
        self._last_request_at = time.monotonic()

    def get_text(self, url: str) -> str:
        path = self._cache_path(url)
        if path is not None and path.exists():
            return path.read_text()
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        if path is not None:
            path.write_text(response.text)
        return response.text

    def get_json(self, url: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.get_text(url))
        return data
