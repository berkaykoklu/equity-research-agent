import hashlib
import json
import threading
import time
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import httpx

# SEC's fair-access policy caps clients at 10 requests/second and bans IPs
# that exceed it. 0.11s keeps us just under that ceiling with margin for
# clock jitter.
MIN_INTERVAL_SECONDS = 0.11

# data.sec.gov serves live indexes — submissions and companyfacts JSON —
# that grow every time a company files something new. Caching them forever
# would silently pin a company's filing list to whatever it looked like on
# first fetch, so we treat entries from this host as stale after a day.
# www.sec.gov/Archives serves the filing documents themselves: SEC never
# edits a published filing, so those are safe to cache indefinitely (the
# default below, `None`, when the host isn't data.sec.gov).
DATA_API_HOST = "data.sec.gov"
DATA_API_MAX_AGE_SECONDS = 24 * 60 * 60

# SEC's own throttle response (429) and transient 503s are worth a couple of
# quiet retries rather than failing the whole run; anything else surfaces
# immediately.
RETRYABLE_STATUS_CODES = {429, 503}
MAX_ATTEMPTS = 3


class MissingUserAgentError(ValueError):
    """SEC blocks requests whose User-Agent lacks contact details."""


class UnexpectedJsonShapeError(ValueError):
    """Raised when a response body is valid JSON but not a JSON object."""


class _UseHostDefault:
    """Sentinel meaning 'no max_age override was given, use the host default'."""


_USE_HOST_DEFAULT = _UseHostDefault()

type MaxAge = float | None | _UseHostDefault


def _default_max_age(url: str) -> float | None:
    if urlparse(url).netloc == DATA_API_HOST:
        return DATA_API_MAX_AGE_SECONDS
    return None


def _retry_wait_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return 2.0**attempt


class EdgarClient:
    """The single door SEC filing data enters through.

    Enforces the two constraints every caller would otherwise have to
    remember: a contact-bearing User-Agent (SEC rejects anonymous bots)
    and the 10 req/s rate limit. An optional on-disk cache avoids repeat
    round trips for the same URL across runs.

    The throttle is guarded by a lock, so it is safe to share one instance
    across threads — callers should do that rather than creating one
    EdgarClient per thread, or they will each think they have their own
    10 req/s budget and collectively exceed SEC's limit.
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
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_at = 0.0
        self._throttle_lock = threading.Lock()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self._cache_dir / f"{digest}.cache"

    def _resolve_max_age(self, url: str, max_age: MaxAge) -> float | None:
        if isinstance(max_age, _UseHostDefault):
            return _default_max_age(url)
        return max_age

    def _cache_is_fresh(self, path: Path, max_age: float | None) -> bool:
        if not path.exists():
            return False
        if max_age is None:
            return True
        age_seconds = time.time() - path.stat().st_mtime
        return age_seconds <= max_age

    def _write_cache(self, path: Path, text: str) -> None:
        # Write to a uniquely-named temp file in the same directory, then
        # rename it onto the real path. os.replace is atomic on the same
        # filesystem, so a crash, Ctrl-C, or full disk mid-write can only
        # ever leave the temp file damaged — the real cache path either
        # keeps its previous (complete) content or never gets created.
        # Without this, a truncated write left a short file that looked
        # like a valid cache hit forever.
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _throttle(self) -> None:
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < MIN_INTERVAL_SECONDS:
                time.sleep(MIN_INTERVAL_SECONDS - elapsed)
            self._last_request_at = time.monotonic()

    def _get_with_retry(self, url: str) -> httpx.Response:
        for attempt in range(MAX_ATTEMPTS):
            response = self._client.get(url)
            if response.status_code not in RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(_retry_wait_seconds(response, attempt))
        response.raise_for_status()
        return response

    def get_text(self, url: str, max_age: MaxAge = _USE_HOST_DEFAULT) -> str:
        resolved_max_age = self._resolve_max_age(url, max_age)
        path = self._cache_path(url)
        if path is not None and self._cache_is_fresh(path, resolved_max_age):
            return path.read_text(encoding="utf-8")
        self._throttle()
        response = self._get_with_retry(url)
        if path is not None:
            self._write_cache(path, response.text)
        return response.text

    def get_json(self, url: str, max_age: MaxAge = _USE_HOST_DEFAULT) -> dict[str, Any]:
        text = self.get_text(url, max_age=max_age)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise UnexpectedJsonShapeError(
                f"Expected a JSON object from {url}, got {type(data).__name__}"
            )
        return data
