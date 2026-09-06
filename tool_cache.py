"""Simple in-process TTL cache for tool results (news, MCP, etc.)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

# Default: 30 minutes (within the 15–60 minute band for jornada/news freshness).
DEFAULT_TTL_SECONDS = 30 * 60


class TtlCache:
    """Thread-safe dict cache with per-entry expiry."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if now >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, *, ttl_seconds: float | None = None) -> None:
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = time.monotonic() + max(0.0, ttl)
        with self._lock:
            self._store[key] = (expires_at, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        now = time.monotonic()
        with self._lock:
            alive = {key: item for key, item in self._store.items() if item[0] > now}
            self._store = alive
            return len(self._store)


def cache_key(prefix: str, payload: Any) -> str:
    """Stable cache key from a JSON-serializable payload."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


# Shared caches used by news helpers and MCP tool calls.
news_cache = TtlCache()
mcp_cache = TtlCache()
