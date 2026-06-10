"""
cache.py — Redis-backed search result cache.

Caches fully-reranked search results to avoid running the cross-encoder on
identical (query, project, top_k) triples within the TTL window.

Design choices:
  - Connection is lazily established and cached per-process.
  - All errors are silently caught — Redis is OPTIONAL infrastructure.
    A cache miss is always safe; a cache failure just adds latency.
  - Keys are SHA-256 of (query, project, top_k) so they are stable and compact.
  - Results are JSON-serialised with a configurable TTL (default 5 minutes).

Usage:
    from repo_knowledge.cache import get_cached, set_cached

    hit = get_cached(query, project, top_k)
    if hit is not None:
        return hit

    results = expensive_search(...)
    set_cached(query, project, top_k, results)
    return results
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any

from repo_knowledge.config import REDIS_TTL_S, REDIS_URL

log = logging.getLogger(__name__)

try:
    import redis  # type: ignore[import]
except ImportError:
    log.warning("Cache disabled: redis not installed. Install with: pip install repo-knowledge[cache]")


# ── Connection singleton ──────────────────────────────────────────────────────

_client: Any = None          # redis.Redis instance once connected
_client_lock = threading.Lock()
_client_failed = False       # Set True after a failed init; skips retries until restart


def _get_client() -> Any | None:
    """Return a lazily-initialised Redis client, or None if unavailable."""
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        return None

    with _client_lock:
        if _client is not None:
            return _client
        if _client_failed:
            return None
        try:
            import redis  # type: ignore[import]
            r = redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
            r.ping()  # Verify reachability
            _client = r
        except ImportError:
            _client_failed = True
            return None
        except Exception:
            _client_failed = True
            return None
    return _client


# ── Key derivation ────────────────────────────────────────────────────────────

def _cache_key(query: str, project: str | None, top_k: int) -> str:
    raw = f"{query}||{project or ''}||{top_k}"
    return "rk:search:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


# ── Public API ────────────────────────────────────────────────────────────────

def get_cached(query: str, project: str | None, top_k: int) -> list[dict] | None:
    """
    Return cached search results if present, otherwise None.

    Never raises — all Redis errors return None (cache miss).
    """
    r = _get_client()
    if r is None:
        return None
    try:
        raw = r.get(_cache_key(query, project, top_k))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        return None


def set_cached(
    query: str,
    project: str | None,
    top_k: int,
    results: list[dict],
    ttl: int = REDIS_TTL_S,
) -> None:
    """
    Store search results in Redis.  Never raises.

    Args:
        ttl: Override the global TTL for this entry (seconds).
    """
    r = _get_client()
    if r is None:
        return
    try:
        r.setex(_cache_key(query, project, top_k), ttl, json.dumps(results))
    except Exception:
        pass


def invalidate(query: str, project: str | None, top_k: int) -> None:
    """Delete a specific cache entry.  Never raises."""
    r = _get_client()
    if r is None:
        return
    try:
        r.delete(_cache_key(query, project, top_k))
    except Exception:
        pass


def flush_project(project: str) -> int:
    """
    Delete all cache entries for a project (called after reindex).

    Uses SCAN to avoid blocking Redis on large key sets.
    Returns the number of keys deleted.
    """
    r = _get_client()
    if r is None:
        return 0
    deleted = 0
    try:
        # We can't reconstruct exact keys, so scan for the namespace prefix
        # and match on project name embedded in the value (simpler: just flush prefix)
        # Since keys are hashed, we flush the entire search namespace instead.
        # For a prod system with millions of queries, a secondary index would be needed.
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="rk:search:*", count=200)
            if keys:
                r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
    except Exception:
        pass
    return deleted


def is_available() -> bool:
    """Return True if Redis is reachable."""
    return _get_client() is not None
