"""
test_cache.py — Unit tests for the Redis search result cache.

All tests mock redis.from_url so they run without a real Redis instance.
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from repo_knowledge import cache


@pytest.fixture(autouse=True)
def reset_cache_state():
    """Reset singleton connection state between tests."""
    import repo_knowledge.cache as c_mod
    original_client = c_mod._client
    original_failed = c_mod._client_failed
    yield
    c_mod._client = original_client
    c_mod._client_failed = original_failed


def _mock_redis(ping_ok: bool = True) -> MagicMock:
    r = MagicMock()
    if not ping_ok:
        r.ping.side_effect = Exception("Connection refused")
    return r


# ── cache key derivation ───────────────────────────────────────────────────────

def test_cache_key_is_deterministic():
    k1 = cache._cache_key("auth middleware", "PROJ", 5)
    k2 = cache._cache_key("auth middleware", "PROJ", 5)
    assert k1 == k2


def test_cache_key_differs_on_different_query():
    k1 = cache._cache_key("auth middleware", "PROJ", 5)
    k2 = cache._cache_key("database pool", "PROJ", 5)
    assert k1 != k2


def test_cache_key_differs_on_different_project():
    k1 = cache._cache_key("query", "PROJ_A", 5)
    k2 = cache._cache_key("query", "PROJ_B", 5)
    assert k1 != k2


def test_cache_key_differs_on_different_top_k():
    k1 = cache._cache_key("query", "PROJ", 5)
    k2 = cache._cache_key("query", "PROJ", 10)
    assert k1 != k2


def test_cache_key_has_namespace_prefix():
    k = cache._cache_key("q", None, 5)
    assert k.startswith("rk:search:")


# ── get_cached ────────────────────────────────────────────────────────────────

def test_get_cached_returns_none_when_redis_unavailable():
    with patch("repo_knowledge.cache._get_client", return_value=None):
        result = cache.get_cached("query", "PROJ", 5)
    assert result is None


def test_get_cached_returns_none_on_cache_miss():
    mock_r = _mock_redis()
    mock_r.get.return_value = None
    with patch("repo_knowledge.cache._get_client", return_value=mock_r):
        result = cache.get_cached("query", "PROJ", 5)
    assert result is None


def test_get_cached_returns_deserialized_results():
    payload = [{"content": "def foo(): pass", "score": 0.9}]
    mock_r = _mock_redis()
    mock_r.get.return_value = json.dumps(payload)
    with patch("repo_knowledge.cache._get_client", return_value=mock_r):
        result = cache.get_cached("query", "PROJ", 5)
    assert result == payload


def test_get_cached_returns_none_on_redis_error():
    mock_r = _mock_redis()
    mock_r.get.side_effect = Exception("Timeout")
    with patch("repo_knowledge.cache._get_client", return_value=mock_r):
        result = cache.get_cached("query", "PROJ", 5)
    assert result is None


# ── set_cached ────────────────────────────────────────────────────────────────

def test_set_cached_calls_setex():
    mock_r = _mock_redis()
    payload = [{"content": "x", "score": 0.8}]
    with patch("repo_knowledge.cache._get_client", return_value=mock_r):
        cache.set_cached("query", "PROJ", 5, payload, ttl=60)
    assert mock_r.setex.called
    call_args = mock_r.setex.call_args[0]
    assert call_args[1] == 60
    assert json.loads(call_args[2]) == payload


def test_set_cached_is_noop_when_redis_unavailable():
    with patch("repo_knowledge.cache._get_client", return_value=None):
        cache.set_cached("query", "PROJ", 5, [{"x": 1}])  # must not raise


def test_set_cached_is_noop_on_redis_error():
    mock_r = _mock_redis()
    mock_r.setex.side_effect = Exception("Timeout")
    with patch("repo_knowledge.cache._get_client", return_value=mock_r):
        cache.set_cached("query", "PROJ", 5, [{"x": 1}])  # must not raise


# ── is_available ──────────────────────────────────────────────────────────────

def test_is_available_true_when_connected():
    mock_r = _mock_redis()
    with patch("repo_knowledge.cache._get_client", return_value=mock_r):
        assert cache.is_available() is True


def test_is_available_false_when_not_connected():
    with patch("repo_knowledge.cache._get_client", return_value=None):
        assert cache.is_available() is False
