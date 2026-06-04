"""
test_store_unit.py — Unit tests for Store, all mocked (no real Qdrant).

Every test patches QdrantClient so these run in CI without infrastructure.
"""

import pytest
from unittest.mock import MagicMock, call, patch
from repo_knowledge.store import Store
from repo_knowledge.chunker import Chunk


# ── Helpers ───────────────────────────────────────────────────────────────────────────────

def _make_store(mock_client: MagicMock) -> Store:
    """Return a Store whose internal client is fully replaced by a mock."""
    with patch("repo_knowledge.store.QdrantClient", return_value=mock_client):
        store = Store(url="http://mock:6333")
    # Mark collection as ready so _ensure_collection is a no-op in tests
    store._collection_ready = True
    return store


def _make_chunk(
    project="PROJ",
    path="src/mod.py",
    content_hash="abc123",
    file_mtime=1_700_000_000.0,
) -> Chunk:
    return Chunk(
        project=project,
        path=path,
        language="python",
        chunk_type="function",
        symbol="foo",
        content="def foo(): pass",
        start_line=1,
        end_line=1,
        content_hash=content_hash,
        file_mtime=file_mtime,
    )


def _scroll_page(records, next_offset=None):
    """Return a (records, next_offset) tuple as QdrantClient.scroll would."""
    return records, next_offset


def _make_record(path: str, content_hash: str) -> MagicMock:
    rec = MagicMock()
    rec.payload = {"path": path, "content_hash": content_hash}
    return rec


# ── upsert_chunks ──────────────────────────────────────────────────────────────────────

def test_upsert_chunks_calls_upsert(tmp_path):
    mock_client = MagicMock()
    store = _make_store(mock_client)
    chunks = [_make_chunk()]
    vectors = [[0.1] * 10]
    store.upsert_chunks(chunks, vectors)
    assert mock_client.upsert.called


def test_upsert_chunks_payload_contains_content_hash():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    chunk = _make_chunk(content_hash="deadbeef")
    store.upsert_chunks([chunk], [[0.1] * 10])
    point = mock_client.upsert.call_args[1]["points"][0]
    assert point.payload["content_hash"] == "deadbeef"


def test_upsert_chunks_payload_contains_file_mtime():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    chunk = _make_chunk(file_mtime=9_999_999.0)
    store.upsert_chunks([chunk], [[0.1] * 10])
    point = mock_client.upsert.call_args[1]["points"][0]
    assert point.payload["file_mtime"] == 9_999_999.0


def test_upsert_chunks_length_mismatch_raises():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert_chunks([_make_chunk()], [])


def test_upsert_chunks_batches_large_input():
    """Inputs of >100 chunks must be split into multiple upsert calls."""
    mock_client = MagicMock()
    store = _make_store(mock_client)
    chunks = [_make_chunk() for _ in range(150)]
    vectors = [[0.1] * 10] * 150
    store.upsert_chunks(chunks, vectors)
    assert mock_client.upsert.call_count == 2  # 100 + 50


# ── delete_project ─────────────────────────────────────────────────────────────────────

def test_delete_project_calls_delete_with_filter():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    store.delete_project("PROJ")
    assert mock_client.delete.called
    selector = mock_client.delete.call_args[1]["points_selector"]
    cond = selector.filter.must[0]
    assert cond.key == "project"
    assert cond.match.value == "PROJ"


# ── delete_file ────────────────────────────────────────────────────────────────────────

def test_delete_file_calls_delete_with_project_and_path():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    store.delete_file("PROJ", "src/mod.py")
    assert mock_client.delete.called
    must_conditions = mock_client.delete.call_args[1]["points_selector"].filter.must
    keys = {c.key for c in must_conditions}
    assert "project" in keys
    assert "path" in keys


def test_delete_file_uses_correct_path_value():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    store.delete_file("PROJ", "src/auth.py")
    must_conditions = mock_client.delete.call_args[1]["points_selector"].filter.must
    path_cond = next(c for c in must_conditions if c.key == "path")
    assert path_cond.match.value == "src/auth.py"


# ── get_indexed_file_hashes ───────────────────────────────────────────────────────────────

def test_get_indexed_file_hashes_returns_path_hash_map():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    records = [
        _make_record("src/a.py", "hash_a"),
        _make_record("src/b.py", "hash_b"),
    ]
    mock_client.scroll.return_value = _scroll_page(records, next_offset=None)
    result = store.get_indexed_file_hashes("PROJ")
    assert result == {"src/a.py": "hash_a", "src/b.py": "hash_b"}


def test_get_indexed_file_hashes_old_chunks_mapped_to_empty_string():
    """Chunks without content_hash in payload must map to '' (not raise)."""
    mock_client = MagicMock()
    store = _make_store(mock_client)
    rec = MagicMock()
    rec.payload = {"path": "src/old.py"}  # no content_hash key
    mock_client.scroll.return_value = _scroll_page([rec], next_offset=None)
    result = store.get_indexed_file_hashes("PROJ")
    assert result == {"src/old.py": ""}


def test_get_indexed_file_hashes_paginates():
    """scroll must be called again when next_offset is not None."""
    mock_client = MagicMock()
    store = _make_store(mock_client)
    page1 = [_make_record("src/a.py", "hash_a")]
    page2 = [_make_record("src/b.py", "hash_b")]
    mock_client.scroll.side_effect = [
        _scroll_page(page1, next_offset="cursor_1"),
        _scroll_page(page2, next_offset=None),
    ]
    result = store.get_indexed_file_hashes("PROJ")
    assert len(result) == 2
    assert mock_client.scroll.call_count == 2


# ── search ─────────────────────────────────────────────────────────────────────────────────

def _make_hit(score: float, content_hash: str = "", path: str = "src/a.py") -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "project": "PROJ", "path": path, "symbol": "foo",
        "content": "def foo(): pass", "chunk_type": "function",
        "content_hash": content_hash,
    }
    return hit


def test_search_filters_below_threshold():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    mock_client.search.return_value = [
        _make_hit(0.80, "hash_a"),
        _make_hit(0.30, "hash_b"),  # below 0.40 threshold
    ]
    results = store.search([0.1] * 10, top_k=5)
    assert len(results) == 1
    assert results[0]["score"] == 0.8


def test_search_deduplicates_by_content_hash():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    mock_client.search.return_value = [
        _make_hit(0.90, "same_hash"),
        _make_hit(0.85, "same_hash"),  # duplicate — must be dropped
    ]
    results = store.search([0.1] * 10, top_k=5)
    assert len(results) == 1


def test_search_no_dedup_for_empty_hash():
    """Chunks with empty content_hash (old data) must all pass through."""
    mock_client = MagicMock()
    store = _make_store(mock_client)
    mock_client.search.return_value = [
        _make_hit(0.90, ""),
        _make_hit(0.85, ""),
    ]
    results = store.search([0.1] * 10, top_k=5)
    assert len(results) == 2


def test_search_fetches_2x_top_k():
    """store.search must request top_k * 2 from Qdrant to allow dedup headroom."""
    mock_client = MagicMock()
    store = _make_store(mock_client)
    mock_client.search.return_value = []
    store.search([0.1] * 10, top_k=5)
    assert mock_client.search.call_args[1]["limit"] == 10


def test_search_returns_at_most_top_k():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    # Return 8 distinct high-scoring hits
    hits = [_make_hit(0.9 - i * 0.01, f"hash_{i}", f"src/{i}.py") for i in range(8)]
    mock_client.search.return_value = hits
    results = store.search([0.1] * 10, top_k=5)
    assert len(results) == 5


def test_search_scores_are_rounded():
    mock_client = MagicMock()
    store = _make_store(mock_client)
    mock_client.search.return_value = [_make_hit(0.912345, "h1")]
    results = store.search([0.1] * 10, top_k=5)
    assert results[0]["score"] == round(0.912345, 4)
