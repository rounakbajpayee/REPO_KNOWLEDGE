"""
test_store_unit.py — Unit tests for Store, all mocked (no real Postgres).

Every test patches PostgresStore so these run in CI without infrastructure.
"""

from unittest.mock import MagicMock

import pytest

from repo_knowledge.chunker import Chunk
from repo_knowledge.store import Store

# ── Helpers ───────────────────────────────────────────────────────────────────────────────


def _make_store(mock_pg: MagicMock) -> Store:
    """Return a Store whose internal pg is fully replaced by a mock."""
    store = Store(postgres_store=mock_pg)
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


# ── upsert_chunks ──────────────────────────────────────────────────────────────────────


def test_upsert_chunks_calls_upsert():
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    chunks = [_make_chunk()]
    vectors = [[0.1] * 10]

    mock_pg.upsert_project.return_value = 1
    mock_pg.register_file.return_value = 1

    store.upsert_chunks(chunks, vectors)

    assert mock_pg.upsert_project.called
    assert mock_pg.register_file.called
    assert mock_pg.upsert_chunks.called


def test_upsert_chunks_length_mismatch_raises():
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert_chunks([_make_chunk()], [])


# ── delete_project ─────────────────────────────────────────────────────────────────────


def test_delete_project_calls_delete():
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    store.delete_project("PROJ")
    mock_pg.delete_project.assert_called_once_with("PROJ")


# ── delete_file ────────────────────────────────────────────────────────────────────────


def test_delete_file_calls_delete():
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    store.delete_file("PROJ", "src/mod.py")
    mock_pg.delete_file.assert_called_once_with("PROJ", "src/mod.py")


# ── get_indexed_file_hashes ───────────────────────────────────────────────────────────────


def test_get_indexed_file_hashes_returns_path_hash_map():
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    mock_hash_map = {"src/a.py": "hash_a", "src/b.py": "hash_b"}
    store._pg.get_indexed_file_hashes.return_value = mock_hash_map
    result = store.get_indexed_file_hashes("PROJ")
    assert result == mock_hash_map
    store._pg.get_indexed_file_hashes.assert_called_once_with("PROJ")


def test_get_indexed_file_hashes_empty_when_no_files():
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    store._pg.get_indexed_file_hashes.return_value = {}
    result = store.get_indexed_file_hashes("PROJ")
    assert result == {}
    store._pg.get_indexed_file_hashes.assert_called_once_with("PROJ")


# ── search ─────────────────────────────────────────────────────────────────────────────────


def _make_hit(score: float, content_hash: str = "", path: str = "src/a.py") -> dict:
    return {
        "score": score,
        "project": "PROJ",
        "path": path,
        "symbol": "foo",
        "content": "def foo(): pass",
        "chunk_type": "function",
        "content_hash": content_hash,
    }


def test_search_filters_below_threshold():
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    mock_pg.search_vector.return_value = [
        _make_hit(0.80, "hash_a"),
        _make_hit(0.30, "hash_b"),  # below threshold
    ]
    results = store.search([0.1] * 10, top_k=5, score_threshold=0.40)
    assert len(results) == 1
    assert results[0]["score"] == 0.8


def test_search_deduplicates_by_content_hash():
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    mock_pg.search_vector.return_value = [
        _make_hit(0.90, "same_hash"),
        _make_hit(0.85, "same_hash"),  # duplicate — must be dropped
    ]
    results = store.search([0.1] * 10, top_k=5)
    assert len(results) == 1


def test_search_no_dedup_for_empty_hash():
    """Chunks with empty content_hash (old data) must all pass through."""
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    mock_pg.search_vector.return_value = [
        _make_hit(0.90, ""),
        _make_hit(0.85, ""),
    ]
    results = store.search([0.1] * 10, top_k=5)
    assert len(results) == 2


def test_search_fetches_2x_top_k():
    """store.search must request top_k * 2 from pg to allow dedup headroom."""
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    mock_pg.search_vector.return_value = []
    store.search([0.1] * 10, top_k=5)
    assert mock_pg.search_vector.call_args[1]["limit"] == 10


def test_search_returns_at_most_top_k():
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    # Return 8 distinct high-scoring hits
    hits = [_make_hit(0.9 - i * 0.01, f"hash_{i}", f"src/{i}.py") for i in range(8)]
    mock_pg.search_vector.return_value = hits
    results = store.search([0.1] * 10, top_k=5)
    assert len(results) == 5


# ── list_projects ──────────────────────────────────────────────────────────────────────


def test_list_projects_uses_postgres_primary():
    """list_projects must query Postgres first."""
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    store._pg.get_project_names.return_value = ["ALPHA", "BETA"]

    result = store.list_projects()

    assert result == ["ALPHA", "BETA"]
    mock_pg.get_project_names.assert_called_once()


def test_list_projects_empty_on_postgres_error():
    """list_projects must return empty on exception."""
    mock_pg = MagicMock()
    store = _make_store(mock_pg)
    store._pg.get_project_names.side_effect = Exception("DB offline")

    result = store.list_projects()

    assert result == []
    mock_pg.get_project_names.assert_called_once()
