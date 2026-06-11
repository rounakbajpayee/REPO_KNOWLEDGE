"""
test_rrf.py — Unit tests for Reciprocal Rank Fusion and the hybrid store search.

Tests _rrf_fuse() directly and verifies Store.search() calls BM25 when
query_text is provided and handles BM25 failures gracefully.
"""

from unittest.mock import MagicMock, patch

from repo_knowledge.store import Store, _rrf_fuse

# ── _rrf_fuse unit tests ───────────────────────────────────────────────────────


def test_rrf_fuse_returns_sorted_descending():
    all_ids = {"a", "b", "c"}
    qdrant = {"a": {"_qdrant_rank": 0}, "b": {"_qdrant_rank": 1}, "c": {"_qdrant_rank": 2}}
    bm25 = {"a": {"_bm25_rank": 2}, "b": {"_bm25_rank": 0}, "c": {"_bm25_rank": 1}}

    result = _rrf_fuse(all_ids, qdrant, bm25)

    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_chunk_in_both_lists_scores_higher_than_single_list():
    """A chunk in both Qdrant and BM25 at rank-0 must outscore one only in Qdrant."""
    qdrant = {"dual": {"_qdrant_rank": 0}, "solo": {"_qdrant_rank": 1}}
    bm25 = {"dual": {"_bm25_rank": 0}}
    all_ids = {"dual", "solo"}

    result = dict(_rrf_fuse(all_ids, qdrant, bm25))

    assert result["dual"] > result["solo"]


def test_rrf_fuse_empty_inputs():
    result = _rrf_fuse(set(), {}, {})
    assert result == []


def test_rrf_fuse_qdrant_only():
    """When BM25 is empty every chunk receives the BM25 penalty rank."""
    qdrant = {"a": {"_qdrant_rank": 0}, "b": {"_qdrant_rank": 1}}
    result = _rrf_fuse({"a", "b"}, qdrant, {})
    ids = [cid for cid, _ in result]
    # "a" was rank-0 in Qdrant, so it should beat "b" rank-1
    assert ids[0] == "a"


def test_rrf_fuse_k_parameter_affects_scores():
    """Higher k smooths the score differences between ranks."""
    qdrant = {"a": {"_qdrant_rank": 0}, "b": {"_qdrant_rank": 1}}
    r_k60 = dict(_rrf_fuse({"a", "b"}, qdrant, {}, k=60))
    r_k1 = dict(_rrf_fuse({"a", "b"}, qdrant, {}, k=1))
    # With k=1 the gap between rank-0 and rank-1 is larger
    gap_k1 = r_k1["a"] - r_k1["b"]
    gap_k60 = r_k60["a"] - r_k60["b"]
    assert gap_k1 > gap_k60


# ── Store.search integration (mocked Qdrant + Postgres) ───────────────────────


def _make_store():
    mock_client = MagicMock()
    mock_pg = MagicMock()
    with patch("repo_knowledge.store.QdrantClient", return_value=mock_client):
        store = Store(url="http://mock:6333", postgres_store=mock_pg)
    store._collection_ready = True
    return store, mock_client, mock_pg


def _qdrant_hit(chunk_id: str, score: float, content: str = "def foo(): pass") -> MagicMock:
    hit = MagicMock()
    hit.id = chunk_id
    hit.score = score
    hit.payload = {
        "project": "PROJ",
        "path": "src/a.py",
        "content": content,
        "chunk_type": "function",
        "symbol": "foo",
        "content_hash": f"hash_{chunk_id}",
        "file_mtime": 0.0,
    }
    return hit


def test_search_calls_bm25_when_query_text_provided():
    store, mock_client, mock_pg = _make_store()
    mock_client.search.return_value = [_qdrant_hit("abc", 0.85)]
    mock_pg.search_bm25.return_value = []

    store.search([0.1] * 10, top_k=5, query_text="auth middleware")

    mock_pg.search_bm25.assert_called_once()
    call_kwargs = mock_pg.search_bm25.call_args
    assert (
        "auth middleware" in call_kwargs[0] or call_kwargs[1].get("query_text") == "auth middleware"
    )


def test_search_skips_bm25_when_no_query_text():
    store, mock_client, mock_pg = _make_store()
    mock_client.search.return_value = [_qdrant_hit("abc", 0.85)]

    store.search([0.1] * 10, top_k=5)

    mock_pg.search_bm25.assert_not_called()


def test_search_degrades_gracefully_when_bm25_raises():
    store, mock_client, mock_pg = _make_store()
    mock_client.search.return_value = [_qdrant_hit("abc", 0.85)]
    mock_pg.search_bm25.side_effect = Exception("DB offline")

    # Must not raise — returns Qdrant-only results
    result = store.search([0.1] * 10, top_k=5, query_text="query")
    assert len(result) >= 1


def test_search_rrf_promotes_dual_list_chunks():
    """A chunk in both Qdrant and BM25 must appear before one only in Qdrant."""
    store, mock_client, mock_pg = _make_store()

    # Qdrant returns chunk_a (rank 0) and chunk_b (rank 1)
    mock_client.search.return_value = [
        _qdrant_hit("chunk_a", 0.90),
        _qdrant_hit("chunk_b", 0.75),
    ]
    # BM25 only returns chunk_b (rank 0) — makes it a dual-list hit
    mock_pg.search_bm25.return_value = [
        {
            "id": "chunk_b",
            "project": "PROJ",
            "path": "src/b.py",
            "content": "def bar(): pass",
            "chunk_type": "function",
            "symbol": "bar",
            "start_line": 1,
            "end_line": 5,
            "language": "python",
            "content_hash": "hash_chunk_b",
            "bm25_score": 1.0,
            "_bm25_rank": 0,
        },
    ]

    result = store.search([0.1] * 10, top_k=5, query_text="bar")

    ids = [r.get("content_hash") for r in result]
    # chunk_b is in both lists — RRF should promote it above chunk_a
    assert ids[0] == "hash_chunk_b"


def test_search_deduplicates_by_content_hash():
    """Chunks with the same content_hash appearing in both lists must be deduplicated."""
    store, mock_client, mock_pg = _make_store()

    mock_client.search.return_value = [_qdrant_hit("id1", 0.90)]
    mock_pg.search_bm25.return_value = [
        {
            "id": "id1",
            "project": "PROJ",
            "path": "src/a.py",
            "content": "def foo(): pass",
            "chunk_type": "function",
            "symbol": "foo",
            "start_line": 1,
            "end_line": 2,
            "language": "python",
            "content_hash": "hash_id1",
            "bm25_score": 1.0,
            "_bm25_rank": 0,
        },
    ]

    result = store.search([0.1] * 10, top_k=5, query_text="foo")
    hashes = [r.get("content_hash") for r in result]
    assert hashes.count("hash_id1") == 1
