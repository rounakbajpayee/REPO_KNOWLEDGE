"""
test_reranker.py — Unit tests for the cross-encoder reranking module.

All tests mock out sentence_transformers so they run without the model download.
"""

from unittest.mock import MagicMock, patch

import pytest

from repo_knowledge import reranker


def _make_chunks(n: int, base_score: float = 0.8) -> list[dict]:
    return [
        {
            "content": f"def func_{i}(): pass",
            "project": "PROJ",
            "path": f"src/mod_{i}.py",
            "chunk_type": "function",
            "symbol": f"func_{i}",
            "score": round(base_score - i * 0.01, 4),
            "content_hash": f"hash_{i}",
        }
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def reset_reranker_state():
    """Reset singleton state between tests."""
    import repo_knowledge.reranker as r_mod

    original_model = r_mod._model
    original_failed = r_mod._model_failed
    yield
    r_mod._model = original_model
    r_mod._model_failed = original_failed


def test_rerank_returns_top_k():
    """rerank() must return at most top_k results."""
    mock_model = MagicMock()
    import numpy as np

    mock_model.predict.return_value = np.array([0.9, 0.5, 0.7, 0.3, 0.8])

    with patch("repo_knowledge.reranker._load_model", return_value=mock_model):
        chunks = _make_chunks(5)
        result = reranker.rerank("auth middleware", chunks, top_k=3)

    assert len(result) == 3


def test_rerank_orders_by_score_descending():
    """rerank() must return results sorted by rerank_score descending."""
    mock_model = MagicMock()
    import numpy as np

    # Scores: chunk0=0.1, chunk1=0.9, chunk2=0.5
    mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])

    with patch("repo_knowledge.reranker._load_model", return_value=mock_model):
        chunks = _make_chunks(3)
        result = reranker.rerank("query", chunks, top_k=3)

    assert result[0]["rerank_score"] == 0.9
    assert result[1]["rerank_score"] == 0.5
    assert result[2]["rerank_score"] == 0.1


def test_rerank_sets_score_field():
    """rerank() must set 'score' == 'rerank_score' for API compatibility."""
    mock_model = MagicMock()
    import numpy as np

    mock_model.predict.return_value = np.array([0.75])

    with patch("repo_knowledge.reranker._load_model", return_value=mock_model):
        result = reranker.rerank("query", _make_chunks(1), top_k=1)

    assert result[0]["score"] == result[0]["rerank_score"]


def test_rerank_degrades_gracefully_when_model_unavailable():
    """rerank() must return original order (truncated) if model not available."""
    with patch("repo_knowledge.reranker._load_model", return_value=None):
        chunks = _make_chunks(10)
        result = reranker.rerank("query", chunks, top_k=4)

    assert len(result) == 4
    # Original order preserved (chunk 0 first)
    assert result[0]["symbol"] == "func_0"


def test_rerank_empty_input_returns_empty():
    """rerank() must return empty list for empty input."""
    with patch("repo_knowledge.reranker._load_model", return_value=None):
        assert reranker.rerank("query", [], top_k=5) == []


def test_is_available_true_when_model_loads():
    mock_model = MagicMock()
    with patch("repo_knowledge.reranker._load_model", return_value=mock_model):
        assert reranker.is_available() is True


def test_is_available_false_when_model_absent():
    with patch("repo_knowledge.reranker._load_model", return_value=None):
        assert reranker.is_available() is False
