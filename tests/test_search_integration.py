import time
import logging
from unittest.mock import MagicMock, patch
from repo_knowledge.knowledge import KnowledgeService
from repo_knowledge.config import QDRANT_URL, OLLAMA_URL

logging.basicConfig(level=logging.INFO)

@patch("repo_knowledge.knowledge.Store")
@patch("repo_knowledge.knowledge.default_embedder")
def test_search_integration(mock_default_embedder, mock_store_class):
    print(f"QDRANT_URL: {QDRANT_URL}")
    print(f"OLLAMA_URL: {OLLAMA_URL}")

    # Set up mocks
    mock_store = MagicMock()
    mock_store_class.return_value = mock_store

    mock_embedder = MagicMock()
    mock_default_embedder.return_value = mock_embedder

    # Mock embedder response
    mock_embedder.embed.return_value = [0.1] * 1024

    # Mock store search response
    mock_store.search.return_value = [
        {"path": "src/knowledge.py", "score": 0.95},
        {"path": "src/store.py", "score": 0.88},
        {"path": "src/embedder.py", "score": 0.76},
    ]

    print("Initializing...")
    t0 = time.time()

    # Optional dependencies caching and reranking check
    with patch("repo_knowledge.knowledge.search_cache.get_cached", return_value=None):
        with patch("repo_knowledge.knowledge.search_cache.set_cached"):
            with patch("repo_knowledge.knowledge.search_reranker.rerank", side_effect=lambda q, c, top_k: c[:top_k]):
                svc = KnowledgeService(store=mock_store, embedder=mock_embedder)
                print(f"Init took {time.time() - t0:.2f}s")

                print("Searching...")
                t1 = time.time()
                res = svc.search("knowledge service embedding vector store", project="REPO_KNOWLEDGE", top_k=3)
                print(f"Search took {time.time() - t1:.2f}s")
                print(f"Found {len(res)} results.")
                for r in res:
                    print(f" - {r.get('path')} (Score: {r.get('score')})")

                assert len(res) == 3
                assert mock_embedder.embed.called
                assert mock_store.search.called
