# ADR-003: Sentence-Transformers Reranking

**Status**: Active
**Date**: 2026-06-12

## Context
Initial semantic search via Qdrant vector retrieval can sometimes return suboptimal ordering for results due to the limitations of bi-encoder models. To improve the precision of the returned context, a secondary reranking stage is needed to compare the search query directly with the content of the retrieved chunks.

## Decision
Use `sentence-transformers` cross-encoders for the reranking stage (`src/repo_knowledge/reranker.py`). The model is configurable via the `RERANK_MODEL` environment variable and runs completely locally.

## Consequences
1. The model weights are downloaded locally on the first run, ensuring data privacy and offline capability.
2. Cross-encoder inference uses local compute (CPU or GPU, if available), which adds measurable latency to search queries.
3. On Windows, loading PyTorch/OpenMP from a non-main thread causes deadlocks. This is handled gracefully in `src/repo_knowledge/mcp_server.py` by forcing main-thread initialization.
4. The dependency is optional (`repo-knowledge[reranker]`). If missing or if the model fails to load, the application gracefully degrades by returning the initial Qdrant results without reranking.

## Alternatives Considered
- **Cohere Rerank API:** Requires external API calls, which violates the local-first design goals and introduces latency/privacy concerns.
- **No reranking:** Lower search precision, especially on complex or nuanced queries.
- **BM25 hybrid scoring:** Improves lexical search but doesn't solve deep semantic match issues as effectively as a cross-encoder.
