# ADR-002: Redis Caching Layer

**Status**: Active
**Date**: 2026-06-12

## Context
Running the cross-encoder reranker during search (`src/repo_knowledge/reranker.py`) is computationally expensive and introduces latency. We need a way to cache fully-reranked search results to avoid re-running the reranker on identical `(query, project, top_k)` triples within a short time window (TTL window).

## Decision
Introduce Redis as an OPTIONAL infrastructure component for caching search results (`src/repo_knowledge/cache.py`). Keys are derived from the SHA-256 hash of the `(query, project, top_k)` tuple.

## Consequences
Redis allows sharing the cache across multiple worker processes. The cache is entirely optional:
1. If Redis is not installed (missing from `repo-knowledge[cache]`), the application degrades gracefully and disables caching.
2. Connection issues or errors during cache gets/sets are silently ignored.
3. A cache miss is completely safe, simply incurring the latency of rerunning the cross-encoder.
4. Requires running a local or remote Redis instance to enable the feature.

## Alternatives Considered
- **In-process LRU cache:** Cannot easily share results across multiple processes or restart bounds without complexity.
- **SQLite-backed cache:** Introduces disk I/O latency and potential locking issues in concurrent scenarios.
- **No cache:** Unacceptable latency when identical queries are made sequentially.
