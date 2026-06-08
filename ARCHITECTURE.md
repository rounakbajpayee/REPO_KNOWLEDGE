# ARCHITECTURE

This document is written for agents and engineers picking up this codebase mid-stream. Read this before touching any code.

---

## Why This Exists

Coding agents were exhausting their context windows loading entire codebases. This system provides a semantic memory layer so agents can retrieve exactly the functions and classes relevant to their task — not entire files.

---

## Component Map

```
index.py (CLI) or watcher.py (Auto-watcher)
    └── KnowledgeService (knowledge.py)
            ├── Scanner (scanner.py)         — discovers Git repos
            ├── Chunker (chunker.py)         — files → chunks
            ├── Embedder (embedder.py)       — text → vectors via Ollama
            ├── PostgresStore (postgres_store.py) — Source of Truth (chunks, files, logs) + BM25 FTS
            ├── Store (store.py)             — Qdrant Vector Cache (linked by UUID) + RRF Rank Fusion
            ├── Reranker (reranker.py)       — Cross-Encoder ms-marco-MiniLM-L-6-v2 reranking stage
            └── Cache (cache.py)             — Redis-backed query result cache (fallback to memory/miss)

mcp_server.py or Web UI Dashboard (server.py)
    └── KnowledgeService                    — thin adapter/API calls
```

**Key rule:** `mcp_server.py` and `server.py` are transport adapters. They have no business logic. All logic lives in `KnowledgeService`.

---

## Component Responsibilities

### config.py
Single source of truth for all configuration. Reads from env vars with defaults. Includes PostgreSQL, Qdrant, Ollama, Redis, and Cross-Encoder configuration parameters.

### postgres_store.py
The absolute Source of Truth for the relational database schema:
- **Projects table:** Tracks scanned workspace names, tech stacks, and last indexing timestamps.
- **Files table:** Stores file paths, content hashes, and mtimes.
- **Chunks table:** Stores raw text content of chunks mapped to file IDs and primary key UUIDs (matching Qdrant point IDs).
- **Decision logs table:** Stores chronological decision logs with parameters.
- **Audit logs table:** Stores trace records asynchronously.
Exposes `search_bm25()` using PostgreSQL full-text search indexing (`tsvector`) over raw chunk content. Handles automatic DDL schema migrations, database setup, and connection pools.

### store.py
Manages Qdrant vector index transactions. Acts as a high-performance vector similarity cache. Point IDs correspond to UUIDs in PostgreSQL's `chunks` table.
Implements the hybrid search fusion logic via **Reciprocal Rank Fusion (RRF)** (`_rrf_fuse()`), promoting chunks that occur in both Qdrant similarity searches and Postgres BM25 searches.

### reranker.py
Implements the second-stage Cross-Encoder reranking using a thread-safe singleton wrapper around `ms-marco-MiniLM-L-6-v2`. Lazy-loads model weights on the first query needing reranking, with full graceful fallback if model initialization fails.

### cache.py
Manages Redis-backed cached search result sets keyed by query, project filter, and top-K. Serializes output to JSON with a configurable TTL. Gracefully bypasses cache operations if the Redis host is offline.

## High-Level Pipeline

```mermaid
flowchart TD
    Disk[(Local Disk / Projects)] -->|Reads| Scanner[Directory Scanner]
    Scanner -->|File Content| Chunker[Chunker (AST / Regex)]
    Chunker -->|Text Chunks| Postgres[(Postgres Store)]
    Postgres -->|Text| Embedder[Ollama Embedder]
    Embedder -->|Vectors| Qdrant[(Qdrant Vector DB)]
    
    Qdrant <--> MCP[MCP Server]
    Postgres <--> MCP
    MCP <--> Claude[Coding Agents]
```

1. **Scanner**: recursively reads supported files (`.py`, `.ts`, `.md`, etc.) from git repositories, honoring `.gitignore`.
2. **Chunker**: splits files into smaller `KnowledgeUnit` blocks.
   - For Python: uses AST to extract functions and classes intact.
   - For Text/Markdown: uses recursive character splitting. for markdown, fixed-line split for others.

### scanner.py
Discovers Git repositories one level deep under `PROJECTS_ROOT`. A directory is a project if it has a `.git` folder. Also does heuristic stack detection.

### chunker.py
Converts files into `Chunk` objects. Strategy by file type: python `ast` for functions/classes, regex declarations for JS/TS, header split for markdown, fixed-line split for others.

### embedder.py
`Embedder` is a Protocol. `OllamaEmbedder` is the default. Supports model swaps without parsing files again.

### knowledge.py
Pure Python. No transport dependency. Orchestrates the 2-stage search pipeline (Cache -> Embed -> Retrieve & RRF -> Rerank -> Cache), incremental indexing, and decision logging.
- `search(query, project?, top_k)` — Coordinates Redis cache check, Ollama query embedding, Qdrant + BM25 RRF recall, Cross-Encoder reranking, and cache updating.
- `reindex_project(name, force?)` — incremental checks compare local file hashes against the PostgreSQL `files` table (fast database query instead of scrolling Qdrant).
- `re_embed_all_projects()` — wipes the Qdrant collection, queries all raw text chunks from PostgreSQL, batch-embeds them using the active model, and recreates the Qdrant vector cache.
- `log_decision(...)` / `get_decision_history(...)` — writes to both Markdown vault (Obsidian compatibility) and PostgreSQL `decision_logs` table (fast query response), falling back to Markdown if Postgres is offline.

### watcher.py
Recursive directory change watcher (`watchdog`). Debounces indexing and calls `reindex_project` incrementally. Runs silently on Windows startup via registered batch script.

### Web UI Dashboard (`src/repo_knowledge/web_ui/`)
- **Backend (server.py):** FastAPI application exposing status monitors, index stats, sandbox search, reindexing, and re-embedding. Features SSE (Server-Sent Events) streaming from `audit_logs` table. Filters project documentation candidates to README, Architecture, and Runbook documents only.
- **Frontend (index.html):** SPA dashboard designed with a dark glassmorphism system (tailored HSL colors, Outfit/Space Grotesk typography) containing tabs for workspace health lights, sandbox playground, and active log console. Incorporates a slide-over project doc viewer with toggles to browse project documentation.

### tracer.py
Structured JSONL tracer asynchronously writing records to the `repo_knowledge.jsonl` file and transactionally appending audit traces to PostgreSQL `audit_logs` table via a background daemon thread.

### mcp_server.py
Exposes the tools interface to LLM client tools (`list_projects`, `get_project_context`, `search_codebase`, `get_file`, `reindex_project`, `log_decision`, `get_decision_history`, `re_embed`).

---

## Data Flow

### Indexing
```
index.py --project LENS
  → scan_projects()
  → chunk_project(LENS)
  → register_file() & upsert_chunks() → writes raw text to PostgreSQL
  → embedder.embed_batch([chunk.content, ...])
  → store.upsert_chunks() → writes points with PostgreSQL chunk UUIDs to Qdrant
```

### Search (Two-Stage Retrieval Pipeline)
```
Agent / User search_codebase("query", project="LENS")
  → 1. Redis Cache check: get_cached(query, project)
       ├── Hit  → returns cached results instantly (0.11ms)
       └── Miss → proceed to Stage 1 Recall:
                    ├── Qdrant client vector search (Dense score)
                    └── PostgreSQL search_bm25() full-text search (BM25 score)
                    └── Combine & Sort via Reciprocal Rank Fusion (RRF)
                 → proceed to Stage 2 Reranking:
                    └── ms-marco-MiniLM-L-6-v2 Cross-Encoder predict scores
                 → save results in Redis cache & return
```

### Lossless Model Swap (Re-embedding)
```
re_embed tool / Web UI
  → Qdrant collection deleted & recreated
  → Postgres: SELECT c.content, c.id FROM chunks JOIN files ...
  → embedder.embed_batch(all_contents) using new EMBEDDING_MODEL
  → Qdrant: upsert new vectors using original PostgreSQL UUIDs
```

---

## Infrastructure

- **PostgreSQL 16:** Running on Mac Mini at `192.168.0.234:5434` (database `repo_knowledge`).
- **Qdrant:** Running on Mac Mini at `192.168.0.234:6333`.
- **Ollama:** Running on Mac Mini at `192.168.0.234:11434`.
- **Redis:** Running on Mac Mini at `192.168.0.234:6379/0`.
- **Dashboard Web UI:** `http://localhost:8000`.
- **Filewatcher Startup:** Silent startup registered in Windows Startup folder (`shell:startup`).
