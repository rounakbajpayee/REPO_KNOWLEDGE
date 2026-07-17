# REPO_KNOWLEDGE: Roadmap & Gap Analysis
*Compiled by Google Jules*

## 1. Executive Summary
The `REPO_KNOWLEDGE` repository serves as a highly effective "Local-first semantic memory layer for codebases," designed to rescue coding agents from context-window exhaustion. It successfully implements an intricate pipeline of AST-aware chunking, dense vector embeddings via local Ollama models, and an MCP server for immediate agent consumption.

However, a deep dive into the architecture reveals a number of discrepancies between the README's stated roadmap (Deferred V2 features) and the actual codebase. Several "future" features have already been shipped silently, while a few key capabilities remain unbuilt. Additionally, the system houses powerful undocumented architectural "superpowers" that drastically improve search relevance.

## 2. README vs. Reality: Gap Analysis

### ✅ Claimed & Implemented
- **Vector DB + Relational Store**: Qdrant handles vectors; Postgres handles metadata and sync state tracking.
- **Model Engine**: Offloads embeddings locally to Ollama (`embedder.py`).
- **MCP Server**: Fully implemented in `mcp_server.py` exposing critical tools (`search_codebase`, `get_file`, `get_project_context`, etc.).
- **Decision History**: Endpoints for logging/getting decision history (`log_decision`, `get_decision_history`) work smoothly, pushing to Postgres and Markdown.

### 🚀 "Deferred (V2)" Features Actually Implemented!
The README marks several features as Deferred (V2), but they are already fully functional in the codebase:
1. **Tree-sitter AST for TypeScript/JavaScript**: `chunker.py` uses `tree-sitter-javascript` and `tree-sitter-typescript` to intelligently chunk JS/TS by classes, functions, and arrow declarations.
2. **Symbol Index**: The `search_symbols` and `get_chunks_for_file` MCP tools explicitly return function/class signatures and line boundaries without consuming context window on content bodies.
3. **OS Filewatcher**: `watcher.py` handles debounced incremental reindexing on file modifications recursively through the Projects root.

### ❌ Deferred (V2) Features Still Missing
1. **Per-file Semantic Summaries**: Not present in `chunker.py` or `knowledge.py`. Currently, it only relies on raw textual chunks and signatures.
2. **Architecture Knowledge Extraction**: Chunking exists for `.md` (via regex), but there is no specific logic to elevate or extract structured knowledge specifically from architecture documents (ADRs, PROD_SPECs).
3. **Embedding Model Benchmarking Tool**: Missing entirely from `tests/` and the application source.
4. **OS-agnostic client deploy script**: Only Windows-specific scripts (`manage.bat`, `register_startup.py`) exist outside of Docker.

### 🕵️ Hidden Superpowers (Undocumented or Under-promoted)
1. **Hybrid Search via RRF (Reciprocal Rank Fusion)**: `store.py` doesn't just do dense Qdrant searches. It queries both Qdrant (Dense Vectors) and Postgres (BM25 text search) concurrently, fusing the results via RRF. The README barely mentions this enterprise-grade hybrid retrieval pipeline.
2. **Cross-Encoder Reranking**: `reranker.py` utilizes a CPU-friendly `ms-marco-MiniLM` model via PyTorch to re-rank the candidate pool. This drastically improves precision but is only hinted at in the config variables.
3. **Antigravity Transcript Integration**: `memory_helper.py` doesn't just parse Git diffs—it traverses the hidden `~/.gemini/antigravity/brain/**/transcript.jsonl` files to automatically extract and log architectural decisions directly from agent chat logs. This is an incredible cross-repo homelab integration that deserves top billing in the README.

## 3. Architectural Recommendations & Roadmap

### A. Consolidation: Drop Qdrant for `pgvector`
**The Issue**: The system currently maintains a dual-write architecture (Qdrant for vectors, Postgres for metadata). This creates synchronization risks, complex `docker-compose` dependencies, and redundant state.
**The Fix**: Migrate vector storage directly into Postgres using the `pgvector` extension.
- **Benefit**: True ACID compliance across code chunks and vectors. BM25 and Dense Vector searches can be fused natively in SQL, entirely dropping a heavy infrastructure dependency.

### B. Implement Semantic Summaries (The "Missing" Feature)
**The Issue**: Raw chunks lack high-level reasoning and overview.
**The Fix**: Since `memory_helper.py` already interacts with Ollama chat models via `httpx`, recycle this capability into `knowledge.py` or a background worker.
- During `reindex_project`, spawn async tasks to ask a local model (e.g., `qwen2.5-coder:7b`) to summarize the file's purpose in 2 sentences. Store this in the Postgres `files` table for rapid retrieval.

### C. First-Class Architectural Knowledge Extraction
**The Issue**: ADRs are treated as standard Markdown blocks with no special weighting.
**The Fix**: Add an `Architecture` chunk type. Modify `chunk_markdown` to detect files matching `*ADR*`, `ARCHITECTURE.md`, or `PROD_SPEC.md`, parse their headers, and boost their RRF weights during `search_codebase`.

### D. Enhance the MCP Interface
- **Model Benchmarker Tool**: Implement an MCP tool `benchmark_embeddings` that takes a set of QA pairs, runs them against the current index, and reports Recall@5. This fulfills the remaining V2 deferred promise and helps agents decide whether to upgrade embeddings.
- **Diff / Recency Search**: Agents often struggle to find "what changed recently." An MCP tool that surfaces `get_recent_changes()` based on the `file_mtime` tracked in the Postgres store would drastically speed up debugging operations.
