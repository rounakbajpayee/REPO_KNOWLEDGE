# ARCHITECTURE

This document is written for agents and engineers picking up this codebase mid-stream. Read this before touching any code.

---

## Why This Exists

Coding agents were exhausting their context windows loading entire codebases. This system provides a semantic search layer so agents can retrieve exactly the functions and classes relevant to their task — not entire files.

---

## Component Map

```
index.py (CLI) or watcher.py (Auto-watcher)
    └── KnowledgeService (knowledge.py)
            ├── Scanner (scanner.py)        — discovers Git repos
            ├── Chunker (chunker.py)        — files → indexed chunks
            ├── Embedder (embedder.py)      — text → vectors via Ollama
            └── Store (store.py)            — Qdrant read/write

mcp_server.py or memory_helper.py (Memory post-mortem)
    └── KnowledgeService                   — thin adapter/API calls

```

**Key rule:** `mcp_server.py` is a transport adapter. It has no business logic. All logic lives in `KnowledgeService`. Future transports (REST, gRPC) call `KnowledgeService` directly.

---

## Component Responsibilities

### config.py
Single source of truth for all configuration. Reads from env vars with defaults. No other module hardcodes URLs or constants.

### scanner.py
Discovers Git repositories one level deep under `PROJECTS_ROOT`. A directory is a project if it has a `.git` folder. Also does heuristic stack detection (checks for `pyproject.toml`, `package.json`, etc.).

### chunker.py
Converts files into `Chunk` objects. Strategy by file type:

| File type | Strategy | Rationale |
|-----------|----------|-----------|
| `.py` | stdlib `ast` module — one chunk per top-level function/class. Imports prepended to each chunk. | AST is stdlib, handles decorators/async correctly |
| `.ts/.tsx/.js/.jsx` | Regex boundary split on function/class/arrow declarations | tree-sitter deferred to V2 |
| `.md` | Split on `##` headers | Sections are the natural unit |
| Everything else | 60-line fixed split, 10-line overlap | Safe fallback |

**Known limitation:** JS/TS regex chunker will misfire on some edge cases (default exports, HOCs, etc.). Acceptable for MVP. V2 replaces with tree-sitter.

### embedder.py
`Embedder` is a Protocol (structural typing). `OllamaEmbedder` is the only implementation for now. To swap models:
1. Change `EMBEDDING_MODEL` and `EMBEDDING_DIM` in `.env`
2. Change `QDRANT_COLLECTION` to a new model-slug name
3. Reindex

Never change the collection name without changing the model — dimensions must match.

### store.py
All Qdrant operations. The `project` field in every chunk payload is a filterable keyword field — this is load-bearing for `delete_project()` used during reindexing. Do not remove it.

Collection is created on first run if missing. Uses cosine distance.

### knowledge.py
Pure Python. No MCP dependency. Public API:
- `list_projects(trace_id?)` — scanner + store combined view (TTL cached for 30s)
- `get_project_context(name, trace_id?)` — README + tree + stack in one call
- `search(query, project?, top_k?, trace_id?)` — embed query → Qdrant search (deduplicated by content hash, similarity threshold applied, search quality classified)
- `get_file(project, path, trace_id?)` — raw file read
- `reindex_project(name, force?, trace_id?)` — incremental reindexing by default, only chunking/embedding changed/new files; deletes stale/removed files. Set `force=True` to wipe and fully rebuild.
- `log_decision(topic, name, description, rationale, options_considered?, trace_id?)` — appends a structured, timestamped decision log inside `knowledge_vault/<topic>.md` with YAML frontmatter.
- `get_decision_history(topic, limit?, full_history?, trace_id?)` — returns the chronological decision entries for a topic, defaulting to returning only the last `limit` (3) entries to save context window tokens.

### watcher.py
A persistent background service utilizing Python's `watchdog` library (binds to native OS-push APIs like Windows `ReadDirectoryChangesW` or macOS `FSEvents`). Monitors `PROJECTS_ROOT` recursively for file saves, filters events by supported extensions and ignore directories, and triggers a debounced incremental reindex (5-second idle timer) for the affected project.

### memory_helper.py
A CLI post-mortem decision recovery script. Queries local Ollama chat models (like Qwen Coder or DeepSeek R1) to analyze workspace git diffs, recent commits, or local client logs, structures the technical choices into a structured decision payload, and logs them to the vault. Implements safety guardrails by using compressed diff representation (`-U1`) and aggressively truncating payloads to 8,000 characters to prevent context window overload.

### tracer.py
Structured JSONL tracer carrying timestamp, trace ID, event, severity, subsystem, duration, and payload. Writes asynchronously in the background. All lines logged during a single MCP tool call share the same `trace_id`.

### mcp_server.py
Thin MCP adapter over `KnowledgeService`. Uses `stdio` transport (MCP default). Handles:
- Tool listing (exposing search, project context, and decision vault logging/history retrieval tools)
- Argument validation
- `RuntimeError` from embedder/store → clean `{"error": "..."}` to agent
- Startup health checks (warns but doesn't block)


---

## Data Flow

### Indexing
```
index.py --project LENS
  → scan_projects() → finds LENS at ~/Projects/LENS
  → chunk_project(LENS) → list[Chunk]
  → embedder.embed_batch([chunk.content, ...]) → list[vector]
  → store.delete_project("LENS")   ← clean slate first
  → store.upsert_chunks(chunks, vectors)
```

### Search
```
Agent: search_codebase("how does auth work", project="LENS")
  → embedder.embed("how does auth work") → vector
  → store.search(vector, top_k=5, project="LENS")
  → returns [{path, symbol, content, score}, ...]
```

---

## Payload Schema (Qdrant)

Every vector point stores:

```json
{
  "project":         "LENS",
  "path":            "src/ocr/service.py",
  "language":        "python",
  "chunk_type":      "function",
  "symbol":          "process_image",
  "content":         "...",
  "start_line":      42,
  "end_line":        78,
  "content_hash":    "sha256...",
  "file_mtime":      1700000000.0,
  "embedding_model": "nomic-embed-text",
  "indexed_at":      "2026-01-01T00:00:00+00:00"
}
```

`embedding_model` is stored per-chunk to support future benchmarking across models.

---

## Benchmarking Design (V2, not yet built)

To benchmark two embedding models side by side:
1. Set `EMBEDDING_MODEL=model-a`, `QDRANT_COLLECTION=code_chunks_model_a` → index
2. Set `EMBEDDING_MODEL=model-b`, `QDRANT_COLLECTION=code_chunks_model_b` → index
3. Run identical queries against both collections, compare scores

No schema migration needed — collections are independent.

---

## Infrastructure

- **Qdrant:** `http://100.70.3.86:6333` (Mac Mini via Tailscale)
- **Ollama:** `http://100.70.3.86:11434` (Mac Mini via Tailscale)
- **Default model:** `nomic-embed-text` (768-dim) — swap to `qwen3-embed` (1024-dim) when pulled
- **Python:** 3.12

---

## Deferred Work (do not implement without a spec)

| Item | Reason deferred |
|------|----------------|
| tree-sitter for JS/TS | Native dependency, not worth it for MVP |
| Per-file summaries | Adds indexing cost, design needed |
| Symbol index | V2 feature, separate collection |
| Architecture knowledge extraction (README, ADR) | V2 feature |
| Benchmarking tool UI | Design deferred, data model already supports it |
| OS-agnostic client deploy script | After client deploy spec |


---

## Adding a New Transport (e.g. REST API)

Do NOT add logic to `mcp_server.py`.

1. Create `src/repo_knowledge/api_server.py` (FastAPI or whatever)
2. Import and instantiate `KnowledgeService`
3. Map HTTP endpoints to service methods
4. Done — all logic is already in `knowledge.py`
