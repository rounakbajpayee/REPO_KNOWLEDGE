# REPO_KNOWLEDGE

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-success.svg)](https://github.com)


Local-first semantic memory layer for codebases. Indexes Git repositories and exposes search capabilities via MCP to coding agents (Claude, Codex, OpenCode, etc.).

**Why this exists:** coding agents were dying mid-PR trying to load entire codebases into context. This system lets agents search for exactly the functions and classes they need, without reading files they don't.

---

## Architecture

```
Projects Folder → Scanner → Chunker → Embedder → Qdrant
                                                      ↕
                        MCP Server ← KnowledgeService
                              ↕
                    Claude / Codex / OpenCode
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for component details and design decisions.

---

## Setup

```bash
# 1. Clone and create venv
git clone https://github.com/rounakbajpayee/REPO_KNOWLEDGE
cd REPO_KNOWLEDGE
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — set PROJECTS_ROOT, confirm QDRANT_URL, OLLAMA_URL, and POSTGRES_HOST / credentials

# 4. Pull the embedding model in Ollama (on Mac)
ollama pull nomic-embed-text
# or: ollama pull qwen3-embed (when available) — update EMBEDDING_MODEL + EMBEDDING_DIM in .env

# 5. Index your projects
python index.py                    # index everything under PROJECTS_ROOT
python index.py --project LENS     # index a single project

# 6. Run the OS Filewatcher (Runs in background on Windows login)
# Register silent watcher startup on Windows:
python register_startup.py
# Or run manually in foreground:
python watcher.py                  # auto-reindexes on file saves with 5s debounce

# 7. Start the Web UI Dashboard Monitor (Optional)
python -m repo_knowledge.web_ui.server
# Open http://localhost:8000 in your browser to monitor indexing/traces

# 8. Post-Mortem Decision Extraction (Optional)
python memory_helper.py --diff     # reconstructs decisions from workspace git diffs

### Docker Compose (Recommended)
You can run the entire service (Postgres, Qdrant, Web UI, Watcher) in Docker.
```bash
docker-compose up -d
```

---

## Running the MCP Server

```bash
python -m repo_knowledge.mcp_server
```

Add to your MCP client config (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "repo-knowledge": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "repo_knowledge.mcp_server"],
      "cwd": "/path/to/REPO_KNOWLEDGE/src"
    }
  }
}
```

---

## MCP Tools Reference

### `list_projects`
Returns all discovered Git repositories with stack and index status.
- No input required
- Call first to orient before working on any project

### `get_project_context`
Single-call cold start for an agent on a project.
- Input: `project` (string)
- Returns: README excerpt, directory tree, file count, stack, index status
- Call this before starting or continuing work — avoids manual file reads

### `search_codebase`
Semantic search over indexed chunks.
- Input: `query` (string), optional `project` (string), optional `top_k` (int, default 5)
- Returns: ranked chunks with path, symbol, line range, content, score
- Use this instead of reading entire files

### `get_file`
Read raw file contents.
- Input: `project` (string), `path` (string, relative to project root)
- Use `search_codebase` first to find the path, then fetch if you need the full file

### `reindex_project`
Reindexes a project. Performs an incremental reindex by default, only chunking/embedding new and changed files, and deleting removed files.
- Input: `project` (string), optional `force` (boolean, default false)
- Run after significant code changes. Pass `force: true` to perform a full clean reindex.

### `log_decision`
Log an architectural, configuration, or dependency choice. Agents are instructed to call this immediately when a decision is made.
- Input: `topic` (string), `name` (string), `description` (string), `rationale` (string), optional `options_considered` (list of objects)
- Saves decision notes to `knowledge_vault/<topic>.md`.

### `get_decision_history`
Retrieve the chronological list of decisions logged under a topic.
- Input: `topic` (string), optional `limit` (int, default 3), optional `full_history` (boolean, default false)
- Defaults to returning the last 3 entries to preserve the agent's context window.

### `re_embed`
Wipe Qdrant vector index and re-embed all code chunks from the PostgreSQL store using the current embedding model. Excellent for lossless model swapping.
- No input required

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://100.70.3.86:6333` | Qdrant via Tailscale |
| `OLLAMA_URL` | `http://100.70.3.86:11434` | Ollama via Tailscale |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model name |
| `EMBEDDING_DIM` | `768` | Must match model (nomic=768, qwen3=1024) |
| `QDRANT_COLLECTION` | `code_chunks_nomic` | Include model slug for benchmarking |
| `PROJECTS_ROOT` | `~/Projects` | Root directory to scan |
| `SEARCH_TOP_K` | `5` | Default search results count |
| `SEARCH_SCORE_THRESHOLD` | `0.40` | Min similarity score to include in search |
| `OLLAMA_TIMEOUT` | `120.0` | Timeout in seconds for Ollama calls |
| `RERANK_ENABLED` | `true` | Set to false to disable Cross-Encoder reranking |
| `RERANK_FETCH_K` | `40` | Candidate pool size retrieved from hybrid recall |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU-friendly cross-encoder model name |
| `REDIS_URL` | `redis://192.168.0.234:6379/0` | Optional Redis cache server connection string |
| `REDIS_TTL_S` | `300` | Redis cached search result time-to-live in seconds |

---

## Switching Embedding Models

1. Pull the new model in Ollama: `ollama pull <model>`
2. Update `.env`: `EMBEDDING_MODEL`, `EMBEDDING_DIM`, and `QDRANT_COLLECTION` (new slug name)
3. Re-embed losslessly:
   - Click **Lossless Re-embed Vector Cache** on the Dashboard Web UI
   - Or run the MCP tool: `re_embed`

The new vector index cache will be rebuilt from PostgreSQL text records, without re-scanning or re-parsing the original code files on disk.


---

## Deferred (V2)

- File watcher for automatic reindexing (fully implemented & tested!)
- Per-file semantic summaries
- Symbol index (classes, functions, modules)
- Architecture knowledge extraction (README, ADR, PROD_SPEC)
- Tree-sitter AST for TypeScript/JavaScript
- Embedding model benchmarking tool
- OS-agnostic client deploy script (Mac + Dell)


## License

This project is licensed under the AGPLv3. For commercial use without open-sourcing your application, please contact the author to purchase a commercial license.
