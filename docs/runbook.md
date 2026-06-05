# Operational Runbook: Repository Knowledge Engine

This runbook outlines deployment, configuration, monitoring, troubleshooting, and recovery procedures for the Repository Knowledge Engine.

---

## 1. Deployment & Service Startup

The engine runs three distinct operational components:
1. **Relational Sync Filewatcher (Background Daemon)**
2. **FastAPI Monitor Dashboard (Web UI)**
3. **Model-agnostic Model Vector Client (MCP Server)**

### Relational Sync Filewatcher
The filewatcher monitors code changes under the configured projects folder and registers new/modified/deleted files incrementally in the PostgreSQL registry and Qdrant cache.

* **Silent Windows Startup (Recommended):**
  Registers a silent background execution wrapper in the Windows Startup directory (`shell:startup`):
  ```powershell
  python register_startup.py
  ```
* **Foreground Execution:**
  ```bash
  python watcher.py
  ```

### FastAPI Monitoring Dashboard
Provides connection health checks, registry counts, search play sandbox, and live Server-Sent Events (SSE) audit logs.
* **Startup:**
  ```bash
  python -m repo_knowledge.web_ui.server
  ```
* **Endpoint:** `http://localhost:8000`

### MCP Server
Provides integration capabilities for coding agents (e.g., Claude Desktop).
* **Startup:**
  ```bash
  python -m repo_knowledge.mcp_server
  ```

---

## 2. Configuration Parameters

Configuration resides in the `.env` file in the root. Verify these connections before deployment:

* **PostgreSQL 16 Database:**
  - `POSTGRES_HOST=192.168.0.234`
  - `POSTGRES_PORT=5434`
  - `POSTGRES_DB=repo_knowledge`
* **Qdrant Vector Server:**
  - `QDRANT_URL=http://192.168.0.234:6333`
* **Ollama Embeddings:**
  - `OLLAMA_URL=http://192.168.0.234:11434`
  - `EMBEDDING_MODEL=qwen3-embedding:4b`
  - `EMBEDDING_DIM=2560`
* **Redis Caching:**
  - `REDIS_URL=redis://192.168.0.234:6379/0`
  - `REDIS_TTL_S=300`

---

## 3. Monitoring & Health Diagnostics

### Visual Dashboard Health Lights
Open `http://localhost:8000` to inspect connection state bubbles.
* **Green:** Connected and responsive.
* **Red:** Service offline or network path unreachable.

### REST API Status check
Query the JSON connection health endpoint directly:
```bash
curl http://127.0.0.1:8000/api/status
```

### Log Auditing
* **Live SSE Stream:** `http://127.0.0.1:8000/api/logs` (updates live on the dashboard Traces Console).
* **Structured File Trace:** Check `logs/repo_knowledge.jsonl` for timestamped JSON traces.

---

## 4. Troubleshooting & Recovery Procedures

### Issue: Ollama Model Loading Timeout
* **Symptoms:** ReadTimeout errors on the first search/indexing run after server restarts.
* **Explanation:** Ollama requires 60–90 seconds to load heavy models (e.g., qwen3-embedding:4b = 2.5GB) into Mac Mini GPU VRAM on first invocation.
* **Solution:** Increase the `OLLAMA_TIMEOUT` value in `.env` to `120.0` or higher, or pull the model in advance (`ollama pull qwen3-embedding:4b`).

### Issue: Embedding Model Swap (Lossless Re-embed)
* **Explanation:** When changing `EMBEDDING_MODEL` or `EMBEDDING_DIM` in `.env`, all existing vectors in Qdrant become obsolete.
* **Solution:** Rebuild the vector cache losslessly from the PostgreSQL text store.
  * **Option A:** Click **Lossless Re-embed Vector Cache** on the Dashboard Web UI.
  * **Option B:** Trigger the `re_embed` MCP tool.
  * **Option C:** Run via CLI:
    ```bash
    python index.py --re-embed
    ```

### Issue: Redis Cache Out of Sync
* **Explanation:** If code updates occur and cached search results are stale.
* **Solution:** Incremental indexing automatically triggers search cache invalidation for the project. To force-clear, restart the Redis client or run:
  ```bash
  redis-cli -h 192.168.0.234 -p 6379 FLUSHALL
  ```

---

## 5. Rollback & Uninstallation

### Disable Filewatcher Startup
To prevent the filewatcher from running automatically on Windows login:
```powershell
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\start_watcher.bat" -ErrorAction SilentlyContinue
```

### Stop Running Services
```powershell
Stop-Process -Name "python"
```
