"""
server.py — FastAPI web dashboard backend server for REPO_KNOWLEDGE.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from repo_knowledge.knowledge import KnowledgeService
from repo_knowledge.config import POSTGRES_PORT, POSTGRES_HOST
from repo_knowledge.tracer import trace

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_ui_server")

app = FastAPI(title="Repository Knowledge Dashboard")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_svc: Optional[KnowledgeService] = None

def get_service() -> KnowledgeService:
    global _svc
    if _svc is None:
        _svc = KnowledgeService()
    return _svc


# Models
class SearchRequest(BaseModel):
    query: str
    project: Optional[str] = None
    top_k: int = 5


class ReindexRequest(BaseModel):
    project: str


# Endpoints

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the single-page application dashboard."""
    html_path = Path(__file__).resolve().parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
def get_status():
    """Retrieve connection status of all external dependencies."""
    svc = get_service()
    
    postgres_ok = False
    try:
        postgres_ok = svc._pg.health_check()
    except Exception:
        pass

    qdrant_ok = False
    try:
        qdrant_ok = svc._store.health_check()
    except Exception:
        pass

    ollama_ok = False
    try:
        ollama_ok = svc._embedder.health_check()
    except Exception:
        pass

    return {
        "postgres": {
            "status": "connected" if postgres_ok else "disconnected",
            "host": f"{POSTGRES_HOST}:{5434}"
        },
        "qdrant": {
            "status": "connected" if qdrant_ok else "disconnected",
            "url": svc._store._url
        },
        "ollama": {
            "status": "connected" if ollama_ok else "disconnected",
            "model": svc._embedder.model_name
        }
    }


@app.get("/api/projects")
def get_projects():
    """Get all projects from scanner and DB statistics."""
    svc = get_service()
    
    # Scanned local projects
    try:
        scanned = svc.list_projects()
    except Exception as e:
        logger.exception("Failed to scan projects")
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {e}")

    # Query detailed counts from PostgreSQL
    pg_stats = {}
    try:
        if svc._pg.health_check():
            with svc._pg._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT p.name, p.stack, p.last_indexed_at,
                               (SELECT COUNT(*) FROM files f WHERE f.project_id = p.id) as file_count,
                               (SELECT COUNT(*) FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.project_id = p.id) as chunk_count
                        FROM projects p;
                    """)
                    for row in cur.fetchall():
                        pg_stats[row[0]] = {
                            "stack": row[1],
                            "last_indexed_at": row[2].isoformat() if row[2] else None,
                            "file_count": row[3],
                            "chunk_count": row[4]
                        }
    except Exception as e:
        logger.warning(f"Could not retrieve PostgreSQL project statistics: {e}")

    results = []
    for proj in scanned:
        name = proj["name"]
        stats = pg_stats.get(name, {})
        results.append({
            "name": name,
            "stack": stats.get("stack") or proj["stack"] or "Unknown",
            "indexed": proj["indexed"],
            "last_indexed_at": stats.get("last_indexed_at"),
            "file_count": stats.get("file_count") or 0,
            "chunk_count": stats.get("chunk_count") or 0
        })

    return results


@app.post("/api/search")
def search_sandbox(req: SearchRequest):
    """Semantic search playground."""
    svc = get_service()
    try:
        results = svc.search(query=req.query, project=req.project, top_k=req.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reindex")
async def reindex_project(req: ReindexRequest):
    """Triggers project reindexing in the background."""
    svc = get_service()
    
    # Run in thread executor to prevent blocking
    loop = asyncio.get_event_loop()
    try:
        res = await loop.run_in_executor(None, svc.reindex_project, req.project)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/re_embed")
async def re_embed_projects():
    """Triggers vector re-embedding for all projects in PostgreSQL."""
    svc = get_service()
    
    loop = asyncio.get_event_loop()
    try:
        res = await loop.run_in_executor(None, svc.re_embed_all_projects)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
def stream_logs(request: Request):
    """SSE endpoint streaming live audit log records from PostgreSQL."""
    svc = get_service()
    
    async def sse_generator():
        last_time = datetime.now(timezone.utc)
        
        # Helper to query new logs from Postgres
        def fetch_new_logs(since_time):
            if not svc._pg.health_check():
                return []
            try:
                with svc._pg._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT ts, trace_id, event, severity, subsystem, duration_ms, payload
                            FROM audit_logs
                            WHERE ts > %s
                            ORDER BY id ASC;
                        """, (since_time,))
                        return [
                            {
                                "ts": r[0].isoformat(), "trace_id": r[1], "event": r[2],
                                "severity": r[3], "subsystem": r[4], "duration_ms": r[5],
                                "payload": r[6]
                            }
                            for r in cur.fetchall()
                        ]
            except Exception as e:
                logger.error(f"Failed to query audit logs: {e}")
                return []

        # Start by yielding the last 30 logs for context
        try:
            initial_logs = svc._pg.get_audit_logs(limit=30)
            initial_logs.reverse()  # Chronological order
            for log_record in initial_logs:
                yield f"data: {json.dumps(log_record)}\n\n"
        except Exception:
            pass

        while True:
            if await request.is_disconnected():
                break

            new_logs = fetch_new_logs(last_time)
            if new_logs:
                for log_record in new_logs:
                    yield f"data: {json.dumps(log_record)}\n\n"
                # Update last time to match the latest retrieved log timestamp
                try:
                    last_time_str = new_logs[-1]["ts"]
                    if last_time_str.endswith("Z"):
                        last_time_str = last_time_str.replace("Z", "+00:00")
                    last_time = datetime.fromisoformat(last_time_str)
                except Exception:
                    last_time = datetime.now(timezone.utc)
            
            await asyncio.sleep(1.0)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("repo_knowledge.web_ui.server:app", host="127.0.0.1", port=8000, reload=True)

