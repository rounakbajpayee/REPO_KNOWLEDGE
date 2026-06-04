"""
knowledge.py — Retrieval and project context logic.

This module has zero MCP dependency. It is the core of the system.
The MCP server, any future REST API, or any other transport adapter
calls into this layer exclusively.

Public API:
  list_projects()             → list of project names with metadata
  get_project_context(name)   → full orientation payload for an agent
  search(query, project?)     → top-K relevant chunks
  get_file(project, path)     → raw file contents
  reindex_project(name)       → delete + rechunk + re-embed + store
"""

import time
from pathlib import Path

from repo_knowledge.chunker import chunk_project
from repo_knowledge.config import PROJECTS_ROOT, SEARCH_TOP_K
from repo_knowledge.embedder import Embedder, default_embedder
from repo_knowledge.logger import log
from repo_knowledge.scanner import Project, get_project, scan_projects
from repo_knowledge.store import Store
from repo_knowledge.tracer import trace


class KnowledgeService:
    def __init__(self, store=None, embedder=None, projects_root=PROJECTS_ROOT):
        self._store = store or Store()
        self._embedder = embedder or default_embedder()
        self._projects_root = projects_root

    def list_projects(self, trace_id: str | None = None) -> list[dict]:
        scanned = {p.name: p for p in scan_projects(self._projects_root)}
        indexed = set(self._store.list_projects())
        result = []
        for name, project in scanned.items():
            result.append({"name": name, "stack": project.stack, "indexed": name in indexed})
        return sorted(result, key=lambda x: x["name"])

    def get_project_context(self, project_name: str, trace_id: str | None = None) -> dict:
        project = get_project(project_name, self._projects_root)
        if not project:
            trace("error", event_source="get_project_context",
                  message=f"Project not found: {project_name}", severity="ERROR",
                  subsystem="knowledge", trace_id=trace_id)
            return {"error": f"Project '{project_name}' not found in {self._projects_root}"}
        readme_excerpt = _read_readme(project.path)
        tree = _build_tree(project.path, max_depth=2)
        file_count = sum(1 for _ in project.path.rglob("*") if _.is_file())
        indexed = project_name in set(self._store.list_projects())
        trace("get_project_context", project=project_name, file_count=file_count,
              indexed=indexed, subsystem="knowledge", trace_id=trace_id)
        return {
            "name": project.name, "path": str(project.path), "stack": project.stack,
            "readme_excerpt": readme_excerpt, "directory_tree": tree,
            "file_count": file_count, "indexed": indexed,
        }

    def search(self, query: str, project: str | None = None, top_k: int = SEARCH_TOP_K, trace_id: str | None = None) -> list[dict]:
        t0 = time.monotonic()
        vector = self._embedder.embed(query)
        results = self._store.search(vector, top_k=top_k, project=project)
        duration_ms = round((time.monotonic() - t0) * 1000)
        trace("search", query=query, project=project, top_k=top_k,
              results=len(results), duration_ms=duration_ms, subsystem="knowledge", trace_id=trace_id)
        return results

    def get_file(self, project_name: str, path: str, trace_id: str | None = None) -> dict:
        project = get_project(project_name, self._projects_root)
        if not project:
            trace("error", event_source="get_file", message=f"Project not found: {project_name}",
                  severity="ERROR", subsystem="knowledge", trace_id=trace_id)
            return {"error": f"Project '{project_name}' not found"}
        file_path = project.path / path
        if not file_path.exists():
            trace("error", event_source="get_file",
                  message=f"File not found: {path}", project=project_name,
                  severity="ERROR", subsystem="knowledge", trace_id=trace_id)
            return {"error": f"File not found: {path} in project {project_name}"}
        if not file_path.is_file():
            return {"error": f"Path is not a file: {path}"}
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            trace("error", event_source="get_file", message=str(e), project=project_name,
                  path=path, severity="ERROR", subsystem="knowledge", trace_id=trace_id)
            return {"error": f"Could not read file: {e}"}
        trace("get_file", project=project_name, path=path, line_count=content.count("\n") + 1,
              subsystem="knowledge", trace_id=trace_id)
        return {"project": project_name, "path": path, "content": content,
                "line_count": content.count("\n") + 1}

    def reindex_project(self, project_name: str, trace_id: str | None = None) -> dict:
        t0 = time.monotonic()
        project = get_project(project_name, self._projects_root)
        if not project:
            trace("error", event_source="reindex", message=f"Project not found: {project_name}",
                  severity="ERROR", subsystem="knowledge", trace_id=trace_id)
            return {"error": f"Project '{project_name}' not found"}

        trace("reindex_start", project=project_name, subsystem="knowledge", trace_id=trace_id)
        self._store.delete_project(project_name)
        trace("reindex_cleared", project=project_name, subsystem="knowledge", trace_id=trace_id)

        chunks = chunk_project(project.path, project_name)
        if not chunks:
            trace("reindex_complete", project=project_name, chunks=0,
                  message="No supported files found", subsystem="knowledge", trace_id=trace_id)
            return {"project": project_name, "chunks_indexed": 0,
                    "message": "No supported files found"}

        trace("reindex_chunked", project=project_name, chunks=len(chunks), subsystem="knowledge", trace_id=trace_id)

        batch_size = 32
        all_vectors: list[list[float]] = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            batch_num = i // batch_size + 1
            t_batch = time.monotonic()
            try:
                vectors = self._embedder.embed_batch([c.content for c in batch])
                all_vectors.extend(vectors)
                duration_ms = round((time.monotonic() - t_batch) * 1000)
                trace("embed_batch", project=project_name, batch=batch_num,
                      total_batches=total_batches, size=len(batch), duration_ms=duration_ms,
                      subsystem="knowledge", trace_id=trace_id)
            except RuntimeError as e:
                trace("error", event_source="embedder", project=project_name,
                      batch=batch_num, message=str(e), severity="ERROR", subsystem="knowledge", trace_id=trace_id)
                return {"project": project_name, "error": str(e), "chunks_indexed": 0}

        self._store.upsert_chunks(chunks, all_vectors)
        duration_ms = round((time.monotonic() - t0) * 1000)
        trace("reindex_complete", project=project_name, chunks=len(chunks),
              duration_ms=duration_ms, subsystem="knowledge", trace_id=trace_id)
        return {"project": project_name, "chunks_indexed": len(chunks),
                "message": f"Successfully indexed {len(chunks)} chunks"}


_IGNORE_IN_TREE = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".ruff_cache", "dist", "build",
}


def _read_readme(project_path: Path, max_lines: int = 100) -> str:
    for name in ("README.md", "README.rst", "README.txt", "readme.md"):
        readme = project_path / name
        if readme.exists():
            lines = readme.read_text(encoding="utf-8", errors="ignore").splitlines()
            return "\n".join(lines[:max_lines])
    return ""


def _build_tree(path: Path, max_depth: int = 2, _depth: int = 0) -> list[str]:
    if _depth >= max_depth:
        return []
    lines = []
    indent = "  " * _depth
    try:
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
    except PermissionError:
        return []
    for entry in entries:
        if entry.name in _IGNORE_IN_TREE or entry.name.startswith("."):
            continue
        prefix = "\U0001f4c1 " if entry.is_dir() else "\U0001f4c4 "
        lines.append(f"{indent}{prefix}{entry.name}")
        if entry.is_dir():
            lines.extend(_build_tree(entry, max_depth, _depth + 1))
    return lines
