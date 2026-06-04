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

import hashlib
import threading
import time
from pathlib import Path

from repo_knowledge.chunker import chunk_file, chunk_project
from repo_knowledge.config import IGNORE_DIRS, PROJECTS_ROOT, SEARCH_TOP_K
from repo_knowledge.embedder import Embedder, default_embedder
from repo_knowledge.logger import log
from repo_knowledge.scanner import Project, get_project, scan_projects
from repo_knowledge.store import Store
from repo_knowledge.tracer import trace

_LIST_PROJECTS_TTL = 30.0  # seconds


class KnowledgeService:
    def __init__(self, store=None, embedder=None, projects_root=PROJECTS_ROOT):
        self._store = store or Store()
        self._embedder = embedder or default_embedder()
        self._projects_root = projects_root
        # TTL cache for list_projects — invalidated on reindex completion
        self._projects_cache: list[dict] | None = None
        self._projects_cache_ts: float = 0.0
        self._projects_cache_lock = threading.Lock()

    def list_projects(self, trace_id: str | None = None) -> list[dict]:
        with self._projects_cache_lock:
            if (
                self._projects_cache is not None
                and (time.monotonic() - self._projects_cache_ts) < _LIST_PROJECTS_TTL
            ):
                return self._projects_cache
        scanned = {p.name: p for p in scan_projects(self._projects_root)}
        indexed = set(self._store.list_projects())
        result = []
        for name, project in scanned.items():
            result.append({"name": name, "stack": project.stack, "indexed": name in indexed})
        result = sorted(result, key=lambda x: x["name"])
        with self._projects_cache_lock:
            self._projects_cache = result
            self._projects_cache_ts = time.monotonic()
        return result

    def _invalidate_projects_cache(self) -> None:
        with self._projects_cache_lock:
            self._projects_cache = None
            self._projects_cache_ts = 0.0

    def get_project_context(self, project_name: str, trace_id: str | None = None) -> dict:
        project = get_project(project_name, self._projects_root)
        if not project:
            trace("error", event_source="get_project_context",
                  message=f"Project not found: {project_name}", severity="ERROR",
                  subsystem="knowledge", trace_id=trace_id)
            return {"error": f"Project '{project_name}' not found in {self._projects_root}"}
        readme_excerpt = _read_readme(project.path)
        tree = _build_tree(project.path, max_depth=2)
        # Exclude ignored dirs from file count (matches what gets indexed)
        file_count = sum(
            1 for p in project.path.rglob("*")
            if p.is_file() and not any(part in IGNORE_DIRS for part in p.parts)
        )
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
        # Classify quality by best score in result set
        if results:
            best_score = max(r.get("score", 0.0) for r in results)
            search_quality = "good" if best_score >= 0.65 else "low"
        else:
            search_quality = "none"
        trace("search", query=query, project=project, top_k=top_k,
              results=len(results), duration_ms=duration_ms, search_quality=search_quality,
              subsystem="knowledge", trace_id=trace_id)
        for r in results:
            r["search_quality"] = search_quality
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

    def reindex_project(self, project_name: str, force: bool = False, trace_id: str | None = None) -> dict:
        t0 = time.monotonic()
        project = get_project(project_name, self._projects_root)
        if not project:
            trace("error", event_source="reindex", message=f"Project not found: {project_name}",
                  severity="ERROR", subsystem="knowledge", trace_id=trace_id)
            return {"error": f"Project '{project_name}' not found"}

        if force:
            # Full reindex: wipe everything, rechunk all files
            trace("reindex_start", project=project_name, mode="force", subsystem="knowledge", trace_id=trace_id)
            self._store.delete_project(project_name)
            trace("reindex_cleared", project=project_name, subsystem="knowledge", trace_id=trace_id)
            chunks = chunk_project(project.path, project_name)
            changed_chunks = chunks
        else:
            # Incremental reindex: skip unchanged files, re-embed changed/new, delete removed
            trace("reindex_start", project=project_name, mode="incremental", subsystem="knowledge", trace_id=trace_id)
            indexed_hashes = self._store.get_indexed_file_hashes(project_name)

            # Walk project files, compute current hashes
            current_files: dict[str, str] = {}  # rel_path → content_hash
            changed_chunks: list = []
            for file_path in project.path.rglob("*"):
                if not file_path.is_file():
                    continue
                if any(part in IGNORE_DIRS or part.endswith(".egg-info")
                       for part in file_path.parts):
                    continue
                try:
                    source = file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                rel_path = str(file_path.relative_to(project.path))
                content_hash = hashlib.sha256(source.encode()).hexdigest()
                current_files[rel_path] = content_hash
                old_hash = indexed_hashes.get(rel_path, None)
                if old_hash != content_hash:  # new or changed (old_hash "" also triggers)
                    file_mtime = 0.0
                    try:
                        file_mtime = file_path.stat().st_mtime
                    except OSError:
                        pass
                    new_chunks = chunk_file(
                        file_path, project.path, project_name,
                        content_hash=content_hash, file_mtime=file_mtime,
                    )
                    if new_chunks:
                        # Delete stale chunks for this file before re-adding
                        if rel_path in indexed_hashes:
                            self._store.delete_file(project_name, rel_path)
                        changed_chunks.extend(new_chunks)

            # Delete indexed files that no longer exist in the project
            removed = set(indexed_hashes) - set(current_files)
            for rel_path in removed:
                self._store.delete_file(project_name, rel_path)

            trace("reindex_incremental_stats", project=project_name,
                  total=len(current_files), changed=len(changed_chunks),
                  removed=len(removed), subsystem="knowledge", trace_id=trace_id)

        if not changed_chunks:
            duration_ms = round((time.monotonic() - t0) * 1000)
            trace("reindex_complete", project=project_name, chunks=0,
                  duration_ms=duration_ms, message="No changes detected", subsystem="knowledge", trace_id=trace_id)
            self._invalidate_projects_cache()
            return {"project": project_name, "chunks_indexed": 0,
                    "message": "No changes detected"}

        trace("reindex_chunked", project=project_name, chunks=len(changed_chunks), subsystem="knowledge", trace_id=trace_id)

        batch_size = 32
        all_vectors: list[list[float]] = []
        total_batches = (len(changed_chunks) + batch_size - 1) // batch_size

        for i in range(0, len(changed_chunks), batch_size):
            batch = changed_chunks[i: i + batch_size]
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

        self._store.upsert_chunks(changed_chunks, all_vectors)
        duration_ms = round((time.monotonic() - t0) * 1000)
        trace("reindex_complete", project=project_name, chunks=len(changed_chunks),
              duration_ms=duration_ms, subsystem="knowledge", trace_id=trace_id)
        self._invalidate_projects_cache()
        return {"project": project_name, "chunks_indexed": len(changed_chunks),
                "message": f"Successfully indexed {len(changed_chunks)} chunks"}


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
