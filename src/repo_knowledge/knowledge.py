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
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repo_knowledge import cache as search_cache
from repo_knowledge import reranker as search_reranker
from repo_knowledge.chunker import chunk_file, chunk_project
from repo_knowledge.config import (
    IGNORE_EXTENSIONS,
    PROJECTS_ROOT,
    RERANK_ENABLED,
    SEARCH_TOP_K,
    SUPPORTED_EXTENSIONS,
)
from repo_knowledge.embedder import default_embedder
from repo_knowledge.scanner import get_project, scan_projects
from repo_knowledge.store import Store
from repo_knowledge.tracer import trace

_LIST_PROJECTS_TTL = 30.0  # seconds


class KnowledgeService:
    def __init__(
        self,
        store: Any = None,
        embedder: Any = None,
        projects_root: Path | str = PROJECTS_ROOT,
    ) -> None:
        self._store = store or Store()
        self._embedder = embedder or default_embedder()
        self._projects_root = str(projects_root)
        # TTL cache for list_projects — invalidated on reindex completion
        self._projects_cache: list[dict] | None = None
        self._projects_cache_ts: float = 0.0
        self._projects_cache_lock = threading.Lock()
        self._vault_lock = threading.Lock()
        self._pg = self._store._pg

    def list_projects(self) -> list[dict]:
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
        result = sorted(result, key=lambda x: str(x["name"]))
        with self._projects_cache_lock:
            self._projects_cache = result
            self._projects_cache_ts = time.monotonic()
        return result

    def _invalidate_projects_cache(self) -> None:
        with self._projects_cache_lock:
            self._projects_cache = None
            self._projects_cache_ts = 0.0

    def get_project_context(self, project_name: str) -> dict:
        project = get_project(project_name, self._projects_root)
        if not project:
            trace(
                "error",
                event_source="get_project_context",
                message=f"Project not found: {project_name}",
                severity="ERROR",
                subsystem="knowledge",
            )
            return {"error": f"Project '{project_name}' not found in {self._projects_root}"}
        readme_excerpt = _read_readme(project.path)
        tree = _build_tree(project.path, max_depth=2)
        # Exclude ignored dirs from file count (matches what gets indexed)
        from repo_knowledge.scanner import list_project_files

        file_count = len(list_project_files(project.path))
        indexed = project_name in set(self._store.list_projects())
        trace(
            "get_project_context",
            project=project_name,
            file_count=file_count,
            indexed=indexed,
            subsystem="knowledge",
        )
        return {
            "name": project.name,
            "path": str(project.path),
            "stack": project.stack,
            "readme_excerpt": readme_excerpt,
            "directory_tree": tree,
            "file_count": file_count,
            "indexed": indexed,
        }

    def search(
        self, query: str, project: str | None = None, top_k: int = SEARCH_TOP_K
    ) -> list[dict]:
        t0 = time.monotonic()

        # ── 1. Redis cache check ──────────────────────────────────────────────────
        cached = search_cache.get_cached(query, project, top_k)
        if cached is not None:
            duration_ms = round((time.monotonic() - t0) * 1000)
            trace(
                "search",
                query=query,
                project=project,
                top_k=top_k,
                results=len(cached),
                duration_ms=duration_ms,
                cache_hit=True,
                search_quality="cached",
                subsystem="knowledge",
            )
            for r in cached:
                r["search_quality"] = "cached"
            return cached

        # ── 2. Embed query ─────────────────────────────────────────────────────────
        vector = self._embedder.embed(query)

        # ── 3. Hybrid recall: Qdrant + BM25 via RRF ────────────────────────────────
        candidates = self._store.search(vector, top_k=top_k, project=project, query_text=query)

        # ── 4. Cross-encoder rerank ───────────────────────────────────────────────
        if RERANK_ENABLED and candidates:
            results = search_reranker.rerank(query, candidates, top_k=top_k)
        else:
            results = candidates[:top_k]

        duration_ms = round((time.monotonic() - t0) * 1000)

        # Classify quality by best score in result set
        if results:
            best_score = max(r.get("rerank_score", r.get("score", 0.0)) for r in results)
            search_quality = "good" if best_score >= 0.65 else "low"
        else:
            search_quality = "none"

        trace(
            "search",
            query=query,
            project=project,
            top_k=top_k,
            results=len(results),
            duration_ms=duration_ms,
            cache_hit=False,
            reranked=RERANK_ENABLED,
            search_quality=search_quality,
            subsystem="knowledge",
        )
        for r in results:
            r["search_quality"] = search_quality

        # ── 5. Store in Redis ───────────────────────────────────────────────────
        if results:
            search_cache.set_cached(query, project, top_k, results)

        return results

    def list_files(
        self, project_name: str, path_prefix: str | None = None, extension: str | None = None
    ) -> dict:
        project_root = Path(self._projects_root) / project_name
        if not project_root.exists() or not project_root.is_dir():
            return {"error": f"Project '{project_name}' not found."}

        # Query database to get precomputed max end_line from chunks table
        db_line_counts = {}
        try:
            with self._pg._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT path, MAX(end_line)
                        FROM chunks
                        WHERE project = %s
                        GROUP BY path;
                    """,
                        (project_name,),
                    )
                    db_line_counts = {row[0].replace("\\", "/"): row[1] for row in cur.fetchall()}
        except Exception:
            pass

        from repo_knowledge.scanner import list_project_files

        files_data = []
        for file_path in list_project_files(project_root):
            rel_path = str(file_path.relative_to(project_root)).replace("\\", "/")
            if path_prefix:
                normalized_prefix = path_prefix.replace("\\", "/")
                if not rel_path.startswith(normalized_prefix):
                    continue

            suffix = file_path.suffix.lower()
            if extension and extension != "*":
                if suffix != extension.lower():
                    continue
            elif suffix in IGNORE_EXTENSIONS or (
                suffix not in SUPPORTED_EXTENSIONS
                and suffix not in (".plist", ".conf", ".ini")
                and file_path.name.lower() not in ("docker-compose.yml", "docker-compose.yaml")
            ):
                continue

            line_count = db_line_counts.get(rel_path)
            if line_count is None:
                try:
                    size = file_path.stat().st_size
                    if size < 50000:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            line_count = sum(1 for _ in f)
                    else:
                        line_count = 0
                except Exception:
                    line_count = 0

            language = "text"
            if suffix == ".py":
                language = "python"
            elif suffix in (".js", ".jsx"):
                language = "javascript"
            elif suffix in (".ts", ".tsx"):
                language = "typescript"
            elif suffix == ".md":
                language = "markdown"
            elif suffix in (".yml", ".yaml"):
                language = "yaml"
            elif suffix == ".json":
                language = "json"

            files_data.append({"path": rel_path, "language": language, "line_count": line_count})

        return {
            "project": project_name,
            "files": sorted(files_data, key=lambda x: x["path"]),
            "total": len(files_data),
            "filters": {"path_prefix": path_prefix, "extension": extension},
        }

    def search_symbols(self, query: str, project: str | None = None, top_k: int = 10) -> list[dict]:
        results = self.search(query, project, top_k)
        for hit in results:
            hit.pop("content", None)
        return results

    def get_chunks_for_file(self, project_name: str, path: str) -> dict:
        chunks = self._store.get_chunks_for_path(project_name, path)
        if not chunks:
            project_root = Path(self._projects_root) / project_name
            if not project_root.exists() or not project_root.is_dir():
                return {"error": f"Project '{project_name}' not found."}
            file_path = project_root / path
            if not file_path.exists() or not file_path.is_file():
                return {"error": f"File not found: {path} in project {project_name}"}
            return {"error": f"No chunks found for {path}", "chunks": [], "total": 0}

        simplified_chunks = []
        for c in chunks:
            simplified_chunks.append(
                {
                    "symbol": c.get("symbol", ""),
                    "chunk_type": c.get("chunk_type", ""),
                    "start_line": c.get("start_line", 0),
                    "end_line": c.get("end_line", 0),
                }
            )

        simplified_chunks.sort(key=lambda x: x["start_line"])
        return {
            "project": project_name,
            "path": path,
            "chunks": simplified_chunks,
            "total": len(simplified_chunks),
        }

    def get_file(
        self,
        project_name: str,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict:
        project = get_project(project_name, self._projects_root)
        if not project:
            trace(
                "error",
                event_source="get_file",
                message=f"Project not found: {project_name}",
                severity="ERROR",
                subsystem="knowledge",
            )
            return {"error": f"Project '{project_name}' not found"}
        file_path = project.path / path
        if not file_path.exists():
            trace(
                "error",
                event_source="get_file",
                message=f"File not found: {path}",
                project=project_name,
                severity="ERROR",
                subsystem="knowledge",
            )
            return {"error": f"File not found: {path} in project {project_name}"}
        if not file_path.is_file():
            return {"error": f"Path is not a file: {path}"}
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            trace(
                "error",
                event_source="get_file",
                message=str(e),
                project=project_name,
                path=path,
                severity="ERROR",
                subsystem="knowledge",
            )
            return {"error": f"Could not read file: {e}"}

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        if start_line and end_line and start_line > end_line:
            return {"error": "start_line must be <= end_line"}
        if start_line and start_line > total_lines:
            return {"error": f"start_line {start_line} exceeds file length {total_lines}"}

        sliced_content = content
        if start_line is not None or end_line is not None:
            s_idx = (start_line - 1) if start_line is not None else 0
            e_idx = end_line if end_line is not None else total_lines
            s_idx = max(0, min(s_idx, total_lines))
            e_idx = max(0, min(e_idx, total_lines))
            sliced_content = "".join(lines[s_idx:e_idx])

        trace(
            "get_file",
            project=project_name,
            path=path,
            start_line=start_line,
            end_line=end_line,
            line_count=total_lines,
            subsystem="knowledge",
        )

        ret = {
            "project": project_name,
            "path": path,
            "content": sliced_content,
            "line_count": total_lines,
        }
        if start_line is not None:
            ret["start_line"] = start_line
        if end_line is not None:
            ret["end_line"] = end_line
        return ret

    def reindex_project(self, project_name: str, force: bool = False) -> dict:
        t0 = time.monotonic()
        project = get_project(project_name, self._projects_root)
        if not project:
            trace(
                "error",
                event_source="reindex",
                message=f"Project not found: {project_name}",
                severity="ERROR",
                subsystem="knowledge",
            )
            return {"error": f"Project '{project_name}' not found"}

        if force:
            # Full reindex: wipe everything, rechunk all files
            trace("reindex_start", project=project_name, mode="force", subsystem="knowledge")
            self._store.delete_project(project_name)
            trace("reindex_cleared", project=project_name, subsystem="knowledge")
            chunks = chunk_project(project.path, project_name)
            changed_chunks = chunks
        else:
            # Incremental reindex: skip unchanged files, re-embed changed/new, delete removed
            trace("reindex_start", project=project_name, mode="incremental", subsystem="knowledge")
            indexed_hashes = self._store.get_indexed_file_hashes(project_name)

            # Fetch indexed modification times from database
            indexed_mtimes: dict[str, float] = {}
            try:
                indexed_mtimes = self._store.get_indexed_file_mtimes(project_name)
            except Exception:
                pass

            # Walk project files, compute current hashes
            from repo_knowledge.scanner import list_project_files

            current_files: dict[str, str] = {}  # rel_path → content_hash
            changed_chunks = []
            for file_path in list_project_files(project.path):
                rel_path = str(file_path.relative_to(project.path))

                # Fetch modification time first
                file_mtime = 0.0
                try:
                    file_mtime = file_path.stat().st_mtime
                except OSError:
                    pass

                # If path exists in DB and mtime matches, skip reading and hashing
                old_mtime = indexed_mtimes.get(rel_path, None)
                old_hash = indexed_hashes.get(rel_path, None)
                if (
                    old_mtime is not None
                    and old_hash is not None
                    and abs(old_mtime - file_mtime) < 0.01
                ):
                    current_files[rel_path] = old_hash
                    continue

                try:
                    source = file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                content_hash = hashlib.sha256(source.encode()).hexdigest()
                current_files[rel_path] = content_hash

                if old_hash != content_hash:  # new or changed (old_hash "" also triggers)
                    new_chunks = chunk_file(
                        file_path,
                        project.path,
                        project_name,
                        content_hash=content_hash,
                        file_mtime=file_mtime,
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

            trace(
                "reindex_incremental_stats",
                project=project_name,
                total=len(current_files),
                changed=len(changed_chunks),
                removed=len(removed),
                subsystem="knowledge",
            )

        if not changed_chunks:
            duration_ms = round((time.monotonic() - t0) * 1000)
            trace(
                "reindex_complete",
                project=project_name,
                chunks=0,
                duration_ms=duration_ms,
                message="No changes detected",
                subsystem="knowledge",
            )
            self._invalidate_projects_cache()
            return {"project": project_name, "chunks_indexed": 0, "message": "No changes detected"}

        trace(
            "reindex_chunked",
            project=project_name,
            chunks=len(changed_chunks),
            subsystem="knowledge",
        )

        batch_size = 32
        all_vectors: list[list[float]] = []
        total_batches = (len(changed_chunks) + batch_size - 1) // batch_size

        for i in range(0, len(changed_chunks), batch_size):
            batch = changed_chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            t_batch = time.monotonic()
            try:
                vectors = self._embedder.embed_batch([c.content for c in batch])
                all_vectors.extend(vectors)
                duration_ms = round((time.monotonic() - t_batch) * 1000)
                trace(
                    "embed_batch",
                    project=project_name,
                    batch=batch_num,
                    total_batches=total_batches,
                    size=len(batch),
                    duration_ms=duration_ms,
                    subsystem="knowledge",
                )
            except RuntimeError as e:
                trace(
                    "error",
                    event_source="embedder",
                    project=project_name,
                    batch=batch_num,
                    message=str(e),
                    severity="ERROR",
                    subsystem="knowledge",
                )
                return {"project": project_name, "error": str(e), "chunks_indexed": 0}

        self._store.upsert_chunks(changed_chunks, all_vectors)
        duration_ms = round((time.monotonic() - t0) * 1000)
        trace(
            "reindex_complete",
            project=project_name,
            chunks=len(changed_chunks),
            duration_ms=duration_ms,
            subsystem="knowledge",
        )
        self._invalidate_projects_cache()
        # Flush stale search cache entries for this project
        search_cache.flush_project(project_name)
        return {
            "project": project_name,
            "chunks_indexed": len(changed_chunks),
            "message": f"Successfully indexed {len(changed_chunks)} chunks",
        }

    def log_decision(
        self,
        topic: str,
        name: str,
        description: str,
        rationale: str,
        options_considered: list[dict] | None = None,
    ) -> dict:
        """
        Append a timestamped decision entry to a Markdown memory file and PostgreSQL.
        Creates the vault directory and topic file if they do not exist.
        Thread-safe.
        """
        # Validate/slugify topic to avoid traversal vulnerabilities
        slugified_topic = re.sub(r"[^a-zA-Z0-9_-]", "_", topic).lower()

        # 1. Try writing to PostgreSQL
        pg_err = None
        try:
            self._pg.log_decision(
                topic=slugified_topic,
                entry_name=name,
                description=description,
                rationale=rationale,
                options_considered=options_considered,
            )
        except Exception as e:
            pg_err = str(e)
            trace(
                "warning",
                event_source="log_decision_db",
                message=f"Failed to log decision to Postgres: {e}",
                severity="WARNING",
                subsystem="knowledge",
            )

        # 2. Write to Markdown file
        vault_dir = Path(self._projects_root) / "knowledge_vault"

        with self._vault_lock:
            try:
                vault_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                trace(
                    "error",
                    event_source="log_decision",
                    message=f"Failed to create vault dir: {e}",
                    severity="ERROR",
                    subsystem="knowledge",
                )
                return {"error": f"Failed to create knowledge_vault directory: {e}"}

            vault_file = vault_dir / f"{slugified_topic}.md"
            now_str = datetime.now(timezone.utc).isoformat()

            # Format options_considered list if present
            options_str = ""
            if options_considered:
                options_str = "- **Options Considered:**\n"
                for opt in options_considered:
                    opt_name = opt.get("name", "Unknown Option")
                    opt_status = opt.get("status", "REJECTED")
                    opt_rat = opt.get("rationale", "")
                    marker = "[REJECTED]" if opt_status.upper() == "REJECTED" else "[SELECTED]"
                    options_str += f"  - {marker} *{opt_name} ({opt_status}):* {opt_rat}\n"

            new_entry = f"## [{now_str}] {name}\n- **Description:** {description}\n{options_str}- **Rationale:** {rationale}\n"  # noqa: E501

            if not vault_file.exists():
                initial = f"""---
topic: {slugified_topic}
created_at: {now_str}
last_modified: {now_str}
entries_count: 1
---
# Decision Log: {slugified_topic.replace("_", " ").title()}

{new_entry}"""
                try:
                    vault_file.write_text(initial, encoding="utf-8")
                except OSError as e:
                    trace(
                        "error",
                        event_source="log_decision",
                        message=f"Failed to write initial file: {e}",
                        severity="ERROR",
                        subsystem="knowledge",
                    )
                    return {"error": f"Failed to write decision file: {e}"}
            else:
                try:
                    content = vault_file.read_text(encoding="utf-8", errors="ignore")
                except OSError as e:
                    return {"error": f"Failed to read existing decision file: {e}"}

                # Parse current frontmatter
                frontmatter = {}
                main_body = content
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        fm_text = parts[1]
                        main_body = parts[2]
                        for line in fm_text.splitlines():
                            if ":" in line:
                                k, v = line.split(":", 1)
                                frontmatter[k.strip()] = v.strip()

                # Update counts
                count = int(frontmatter.get("entries_count", 0)) + 1
                frontmatter["entries_count"] = str(count)
                frontmatter["last_modified"] = now_str

                fm_str = (
                    "---\n" + "\n".join(f"{k}: {v}" for k, v in frontmatter.items()) + "\n---\n"
                )
                main_body_str = main_body.strip()
                updated_content = fm_str + main_body_str + "\n\n" + new_entry.strip() + "\n"

                try:
                    vault_file.write_text(updated_content, encoding="utf-8")
                except OSError as e:
                    trace(
                        "error",
                        event_source="log_decision",
                        message=f"Failed to append to file: {e}",
                        severity="ERROR",
                        subsystem="knowledge",
                    )
                    return {"error": f"Failed to update decision file: {e}"}

        trace("log_decision", topic=slugified_topic, entry_name=name, subsystem="knowledge")
        msg = f"Successfully logged decision '{name}' under topic '{slugified_topic}'"
        if pg_err:
            msg += f" (Postgres write failed: {pg_err})"
        return {"topic": slugified_topic, "message": msg}

    def get_decision_history(
        self,
        topic: str,
        limit: int = 3,
        full_history: bool = False,
    ) -> dict:
        """
        Retrieve chronological decision log entries for a topic.
        Queries Postgres first, falling back to markdown if unavailable.
        To save token window, limits to last N entries by default.
        """
        slugified_topic = re.sub(r"[^a-zA-Z0-9_-]", "_", topic).lower()

        # Try loading from PostgreSQL
        try:
            entries_db = self._pg.get_decision_history(slugified_topic, limit=0, full_history=True)
            total_count = len(entries_db)

            # Slice according to limit/full_history
            if not full_history and limit > 0:
                ret_entries = entries_db[-limit:]
                truncated = total_count > limit
            else:
                ret_entries = entries_db
                truncated = False

            # Reconstruct history markdown string to preserve format contract
            history_parts = []
            for entry in ret_entries:
                entry_name = entry["name"]
                logged_at = entry["logged_at"]
                description = entry["description"]
                rationale = entry["rationale"]
                options_considered = entry.get("options_considered")

                options_str = ""
                if options_considered:
                    options_str = "- **Options Considered:**\n"
                    for opt in options_considered:
                        opt_name = opt.get("name", "Unknown Option")
                        opt_status = opt.get("status", "REJECTED")
                        opt_rat = opt.get("rationale", "")
                        marker = "[REJECTED]" if opt_status.upper() == "REJECTED" else "[SELECTED]"
                        options_str += f"  - {marker} *{opt_name} ({opt_status}):* {opt_rat}\n"

                history_parts.append(
                    f"## [{logged_at}] {entry_name}\n- **Description:** {description}\n{options_str}- **Rationale:** {rationale}"  # noqa: E501
                )

            history_text = (
                f"# Decision Log: {slugified_topic.replace('_', ' ').title()}\n\n"
                + "\n\n".join(history_parts)
            )
            if truncated:
                history_text += f"\n\n*Note: History truncated. Showing last {limit} of {total_count} entries. Retrieve with full_history=true to view all.*"  # noqa: E501

            # Create a mock frontmatter dict
            frontmatter = {
                "topic": slugified_topic,
                "entries_count": str(total_count),
            }
            if entries_db:
                frontmatter["created_at"] = entries_db[0]["logged_at"]
                frontmatter["last_modified"] = entries_db[-1]["logged_at"]

            trace(
                "get_decision_history",
                topic=slugified_topic,
                limit=limit,
                full_history=full_history,
                total_entries=total_count,
                shown_entries=len(ret_entries),
                subsystem="knowledge",
                source="postgres",
            )

            return {
                "topic": slugified_topic,
                "frontmatter": frontmatter,
                "history": history_text,
                "total_entries": total_count,
                "shown_entries": len(ret_entries),
            }

        except Exception as e:
            trace(
                "warning",
                event_source="get_decision_history_db",
                message=f"Failed to query decision history from Postgres: {e}. Falling back to Markdown.",  # noqa: E501
                severity="WARNING",
                subsystem="knowledge",
            )

        # Fallback to Markdown
        vault_dir = Path(self._projects_root) / "knowledge_vault"
        vault_file = vault_dir / f"{slugified_topic}.md"

        if not vault_file.exists():
            return {"error": f"Decision log for topic '{slugified_topic}' does not exist"}

        with self._vault_lock:
            try:
                content = vault_file.read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                trace(
                    "error",
                    event_source="get_decision_history",
                    message=f"Failed to read file: {e}",
                    severity="ERROR",
                    subsystem="knowledge",
                )
                return {"error": f"Could not read decision log: {e}"}

        frontmatter = {}
        main_body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                fm_text = parts[1]
                main_body = parts[2]
                for line in fm_text.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        frontmatter[k.strip()] = v.strip()

        # Split main body by entry headers
        entry_splits = re.split(r"^(## \[[^\]]+\] .*?)$", main_body, flags=re.MULTILINE)
        intro = entry_splits[0].strip()
        entries = []
        for i in range(1, len(entry_splits), 2):
            header = entry_splits[i]
            body = entry_splits[i + 1] if i + 1 < len(entry_splits) else ""
            entries.append(header + "\n" + body.strip())

        total_count = len(entries)

        if not full_history and limit > 0:
            ret_entries = entries[-limit:]
            truncated = len(entries) > limit
        else:
            ret_entries = entries
            truncated = False

        history_text = intro + "\n\n" + "\n\n".join(ret_entries)
        if truncated:
            history_text += f"\n\n*Note: History truncated. Showing last {limit} of {total_count} entries. Retrieve with full_history=true to view all.*"  # noqa: E501

        trace(
            "get_decision_history",
            topic=slugified_topic,
            limit=limit,
            full_history=full_history,
            subsystem="knowledge",
            source="markdown",
        )

        return {
            "topic": slugified_topic,
            "frontmatter": frontmatter,
            "history": history_text,
            "total_entries": total_count,
            "shown_entries": len(ret_entries),
        }

    def re_embed_all_projects(self, trace_id: str | None = None) -> dict:
        """
        Wipe the Qdrant collection and re-embed all chunks stored in PostgreSQL
        using the currently active embedding model.
        """
        t0 = time.monotonic()
        trace("re_embed_start", subsystem="knowledge", trace_id=trace_id)

        # 1. Fetch all chunks from Postgres
        try:
            with self._pg._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT c.id, c.project, c.path, c.language,
                               c.chunk_type, c.symbol, c.content,
                               c.start_line, c.end_line,
                               f.content_hash, f.file_mtime
                        FROM chunks c
                        JOIN files f ON c.file_id = f.id;
                    """)
                    chunks_data = [
                        {
                            "id": str(row[0]),
                            "project": row[1],
                            "path": row[2],
                            "language": row[3],
                            "chunk_type": row[4],
                            "symbol": row[5],
                            "content": row[6],
                            "start_line": row[7],
                            "end_line": row[8],
                            "content_hash": row[9],
                            "file_mtime": row[10],
                        }
                        for row in cur.fetchall()
                    ]
        except Exception as e:
            trace(
                "error",
                event_source="re_embed",
                message=f"Failed to fetch chunks from DB: {e}",
                severity="ERROR",
                subsystem="knowledge",
                trace_id=trace_id,
            )
            return {"error": f"Failed to fetch chunks from database: {e}"}

        if not chunks_data:
            trace(
                "re_embed_complete",
                message="No chunks found in database",
                duration_ms=0,
                subsystem="knowledge",
                trace_id=trace_id,
            )
            return {"message": "No chunks found in database to re-embed", "chunks_reembedded": 0}

        # 2. Reset the Qdrant collection
        try:
            self._store._ensure_collection()  # Ensure it's active/created first
            self._store._client.delete_collection(self._store._collection)
            self._store._collection_ready = False
            self._store._ensure_collection()  # Recreate it fresh
        except Exception as e:
            trace(
                "error",
                event_source="re_embed",
                message=f"Failed to reset Qdrant collection: {e}",
                severity="ERROR",
                subsystem="knowledge",
                trace_id=trace_id,
            )
            return {"error": f"Failed to reset Qdrant collection: {e}"}

        # 3. Batch embed the text content and upsert to Qdrant
        batch_size = 32
        total_chunks = len(chunks_data)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        now = datetime.now(timezone.utc).isoformat()

        from qdrant_client.http import models as qdrant_models

        for i in range(0, total_chunks, batch_size):
            batch = chunks_data[i : i + batch_size]
            batch_num = i // batch_size + 1
            t_batch = time.monotonic()

            try:
                # Embed batch
                texts = [c["content"] for c in batch]
                vectors = self._embedder.embed_batch(texts)

                # Prepare Qdrant points
                points = [
                    qdrant_models.PointStruct(
                        id=c["id"],
                        vector=vec,
                        payload={
                            "project": c["project"],
                            "path": c["path"],
                            "language": c["language"],
                            "chunk_type": c["chunk_type"],
                            "symbol": c["symbol"],
                            "content": c["content"],
                            "start_line": c["start_line"],
                            "end_line": c["end_line"],
                            "embedding_model": self._store._embedding_model,
                            "indexed_at": now,
                            "content_hash": c["content_hash"],
                            "file_mtime": c["file_mtime"],
                        },
                    )
                    for c, vec in zip(batch, vectors)
                ]

                # Upsert to Qdrant
                self._store._client.upsert(collection_name=self._store._collection, points=points)

                duration_ms = round((time.monotonic() - t_batch) * 1000)
                trace(
                    "embed_batch",
                    project="all_reembed",
                    batch=batch_num,
                    total_batches=total_batches,
                    size=len(batch),
                    duration_ms=duration_ms,
                    subsystem="knowledge",
                    trace_id=trace_id,
                )

            except Exception as e:
                trace(
                    "error",
                    event_source="re_embed_batch",
                    batch=batch_num,
                    message=f"Failed to embed/upsert batch: {e}",
                    severity="ERROR",
                    subsystem="knowledge",
                    trace_id=trace_id,
                )
                return {"error": f"Failed to embed/upsert batch {batch_num}: {e}"}

        duration_ms = round((time.monotonic() - t0) * 1000)
        trace(
            "re_embed_complete",
            chunks=total_chunks,
            duration_ms=duration_ms,
            subsystem="knowledge",
            trace_id=trace_id,
        )
        self._invalidate_projects_cache()
        return {
            "message": f"Successfully re-embedded {total_chunks} chunks using model '{self._store._embedding_model}'",  # noqa: E501
            "chunks_reembedded": total_chunks,
        }


_IGNORE_IN_TREE = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
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
