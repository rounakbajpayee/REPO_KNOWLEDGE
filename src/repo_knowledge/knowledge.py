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

from pathlib import Path

from repo_knowledge.chunker import chunk_project
from repo_knowledge.config import PROJECTS_ROOT, SEARCH_TOP_K
from repo_knowledge.embedder import Embedder, default_embedder
from repo_knowledge.scanner import Project, get_project, scan_projects
from repo_knowledge.store import Store


class KnowledgeService:
    def __init__(
        self,
        store: Store | None = None,
        embedder: Embedder | None = None,
        projects_root: str = PROJECTS_ROOT,
    ) -> None:
        self._store = store or Store()
        self._embedder = embedder or default_embedder()
        self._projects_root = projects_root

    # ── list_projects ─────────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        """
        Return all indexed projects with name, stack, and last indexed time.
        Combines scanner discovery (for stack) with store data (for index state).
        """
        scanned: dict[str, Project] = {
            p.name: p for p in scan_projects(self._projects_root)
        }
        indexed: set[str] = set(self._store.list_projects())

        result = []
        for name, project in scanned.items():
            result.append({
                "name": name,
                "stack": project.stack,
                "indexed": name in indexed,
            })
        return sorted(result, key=lambda x: x["name"])

    # ── get_project_context ───────────────────────────────────────────────────

    def get_project_context(self, project_name: str) -> dict:
        """
        Single-call cold start for an agent.

        Returns:
          - name, stack
          - readme_excerpt (first 100 lines of README if present)
          - directory_tree (2 levels, ignoring noise dirs)
          - file_count
          - indexed (bool)

        An agent calling this once has everything needed to start working
        on a project without any further file reads.
        """
        project = get_project(project_name, self._projects_root)
        if not project:
            return {"error": f"Project '{project_name}' not found in {self._projects_root}"}

        readme_excerpt = _read_readme(project.path)
        tree = _build_tree(project.path, max_depth=2)
        file_count = sum(1 for _ in project.path.rglob("*") if _.is_file())
        indexed = project_name in set(self._store.list_projects())

        return {
            "name": project.name,
            "path": str(project.path),
            "stack": project.stack,
            "readme_excerpt": readme_excerpt,
            "directory_tree": tree,
            "file_count": file_count,
            "indexed": indexed,
        }

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: str, project: str | None = None, top_k: int = SEARCH_TOP_K) -> list[dict]:
        """
        Semantic search over indexed chunks.
        Optionally scoped to a single project.
        Returns top-K chunks with score, path, symbol, content.
        """
        vector = self._embedder.embed(query)
        results = self._store.search(vector, top_k=top_k, project=project)
        return results

    # ── get_file ──────────────────────────────────────────────────────────────

    def get_file(self, project_name: str, path: str) -> dict:
        """
        Return raw file contents for a given project + relative path.
        Both arguments are required to avoid cross-project path ambiguity.
        """
        project = get_project(project_name, self._projects_root)
        if not project:
            return {"error": f"Project '{project_name}' not found"}

        file_path = project.path / path
        if not file_path.exists():
            return {"error": f"File not found: {path} in project {project_name}"}
        if not file_path.is_file():
            return {"error": f"Path is not a file: {path}"}

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            return {"error": f"Could not read file: {e}"}

        return {
            "project": project_name,
            "path": path,
            "content": content,
            "line_count": content.count("\n") + 1,
        }

    # ── reindex_project ───────────────────────────────────────────────────────

    def reindex_project(self, project_name: str) -> dict:
        """
        Full reindex of a single project:
          1. Delete existing vectors for project
          2. Chunk all supported files
          3. Embed chunks in batches
          4. Store in Qdrant

        Returns a summary dict with chunk count and any errors encountered.
        """
        project = get_project(project_name, self._projects_root)
        if not project:
            return {"error": f"Project '{project_name}' not found"}

        # Step 1: clear existing vectors
        self._store.delete_project(project_name)

        # Step 2: chunk
        chunks = chunk_project(project.path, project_name)
        if not chunks:
            return {"project": project_name, "chunks_indexed": 0, "message": "No supported files found"}

        # Step 3: embed in batches
        batch_size = 32
        all_vectors: list[list[float]] = []
        errors: list[str] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            try:
                vectors = self._embedder.embed_batch([c.content for c in batch])
                all_vectors.extend(vectors)
            except RuntimeError as e:
                errors.append(str(e))
                break  # Embedder is down — stop, don't partial-write

        if errors:
            return {"project": project_name, "error": errors[0], "chunks_indexed": 0}

        # Step 4: store
        self._store.upsert_chunks(chunks, all_vectors)

        return {
            "project": project_name,
            "chunks_indexed": len(chunks),
            "message": f"Successfully indexed {len(chunks)} chunks",
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

_IGNORE_IN_TREE = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".ruff_cache", "dist", "build",
}


def _read_readme(project_path: Path, max_lines: int = 100) -> str:
    """Return first max_lines of README if present, else empty string."""
    for name in ("README.md", "README.rst", "README.txt", "readme.md"):
        readme = project_path / name
        if readme.exists():
            lines = readme.read_text(encoding="utf-8", errors="ignore").splitlines()
            return "\n".join(lines[:max_lines])
    return ""


def _build_tree(path: Path, max_depth: int = 2, _depth: int = 0) -> list[str]:
    """Return a flat list of strings representing the directory tree."""
    if _depth >= max_depth:
        return []
    lines: list[str] = []
    indent = "  " * _depth
    try:
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
    except PermissionError:
        return []
    for entry in entries:
        if entry.name in _IGNORE_IN_TREE or entry.name.startswith("."):
            continue
        prefix = "📁 " if entry.is_dir() else "📄 "
        lines.append(f"{indent}{prefix}{entry.name}")
        if entry.is_dir():
            lines.extend(_build_tree(entry, max_depth, _depth + 1))
    return lines
