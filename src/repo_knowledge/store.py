"""
store.py — Qdrant read/write operations.

Responsibilities:
  - Create collection on first run if it doesn't exist
  - Upsert chunks with vectors + full metadata payload
  - Delete all vectors for a project (used before reindexing)
  - Vector search with optional project filter

Payload schema per chunk:
  {
    "project":         str,   # e.g. "LENS"
    "path":            str,   # relative path e.g. "src/ocr/service.py"
    "language":        str,   # "python" | "typescript" | ...
    "chunk_type":      str,   # "function" | "class" | "section" | "block"
    "symbol":          str,   # function/class name or ""
    "content":         str,   # raw chunk text
    "start_line":      int,
    "end_line":        int,
    "embedding_model": str,   # e.g. "nomic-embed-text" — required for benchmarking
    "indexed_at":      str,   # ISO timestamp
  }

`project` is stored as a keyword field so Qdrant can filter-delete by it
during reindexing — this is a required design constraint, do not remove.
"""

import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from repo_knowledge.chunker import Chunk
from repo_knowledge.config import EMBEDDING_DIM, EMBEDDING_MODEL, QDRANT_COLLECTION, QDRANT_URL


class Store:
    def __init__(
        self,
        url: str = QDRANT_URL,
        collection: str = QDRANT_COLLECTION,
        embedding_dim: int = EMBEDDING_DIM,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._embedding_dim = embedding_dim
        self._embedding_model = embedding_model
        self._ensure_collection()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=self._embedding_dim,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def health_check(self) -> bool:
        """Returns True if Qdrant is reachable."""
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store chunks and their vectors. chunks and vectors must be same length."""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")

        now = datetime.now(timezone.utc).isoformat()
        points = [
            qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "project": chunk.project,
                    "path": chunk.path,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "symbol": chunk.symbol,
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "embedding_model": self._embedding_model,
                    "indexed_at": now,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        # Upsert in batches of 100 to avoid large payloads
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i: i + batch_size],
            )

    def delete_project(self, project: str) -> None:
        """
        Delete all vectors for a project.
        Called before reindexing to ensure a clean slate.
        Relies on `project` being stored as a filterable payload field.
        """
        self._client.delete(
            collection_name=self._collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="project",
                            match=qdrant_models.MatchValue(value=project),
                        )
                    ]
                )
            ),
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        project: str | None = None,
    ) -> list[dict]:
        """
        Semantic search. Optionally filter to a single project.
        Returns list of payload dicts with an added `score` field.
        """
        query_filter = None
        if project:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="project",
                        match=qdrant_models.MatchValue(value=project),
                    )
                ]
            )

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {**hit.payload, "score": round(hit.score, 4)}
            for hit in results
        ]

    def list_projects(self) -> list[str]:
        """Return distinct project names currently indexed in the collection."""
        # Scroll through all points and collect unique project names.
        # Acceptable for MVP — replace with a facet query when collection grows large.
        projects: set[str] = set()
        offset = None

        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=None,
                limit=250,
                offset=offset,
                with_payload=["project"],
                with_vectors=False,
            )
            for record in records:
                if record.payload and "project" in record.payload:
                    projects.add(record.payload["project"])
            if next_offset is None:
                break
            offset = next_offset

        return sorted(projects)
