import uuid
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from repo_knowledge.chunker import Chunk
from repo_knowledge.config import (
    EMBEDDING_DIM, EMBEDDING_MODEL, QDRANT_COLLECTION, QDRANT_URL,
    SEARCH_SCORE_THRESHOLD,
)


class Store:
    def __init__(self, url=QDRANT_URL, collection=QDRANT_COLLECTION,
                 embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL):
        self._url = url
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._embedding_dim = embedding_dim
        self._embedding_model = embedding_model
        self._collection_ready = False

    def _ensure_collection(self) -> None:
        """
        Create collection if it doesn't already exist.
        Called lazily on first read/write — not at construction time.
        Raises RuntimeError with a clean message if Qdrant is unreachable.
        """
        if self._collection_ready:
            return
        try:
            existing = {c.name for c in self._client.get_collections().collections}
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Qdrant at {self._url}. "
                "Check that Qdrant is running and reachable via Tailscale/LAN."
            ) from e
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qdrant_models.VectorParams(
                    size=self._embedding_dim,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
        self._collection_ready = True

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")
        self._ensure_collection()
        now = datetime.now(timezone.utc).isoformat()
        points = [
            qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "project": chunk.project, "path": chunk.path,
                    "language": chunk.language, "chunk_type": chunk.chunk_type,
                    "symbol": chunk.symbol, "content": chunk.content,
                    "start_line": chunk.start_line, "end_line": chunk.end_line,
                    "embedding_model": self._embedding_model,
                    "indexed_at": now,
                    # New fields — old chunks without these remain queryable
                    "content_hash": chunk.content_hash,
                    "file_mtime": chunk.file_mtime,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(collection_name=self._collection, points=points[i: i + batch_size])

    def delete_project(self, project: str) -> None:
        self._ensure_collection()
        self._client.delete(
            collection_name=self._collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[qdrant_models.FieldCondition(
                        key="project", match=qdrant_models.MatchValue(value=project),
                    )]
                )
            ),
        )

    def delete_file(self, project: str, rel_path: str) -> None:
        """Delete all indexed chunks for a specific file within a project."""
        self._ensure_collection()
        self._client.delete(
            collection_name=self._collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="project", match=qdrant_models.MatchValue(value=project),
                        ),
                        qdrant_models.FieldCondition(
                            key="path", match=qdrant_models.MatchValue(value=rel_path),
                        ),
                    ]
                )
            ),
        )

    def get_indexed_file_hashes(self, project: str) -> dict[str, str]:
        """
        Return {rel_path: content_hash} for every file already indexed under project.
        Files indexed before Issue #4 (no content_hash in payload) are mapped to "".
        This allows incremental reindex to treat them as changed and re-embed them.
        """
        self._ensure_collection()
        result: dict[str, str] = {}
        offset = None
        project_filter = qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(
                key="project", match=qdrant_models.MatchValue(value=project),
            )]
        )
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=project_filter,
                limit=250,
                offset=offset,
                with_payload=["path", "content_hash"],
                with_vectors=False,
            )
            for record in records:
                if record.payload:
                    path = record.payload.get("path", "")
                    content_hash = record.payload.get("content_hash", "")  # "" for old chunks
                    if path:
                        result[path] = content_hash
            if next_offset is None:
                break
            offset = next_offset
        return result

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        project: str | None = None,
        score_threshold: float = SEARCH_SCORE_THRESHOLD,
    ) -> list[dict]:
        self._ensure_collection()
        query_filter = None
        if project:
            query_filter = qdrant_models.Filter(
                must=[qdrant_models.FieldCondition(
                    key="project", match=qdrant_models.MatchValue(value=project),
                )]
            )
        # Fetch 2x top_k to allow content_hash deduplication and score filtering
        fetch_k = top_k * 2
        results = self._client.search(
            collection_name=self._collection, query_vector=query_vector,
            limit=fetch_k, query_filter=query_filter, with_payload=True,
        )

        seen_hashes: set[str] = set()
        deduped: list[dict] = []
        for hit in results:
            if hit.score < score_threshold:
                continue
            payload = hit.payload or {}
            content_hash = payload.get("content_hash", "")
            # Dedup by content_hash; fall through (no dedup) when hash is absent
            if content_hash and content_hash in seen_hashes:
                continue
            if content_hash:
                seen_hashes.add(content_hash)
            deduped.append({**payload, "score": round(hit.score, 4)})
            if len(deduped) >= top_k:
                break

        return deduped

    def list_projects(self) -> list[str]:
        self._ensure_collection()
        projects: set[str] = set()
        offset = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection, scroll_filter=None,
                limit=250, offset=offset, with_payload=["project"], with_vectors=False,
            )
            for record in records:
                if record.payload and "project" in record.payload:
                    projects.add(record.payload["project"])
            if next_offset is None:
                break
            offset = next_offset
        return sorted(projects)
