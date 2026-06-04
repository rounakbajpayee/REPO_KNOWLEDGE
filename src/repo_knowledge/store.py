import uuid
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from repo_knowledge.chunker import Chunk
from repo_knowledge.config import EMBEDDING_DIM, EMBEDDING_MODEL, QDRANT_COLLECTION, QDRANT_URL


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

    def search(self, query_vector: list[float], top_k: int, project: str | None = None) -> list[dict]:
        self._ensure_collection()
        query_filter = None
        if project:
            query_filter = qdrant_models.Filter(
                must=[qdrant_models.FieldCondition(
                    key="project", match=qdrant_models.MatchValue(value=project),
                )]
            )
        results = self._client.search(
            collection_name=self._collection, query_vector=query_vector,
            limit=top_k, query_filter=query_filter, with_payload=True,
        )
        return [{**hit.payload, "score": round(hit.score, 4)} for hit in results]

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
