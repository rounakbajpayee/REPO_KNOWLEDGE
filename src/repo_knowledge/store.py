import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from repo_knowledge.chunker import Chunk
from repo_knowledge.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANK_FETCH_K,
    SEARCH_SCORE_THRESHOLD,
)
from repo_knowledge.postgres_store import PostgresStore


class Store:
    def __init__(
        self,
        url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        postgres_store=None,
    ):
        self._url = url
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._embedding_dim = embedding_dim
        self._embedding_model = embedding_model
        self._collection_ready = False
        self._pg = postgres_store or PostgresStore()

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

        # Pre-generate UUIDs so they match between Qdrant and PostgreSQL
        chunk_uuids = [str(uuid.uuid4()) for _ in chunks]

        points = [
            qdrant_models.PointStruct(
                id=cuuid,
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
                    "content_hash": chunk.content_hash,
                    "file_mtime": chunk.file_mtime,
                },
            )
            for chunk, vector, cuuid in zip(chunks, vectors, chunk_uuids)
        ]
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(collection_name=self._collection, points=points[i : i + batch_size])

        # Write to PostgreSQL relational database (Source of Truth)
        if chunks:
            # Heuristic to find dominant programming stack
            stack = "Python"
            langs = [c.language for c in chunks if c.language]
            # Ignore markdown and text if there are other languages present
            filtered_langs = [lang for lang in langs if lang.lower() not in {"markdown", "text"}]
            if filtered_langs:
                stack_lang = max(set(filtered_langs), key=filtered_langs.count)
            elif langs:
                stack_lang = max(set(langs), key=langs.count)
            else:
                stack_lang = "python"

            # Normalize language names to human-readable stack names
            if stack_lang.lower() in {"python"}:
                stack = "Python"
            elif stack_lang.lower() in {"javascript", "typescript"}:
                stack = "Node.js"
            elif stack_lang.lower() in {"go"}:
                stack = "Go"
            elif stack_lang.lower() in {"rust"}:
                stack = "Rust"
            elif stack_lang.lower() in {"yaml", "yml"}:
                stack = "YAML"
            elif stack_lang.lower() in {"json"}:
                stack = "JSON"
            else:
                stack = stack_lang.title()

            project_id = self._pg.upsert_project(chunks[0].project, stack)

            # Group chunks by file path
            file_chunks: dict[str, list[tuple[Chunk, str]]] = {}
            for chunk, cuuid in zip(chunks, chunk_uuids):
                file_chunks.setdefault(chunk.path, []).append((chunk, cuuid))

            for path, c_list in file_chunks.items():
                first_chunk = c_list[0][0]
                file_id = self._pg.register_file(
                    project_id=project_id,
                    path=path,
                    content_hash=first_chunk.content_hash,
                    file_mtime=first_chunk.file_mtime,
                )
                just_chunks = [item[0] for item in c_list]
                just_uuids = [item[1] for item in c_list]
                self._pg.upsert_chunks(file_id, chunks[0].project, path, just_chunks, just_uuids)

    def delete_project(self, project: str) -> None:
        self._ensure_collection()
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
        self._pg.delete_project(project)

    def delete_file(self, project: str, rel_path: str) -> None:
        """Delete all indexed chunks for a specific file within a project."""
        self._ensure_collection()
        self._client.delete(
            collection_name=self._collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="project",
                            match=qdrant_models.MatchValue(value=project),
                        ),
                        qdrant_models.FieldCondition(
                            key="path",
                            match=qdrant_models.MatchValue(value=rel_path),
                        ),
                    ]
                )
            ),
        )
        self._pg.delete_file(project, rel_path)

    def get_indexed_file_hashes(self, project: str) -> dict[str, str]:
        """
        Return {rel_path: content_hash} for every file already indexed under project.
        Loads directly from PostgreSQL.
        """
        return self._pg.get_indexed_file_hashes(project)

    def get_indexed_file_mtimes(self, project: str) -> dict[str, float]:
        """
        Return {rel_path: file_mtime} for every file already indexed under project.
        Loads directly from PostgreSQL.
        """
        return self._pg.get_indexed_file_mtimes(project)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        project: str | None = None,
        score_threshold: float = SEARCH_SCORE_THRESHOLD,
        query_text: str | None = None,
    ) -> list[dict]:
        """Retrieve candidate chunks using Qdrant cosine similarity.

        When *query_text* is provided, performs a hybrid Postgres BM25 + Qdrant
        search fused via RRF. Otherwise, runs a pure dense vector search.
        """
        self._ensure_collection()

        # ── Pure dense vector search path ─────────────────────────────────────────
        if not query_text:
            fetch_k = top_k * 2
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
            qdrant_hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=fetch_k,
                query_filter=query_filter,
                with_payload=True,
            )

            seen_hashes: set[str] = set()
            candidates: list[dict] = []
            for hit in qdrant_hits:
                if hit.score < score_threshold:
                    continue
                payload = dict(hit.payload or {})
                content_hash = payload.get("content_hash", "")
                if content_hash and content_hash in seen_hashes:
                    continue
                if content_hash:
                    seen_hashes.add(content_hash)
                payload["score"] = round(hit.score, 4)
                candidates.append(payload)
                if len(candidates) >= top_k:
                    break
            return candidates

        # ── Hybrid search path (BM25 + Qdrant via RRF) ────────────────────────────
        fetch_k = max(top_k * 2, RERANK_FETCH_K)

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
        qdrant_hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=fetch_k,
            query_filter=query_filter,
            with_payload=True,
        )

        # Build: chunk_id → payload dict with cosine score (above threshold)
        qdrant_by_id: dict[str, dict] = {}
        for rank, hit in enumerate(qdrant_hits):
            if hit.score < score_threshold:
                continue
            payload = dict(hit.payload or {})
            payload["score"] = round(hit.score, 4)
            payload["_qdrant_rank"] = rank
            qdrant_by_id[str(hit.id)] = payload

        bm25_by_id: dict[str, dict] = {}
        try:
            bm25_rows = self._pg.search_bm25(query_text, project=project, limit=fetch_k)
            for rank, row in enumerate(bm25_rows):
                row["_bm25_rank"] = rank
                bm25_by_id[row["id"]] = row
        except Exception:
            pass  # BM25 failure degrades gracefully — Qdrant results still returned

        all_ids = set(qdrant_by_id) | set(bm25_by_id)
        fused = _rrf_fuse(all_ids, qdrant_by_id, bm25_by_id)

        # Build final payload list, dedup by content_hash, truncate to fetch_k
        seen_hashes = set()
        candidates = []
        for chunk_id, rrf_score in fused:
            payload = qdrant_by_id.get(chunk_id) or bm25_by_id.get(chunk_id, {})
            content_hash = payload.get("content_hash", "")
            if content_hash and content_hash in seen_hashes:
                continue
            if content_hash:
                seen_hashes.add(content_hash)
            item = {k: v for k, v in payload.items() if not k.startswith("_")}
            item["score"] = round(rrf_score, 4)
            candidates.append(item)
            if len(candidates) >= fetch_k:
                break

        return candidates

    def list_projects(self) -> list[str]:
        """Return indexed project names.

        Primary source: PostgreSQL (single SELECT — fast, Qdrant-independent).
        Fallback: Qdrant scroll (original behaviour, used only if Postgres is down).
        """
        try:
            names = self._pg.get_project_names()
            if names:
                return names
        except Exception:
            pass

        # Qdrant fallback — full scroll (O(n) over all vectors, kept for resilience)
        self._ensure_collection()
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

    def get_chunks_for_path(self, project: str, rel_path: str) -> list[dict]:
        self._ensure_collection()
        chunks: list[dict] = []
        offset = None
        filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="project",
                    match=qdrant_models.MatchValue(value=project),
                ),
                qdrant_models.FieldCondition(
                    key="path",
                    match=qdrant_models.MatchValue(value=rel_path),
                ),
            ]
        )
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=filter,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if record.payload:
                    chunks.append(record.payload)
            if next_offset is None:
                break
            offset = next_offset
        return chunks


def _rrf_fuse(
    all_ids: set[str],
    qdrant_by_id: dict[str, dict],
    bm25_by_id: dict[str, dict],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over two ranked lists.

    RRF score = 1/(k + rank_qdrant) + 1/(k + rank_bm25)
    k=60 is the standard value from the original 2009 paper.
    Chunks present in only one list receive a large rank penalty (len+1)
    for the missing list, keeping them eligible but ranked below dual-list hits.

    Returns: sorted list of (chunk_id, rrf_score) descending.
    """
    # Build rank lookup (0-indexed → 1-indexed below)
    qdrant_ranks = {cid: d["_qdrant_rank"] for cid, d in qdrant_by_id.items()}
    bm25_ranks = {cid: d["_bm25_rank"] for cid, d in bm25_by_id.items()}

    scored: list[tuple[str, float]] = []
    for cid in all_ids:
        rrf = 0.0
        if cid in qdrant_ranks:
            rrf += 1.0 / (k + qdrant_ranks[cid] + 1)
        if cid in bm25_ranks:
            rrf += 1.0 / (k + bm25_ranks[cid] + 1)
        scored.append((cid, rrf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
