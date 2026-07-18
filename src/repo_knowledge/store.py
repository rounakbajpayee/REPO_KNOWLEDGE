import uuid

from repo_knowledge.chunker import Chunk
from repo_knowledge.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    RERANK_FETCH_K,
    SEARCH_SCORE_THRESHOLD,
)
from repo_knowledge.postgres_store import PostgresStore


class Store:
    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        postgres_store=None,
    ):
        self._embedding_dim = embedding_dim
        self._embedding_model = embedding_model
        self._pg = postgres_store or PostgresStore()

    def health_check(self) -> bool:
        return self._pg.health_check()

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch"
            )

        # Pre-generate UUIDs
        chunk_uuids = [str(uuid.uuid4()) for _ in chunks]

        # Write to PostgreSQL relational database (Source of Truth)
        if chunks:
            # Heuristic to find dominant programming stack
            stack = "Python"
            langs = [c.language for c in chunks if c.language]
            # Ignore markdown and text if there are other languages present
            filtered_langs = [
                lang for lang in langs if lang.lower() not in {"markdown", "text"}
            ]
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
            file_chunks: dict[str, list[tuple[Chunk, str, list[float]]]] = {}
            for chunk, cuuid, vector in zip(chunks, chunk_uuids, vectors):
                file_chunks.setdefault(chunk.path, []).append((chunk, cuuid, vector))

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
                just_vectors = [item[2] for item in c_list]
                self._pg.upsert_chunks(
                    file_id,
                    chunks[0].project,
                    path,
                    just_chunks,
                    just_uuids,
                    just_vectors,
                )

    def delete_project(self, project: str) -> None:
        self._pg.delete_project(project)

    def delete_file(self, project: str, rel_path: str) -> None:
        """Delete all indexed chunks for a specific file within a project."""
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
        """Retrieve candidate chunks using pgvector cosine similarity.

        When *query_text* is provided, performs a hybrid Postgres BM25 + pgvector
        search fused via RRF. Otherwise, runs a pure dense vector search.
        """
        # ── Pure dense vector search path ─────────────────────────────────────────
        if not query_text:
            fetch_k = top_k * 2

            vector_hits = self._pg.search_vector(
                query_vector=query_vector, project=project, limit=fetch_k
            )

            seen_hashes: set[str] = set()
            candidates: list[dict] = []
            for hit in vector_hits:
                if hit["score"] < score_threshold:
                    continue
                content_hash = hit.get("content_hash", "")
                if content_hash and content_hash in seen_hashes:
                    continue
                if content_hash:
                    seen_hashes.add(content_hash)
                candidates.append(hit)
                if len(candidates) >= top_k:
                    break
            return candidates

        # ── Hybrid search path (BM25 + pgvector via RRF) ────────────────────────────
        fetch_k = max(top_k * 2, RERANK_FETCH_K)

        vector_hits = self._pg.search_vector(
            query_vector=query_vector, project=project, limit=fetch_k
        )

        # Build: chunk_id → payload dict with cosine score (above threshold)
        vector_by_id: dict[str, dict] = {}
        for rank, hit in enumerate(vector_hits):
            if hit["score"] < score_threshold:
                continue
            payload = hit
            payload["_vector_rank"] = rank
            vector_by_id[str(hit["id"])] = payload

        bm25_by_id: dict[str, dict] = {}
        try:
            bm25_rows = self._pg.search_bm25(query_text, project=project, limit=fetch_k)
            for rank, row in enumerate(bm25_rows):
                row["_bm25_rank"] = rank
                bm25_by_id[row["id"]] = row
        except Exception:
            pass  # BM25 failure degrades gracefully — Vector results still returned

        all_ids = set(vector_by_id) | set(bm25_by_id)
        fused = _rrf_fuse(all_ids, vector_by_id, bm25_by_id)

        # Build final payload list, dedup by content_hash, truncate to fetch_k
        seen_hashes = set()
        candidates = []
        for chunk_id, rrf_score in fused:
            payload = vector_by_id.get(chunk_id) or bm25_by_id.get(chunk_id, {})
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
        """Return indexed project names."""
        try:
            names = self._pg.get_project_names()
            if names:
                return names
        except Exception as e:
            import traceback, sys
            print(f"DEBUG Error in store.list_projects: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        return []

    def get_chunks_for_path(self, project: str, rel_path: str) -> list[dict]:
        # get from pg directly
        with self._pg._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, project, path, language, chunk_type, symbol, content, 
                        start_line, end_line
                    FROM chunks
                    WHERE project = %s AND path = %s
                """,
                    (project, rel_path),
                )

                rows = cur.fetchall()

        return [
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
            }
            for row in rows
        ]


def _rrf_fuse(
    all_ids: set[str],
    vector_by_id: dict[str, dict],
    bm25_by_id: dict[str, dict],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over two ranked lists.

    RRF score = 1/(k + rank_vector) + 1/(k + rank_bm25)
    k=60 is the standard value from the original 2009 paper.
    Chunks present in only one list receive a large rank penalty (len+1)
    for the missing list, keeping them eligible but ranked below dual-list hits.

    Returns: sorted list of (chunk_id, rrf_score) descending.
    """
    # Build rank lookup (0-indexed → 1-indexed below)
    vector_ranks = {cid: d["_vector_rank"] for cid, d in vector_by_id.items()}
    bm25_ranks = {cid: d["_bm25_rank"] for cid, d in bm25_by_id.items()}

    scored: list[tuple[str, float]] = []
    for cid in all_ids:
        rrf = 0.0
        if cid in vector_ranks:
            rrf += 1.0 / (k + vector_ranks[cid] + 1)
        if cid in bm25_ranks:
            rrf += 1.0 / (k + bm25_ranks[cid] + 1)
        scored.append((cid, rrf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
