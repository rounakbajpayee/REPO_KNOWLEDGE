"""
postgres_store.py — PostgreSQL relational storage manager for REPO_KNOWLEDGE.

Uses a ThreadedConnectionPool (min=2, max=10) to eliminate the ~80ms per-call
TCP handshake overhead that bare psycopg2.connect() incurs over a LAN connection.
Connections are returned to the pool after each operation.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
from psycopg2 import pool as pgpool
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import Json, execute_values

from repo_knowledge.config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)


class PostgresStore:
    def __init__(
        self,
        host: str = POSTGRES_HOST,
        port: int = POSTGRES_PORT,
        user: str = POSTGRES_USER,
        password: str = POSTGRES_PASSWORD,
        database: str = POSTGRES_DB,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._db = database
        self._initialized = False
        self._pool: pgpool.ThreadedConnectionPool | None = None
        self._pool_lock = threading.Lock()

    def _create_database_if_not_exists(self) -> None:
        """Connect to default 'postgres' database to check/create the target database."""
        try:
            conn = psycopg2.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database="postgres",
                connect_timeout=3,
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self._db,))
                if not cur.fetchone():
                    cur.execute(f'CREATE DATABASE "{self._db}"')
            conn.close()
        except Exception:
            pass  # Fall through if we can't create or it already exists

    def _ensure_tables(self) -> None:
        """Run DDL queries to verify and construct the relational schema."""
        if self._initialized:
            return

        self._create_database_if_not_exists()

        try:
            # We don't import at module level to avoid failure if alembic is not installed
            import os

            from alembic.config import Config

            from alembic import command  # type: ignore[attr-defined]
        except ImportError as e:
            raise RuntimeError(
                "Alembic is not installed but is required for database migrations. "
                "Please run `pip install -r requirements.txt`."
            ) from e

        try:
            # Locate alembic.ini from the package root
            pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            alembic_ini_path = os.path.join(pkg_root, "alembic.ini")

            alembic_cfg = Config(alembic_ini_path)

            # We pass the db credentials explicitly if needed, but env.py reads from env anyway.
            # Running upgrade head directly
            command.upgrade(alembic_cfg, "head")
        except Exception as e:
            raise RuntimeError(
                f"Failed to run database migrations against {self._host}:{self._port}. "
                f"Check credentials and ensure database is running. Error: {e}"
            )

        self._initialized = True

    def health_check(self) -> bool:
        """Returns True if Postgres is reachable, False otherwise."""
        try:
            conn = psycopg2.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._db,
                connect_timeout=2,
            )
            conn.close()
            return True
        except Exception:
            # Try to connect to default database if target db is not created yet
            try:
                conn = psycopg2.connect(
                    host=self._host,
                    port=self._port,
                    user=self._user,
                    password=self._password,
                    database="postgres",
                    connect_timeout=2,
                )
                conn.close()
                return True
            except Exception:
                return False

    def _init_pool(self) -> None:
        """Initialise the connection pool (called lazily, once per process)."""
        with self._pool_lock:
            if self._pool is not None:
                return
            try:
                self._pool = pgpool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=10,
                    host=self._host,
                    port=self._port,
                    user=self._user,
                    password=self._password,
                    dbname=self._db,
                    connect_timeout=5,
                )
            except Exception:
                self._pool = None  # Will fall back to bare connect

    @contextmanager
    def _get_connection(self) -> Iterator[psycopg2.extensions.connection]:
        """Yield a connection from the pool (or a bare connect if pool unavailable)."""
        self._ensure_tables()
        if self._pool is None:
            self._init_pool()

        if self._pool is not None:
            conn = self._pool.getconn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._pool.putconn(conn)
        else:
            # Fallback: bare connection (original behaviour)
            conn = psycopg2.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._db,
                connect_timeout=5,
            )
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def close_pool(self) -> None:
        """Close all pool connections. Call on application shutdown."""
        with self._pool_lock:
            if self._pool:
                self._pool.closeall()
                self._pool = None

    def upsert_project(self, name: str, stack: str) -> int:
        """Insert or update project metadata, returning project database ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projects (name, stack, last_indexed_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (name) DO UPDATE 
                    SET stack = EXCLUDED.stack, last_indexed_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """,
                    (name, stack),
                )
                return cur.fetchone()[0]

    def register_file(
        self, project_id: int, path: str, content_hash: str, file_mtime: float
    ) -> int:
        """Insert or update file hash state, returning file database ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO files (project_id, path, content_hash, file_mtime, last_indexed_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (project_id, path) DO UPDATE
                    SET content_hash = EXCLUDED.content_hash,
                        file_mtime = EXCLUDED.file_mtime,
                        last_indexed_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """,
                    (project_id, path, content_hash, file_mtime),
                )
                return cur.fetchone()[0]

    def delete_file(self, project_name: str, path: str) -> None:
        """Delete file and all its associated chunks recursively (via foreign keys)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM files 
                    WHERE path = %s AND project_id = (
                        SELECT id FROM projects WHERE name = %s
                    );
                """,
                    (path, project_name),
                )

    def delete_project(self, project_name: str) -> None:
        """Delete project registry and all associated files/chunks recursively."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM projects WHERE name = %s", (project_name,))

    def upsert_chunks(
        self, file_id: int, project: str, path: str, chunks: list, chunk_uuids: list[str]
    ) -> None:
        """Save raw chunk text records transactionally in the database."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for chunk, cuuid in zip(chunks, chunk_uuids):
                    cur.execute(
                        """
                        INSERT INTO chunks
                        (id, file_id, project, path, language,
                        chunk_type, symbol, content, start_line, end_line)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET content = EXCLUDED.content,
                            start_line = EXCLUDED.start_line,
                            end_line = EXCLUDED.end_line;
                    """,
                        (
                            cuuid,
                            file_id,
                            project,
                            path,
                            chunk.language,
                            chunk.chunk_type,
                            chunk.symbol,
                            chunk.content,
                            chunk.start_line,
                            chunk.end_line,
                        ),
                    )

    def get_indexed_file_hashes(self, project_name: str) -> dict[str, str]:
        """Return a mapping of {file_path: content_hash} stored in PostgreSQL."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT path, content_hash 
                    FROM files 
                    WHERE project_id = (SELECT id FROM projects WHERE name = %s);
                """,
                    (project_name,),
                )
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_indexed_file_mtimes(self, project_name: str) -> dict[str, float]:
        """Return a mapping of {file_path: file_mtime} stored in PostgreSQL."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT path, file_mtime 
                    FROM files 
                    WHERE project_id = (SELECT id FROM projects WHERE name = %s);
                """,
                    (project_name,),
                )
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_all_chunks(self) -> list[dict]:
        """Return all code chunks stored in the relational database."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, project, path, language,
                           chunk_type, symbol, content, start_line, end_line
                    FROM chunks;
                """)
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
                    for row in cur.fetchall()
                ]

    def log_decision(
        self,
        topic: str,
        entry_name: str,
        description: str,
        rationale: str,
        options_considered: list | None,
    ) -> None:
        """Save a decision card entry in PostgreSQL."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO decision_logs
                    (topic, entry_name, description, rationale, options_considered, logged_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP);
                """,
                    (
                        topic,
                        entry_name,
                        description,
                        rationale,
                        Json(options_considered) if options_considered else None,
                    ),
                )

    def get_decision_history(
        self, topic: str, limit: int = 3, full_history: bool = False
    ) -> list[dict]:
        """Query decision logs from PostgreSQL."""
        query = """
            SELECT topic, entry_name, description, rationale, options_considered, logged_at 
            FROM decision_logs 
            WHERE topic = %s 
            ORDER BY logged_at ASC
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (topic,))
                rows = cur.fetchall()

        entries = [
            {
                "topic": r[0],
                "name": r[1],
                "description": r[2],
                "rationale": r[3],
                "options_considered": r[4],
                "logged_at": r[5].isoformat(),
            }
            for r in rows
        ]
        if not full_history and limit > 0:
            return entries[-limit:]
        return entries

    def log_audit_trace(
        self,
        ts_str: str,
        trace_id: str | None,
        event: str,
        severity: str,
        subsystem: str,
        duration_ms: int | None,
        payload: dict | None,
    ) -> None:
        """Insert a system trace record transactionally in PostgreSQL."""
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(timezone.utc)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_logs
                    (ts, trace_id, event, severity, subsystem, duration_ms, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                    (
                        ts,
                        trace_id,
                        event,
                        severity,
                        subsystem,
                        duration_ms,
                        Json(payload) if payload else None,
                    ),
                )

    def list_projects(self) -> list[dict]:
        """Return all projects from database."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name, stack, last_indexed_at FROM projects ORDER BY name ASC;")
                return [
                    {"name": r[0], "stack": r[1], "last_indexed_at": r[2].isoformat()}
                    for r in cur.fetchall()
                ]

    def get_project_names(self) -> list[str]:
        """Return only project names — fast single-column query used by store.list_projects()."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM projects ORDER BY name ASC;")
                return [r[0] for r in cur.fetchall()]

    def search_bm25(
        self,
        query_text: str,
        project: str | None = None,
        limit: int = 40,
    ) -> list[dict]:
        """Full-text BM25 search via Postgres tsvector GIN index.

        Returns rows scored by ts_rank_cd, normalised to [0, 1] against the
        maximum rank in the result set so scores are RRF-compatible with
        cosine similarity scores from Qdrant.

        Args:
            query_text: Raw query string (converted to plainto_tsquery internally).
            project:    Optional project filter.
            limit:      Maximum candidate rows to return.

        Returns:
            List of dicts with: id, project, path, language, chunk_type,
            symbol, content, start_line, end_line, bm25_score.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                base_sql = """
                    SELECT
                        c.id::text,
                        c.project,
                        c.path,
                        c.language,
                        c.chunk_type,
                        c.symbol,
                        c.content,
                        c.start_line,
                        c.end_line,
                        ts_rank_cd(c.content_tsv, query) AS rank
                    FROM chunks c,
                         plainto_tsquery('english', %s) query
                    WHERE c.content_tsv @@ query
                """
                params: list = [query_text]
                if project:
                    base_sql += " AND c.project = %s"
                    params.append(project)
                base_sql += " ORDER BY rank DESC LIMIT %s;"
                params.append(limit)

                cur.execute(base_sql, params)
                rows = cur.fetchall()

        if not rows:
            return []

        max_rank = max(r[9] for r in rows) or 1.0
        return [
            {
                "id": r[0],
                "project": r[1],
                "path": r[2],
                "language": r[3],
                "chunk_type": r[4],
                "symbol": r[5],
                "content": r[6],
                "start_line": r[7],
                "end_line": r[8],
                "bm25_score": round(float(r[9]) / max_rank, 4),
            }
            for r in rows
        ]

    def log_audit_traces_batch(self, records: list[dict]) -> None:
        """Bulk-insert multiple audit trace records in a single round-trip.

        Each record must have: ts_str, trace_id, event, severity, subsystem,
        duration_ms (optional), payload (optional dict).
        """
        if not records:
            return

        rows = []
        for rec in records:
            try:
                ts = datetime.fromisoformat(rec["ts_str"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                ts = datetime.now(timezone.utc)
            rows.append(
                (
                    ts,
                    rec.get("trace_id"),
                    rec.get("event", ""),
                    rec.get("severity", "INFO"),
                    rec.get("subsystem", "unknown"),
                    rec.get("duration_ms"),
                    Json(rec["payload"]) if rec.get("payload") else None,
                )
            )

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO audit_logs
                    (ts, trace_id, event, severity, subsystem, duration_ms, payload)
                    VALUES %s;
                    """,
                    rows,
                )

    def get_audit_logs(self, limit: int = 100, severity: str | None = None) -> list[dict]:
        """Query the structured trace logs."""
        query = (
            "SELECT ts, trace_id, event, severity, subsystem, duration_ms, payload FROM audit_logs "
        )
        params: list[str | int] = []
        if severity:
            query += "WHERE severity = %s "
            params.append(severity)
        query += "ORDER BY id DESC LIMIT %s;"
        params.append(limit)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                return [
                    {
                        "ts": r[0].isoformat(),
                        "trace_id": r[1],
                        "event": r[2],
                        "severity": r[3],
                        "subsystem": r[4],
                        "duration_ms": r[5],
                        "payload": r[6],
                    }
                    for r in cur.fetchall()
                ]
