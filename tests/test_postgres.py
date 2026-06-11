"""
test_postgres.py — Unit tests for PostgresStore using a real test database.
"""

from unittest.mock import patch

from repo_knowledge.postgres_store import PostgresStore


def test_get_project_names(db):
    """get_project_names must return only the name column."""
    store = PostgresStore(connection=db)

    # Insert some dummy projects
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO projects (name, stack) VALUES ('AI_LAB', 'python'), ('APRIL', 'python'), ('ECHO', 'python')"
        )

    names = store.get_project_names()

    assert set(names) == {"AI_LAB", "APRIL", "ECHO"}


def test_log_audit_traces_batch_empty_is_noop(db):
    """log_audit_traces_batch must be a no-op when given an empty list."""
    store = PostgresStore(connection=db)

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM audit_logs")
        initial_count = cur.fetchone()[0]

    store.log_audit_traces_batch([])

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM audit_logs")
        final_count = cur.fetchone()[0]

    assert initial_count == final_count


def test_log_audit_traces_batch_inserts_rows(db):
    """log_audit_traces_batch must call execute_values with all records."""
    store = PostgresStore(connection=db)
    records = [
        {
            "ts_str": "2026-06-05T00:00:00+00:00",
            "trace_id": "abc123",
            "event": "test_event",
            "severity": "INFO",
            "subsystem": "test",
            "duration_ms": 42,
            "payload": {"key": "val"},
        },
        {
            "ts_str": "2026-06-05T00:00:01+00:00",
            "trace_id": None,
            "event": "second_event",
            "severity": "WARNING",
            "subsystem": "store",
            "duration_ms": None,
            "payload": None,
        },
    ]
    store.log_audit_traces_batch(records)

    with db.cursor() as cur:
        cur.execute("SELECT event, severity, subsystem FROM audit_logs ORDER BY ts ASC")
        rows = cur.fetchall()

    assert len(rows) >= 2
    events = [r[0] for r in rows]
    assert "test_event" in events
    assert "second_event" in events


def test_postgres_store_initialization():
    store = PostgresStore(
        host="localhost", port=5434, user="test_user", password="test_password", database="test_db"
    )

    assert not store._initialized
    assert store._host == "localhost"
    assert store._db == "test_db"


def test_ensure_tables_runs_migrations(db):
    store = PostgresStore(connection=db)
    # The tables should already be ensured by the fixture,
    # but calling it again shouldn't fail and should mark initialized.
    # We must patch _create_database_if_not_exists or connect calls if it uses default hardcoded host when missing.
    # Wait, the store uses default env vars if not passed explicitly in test_ensure_tables_runs_migrations
    # Let's pass the host/port/etc so it can connect to the test db if it needs a bare connection.
    import os
    store = PostgresStore(
        host=os.getenv("TEST_PG_HOST", "localhost"),
        port=int(os.getenv("TEST_PG_PORT", "5432")),
        user=os.getenv("TEST_PG_USER", "postgres"),
        password=os.getenv("TEST_PG_PASSWORD", ""),
        database=os.getenv("TEST_PG_DB", "repo_knowledge_test"),
        connection=db
    )
    store._ensure_tables()
    assert store._initialized


def test_upsert_project(db):
    store = PostgresStore(connection=db)
    project_id = store.upsert_project("my_proj", "Python")

    with db.cursor() as cur:
        cur.execute("SELECT name, stack FROM projects WHERE id = %s", (project_id,))
        row = cur.fetchone()

    assert row is not None
    assert row[0] == "my_proj"
    assert row[1] == "Python"


def test_register_file(db):
    store = PostgresStore(connection=db)
    project_id = store.upsert_project("file_proj", "Python")

    file_id = store.register_file(project_id, "src/main.py", "hash123", 12345.6)

    with db.cursor() as cur:
        cur.execute("SELECT path, content_hash FROM files WHERE id = %s", (file_id,))
        row = cur.fetchone()

    assert row is not None
    assert row[0] == "src/main.py"
    assert row[1] == "hash123"


def test_delete_file(db):
    store = PostgresStore(connection=db)
    project_id = store.upsert_project("delete_proj", "Python")
    store.register_file(project_id, "src/main.py", "hash123", 12345.6)

    # Validate insertion
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM files WHERE project_id = %s", (project_id,))
        assert cur.fetchone()[0] == 1

    store.delete_file("delete_proj", "src/main.py")

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM files WHERE project_id = %s", (project_id,))
        assert cur.fetchone()[0] == 0


def test_upsert_chunks(db):
    class FakeChunk:
        def __init__(self, language, chunk_type, symbol, content, start_line, end_line):
            self.language = language
            self.chunk_type = chunk_type
            self.symbol = symbol
            self.content = content
            self.start_line = start_line
            self.end_line = end_line

    chunk = FakeChunk("python", "class", "MyClass", "class MyClass: pass", 1, 10)

    store = PostgresStore(connection=db)
    project_id = store.upsert_project("chunk_proj", "Python")
    file_id = store.register_file(project_id, "src/main.py", "hash123", 12345.6)

    import uuid
    test_uuid = str(uuid.uuid4())

    store.upsert_chunks(
        file_id=file_id,
        project="chunk_proj",
        path="src/main.py",
        chunks=[chunk],
        chunk_uuids=[test_uuid],
    )

    with db.cursor() as cur:
        cur.execute("SELECT id, symbol, content FROM chunks WHERE id = %s", (test_uuid,))
        row = cur.fetchone()

    assert row is not None
    assert str(row[0]) == test_uuid
    assert row[1] == "MyClass"
    assert row[2] == "class MyClass: pass"


def test_get_indexed_file_hashes(db):
    store = PostgresStore(connection=db)
    project_id = store.upsert_project("hash_proj", "Python")
    store.register_file(project_id, "src/main.py", "hash_main", 12345.6)
    store.register_file(project_id, "src/utils.py", "hash_utils", 12345.6)

    hashes = store.get_indexed_file_hashes("hash_proj")

    assert hashes == {"src/main.py": "hash_main", "src/utils.py": "hash_utils"}


def test_log_decision(db):
    store = PostgresStore(connection=db)
    store.log_decision(
        topic="embedding_model",
        entry_name="switch_to_qwen",
        description="Switch embedding model",
        rationale="Better accuracy",
        options_considered=[{"name": "mxbai", "status": "REJECTED"}],
    )

    with db.cursor() as cur:
        cur.execute(
            "SELECT entry_name, options_considered FROM decision_logs WHERE topic = 'embedding_model'"
        )
        row = cur.fetchone()

    assert row is not None
    assert row[0] == "switch_to_qwen"
    assert row[1] == [{"name": "mxbai", "status": "REJECTED"}]


def test_get_decision_history(db):
    store = PostgresStore(connection=db)
    store.log_decision(
        topic="embedding_model2",
        entry_name="switch_to_qwen",
        description="Switch",
        rationale="Rationale",
        options_considered=[{"name": "mxbai"}],
    )

    history = store.get_decision_history("embedding_model2", limit=3, full_history=False)

    assert len(history) == 1
    assert history[0]["topic"] == "embedding_model2"
    assert history[0]["name"] == "switch_to_qwen"
    assert history[0]["description"] == "Switch"
    assert history[0]["rationale"] == "Rationale"
    assert history[0]["options_considered"] == [{"name": "mxbai"}]
    assert "logged_at" in history[0]


def test_health_check_returns_true_when_ok(db):
    import os
    store = PostgresStore(
        host=os.getenv("TEST_PG_HOST", "localhost"),
        port=int(os.getenv("TEST_PG_PORT", "5432")),
        user=os.getenv("TEST_PG_USER", "postgres"),
        password=os.getenv("TEST_PG_PASSWORD", ""),
        database=os.getenv("TEST_PG_DB", "repo_knowledge_test"),
        connection=db
    )
    assert store.health_check() is True


def test_health_check_returns_false_on_failure():
    # Setup connection to fail completely
    with patch(
        "repo_knowledge.postgres_store.psycopg2.connect", side_effect=Exception("Connection failed")
    ):
        store = PostgresStore()
        assert store.health_check() is False
