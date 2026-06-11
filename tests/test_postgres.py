"""
test_postgres.py — Unit tests for PostgresStore using mocked psycopg2 connections.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from repo_knowledge.postgres_store import PostgresStore


@pytest.fixture
def mock_connect():
    # Patch both the pool and the bare connect so tests fully control DB access.
    # Pool is set to always fail → store falls back to bare psycopg2.connect (also mocked).
    with patch(
        "repo_knowledge.postgres_store.pgpool.ThreadedConnectionPool",
        side_effect=Exception("pool disabled in tests"),
    ):
        with patch("repo_knowledge.postgres_store.psycopg2.connect") as mock_conn_fn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            # Chained context manager returns
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.__enter__.return_value = mock_cursor

            mock_conn_fn.return_value = mock_conn
            yield mock_conn_fn, mock_conn, mock_cursor


def test_get_project_names(mock_connect):
    """get_project_names must return only the name column."""
    mock_conn_fn, mock_conn, mock_cursor = mock_connect
    mock_cursor.fetchall.return_value = [("AI_LAB",), ("APRIL",), ("ECHO",)]

    store = PostgresStore()
    names = store.get_project_names()

    assert names == ["AI_LAB", "APRIL", "ECHO"]
    exec_args = mock_cursor.execute.call_args[0]
    assert "SELECT name FROM projects" in exec_args[0]


def test_log_audit_traces_batch_empty_is_noop(mock_connect):
    """log_audit_traces_batch must be a no-op when given an empty list."""
    mock_conn_fn, mock_conn, mock_cursor = mock_connect

    store = PostgresStore()
    store.log_audit_traces_batch([])

    # No DB interaction should have occurred (beyond _ensure_tables)
    # confirm execute_values was not called for audit_logs
    executed = [str(c) for c in mock_cursor.execute.call_args_list]
    assert not any("INSERT INTO audit_logs" in e for e in executed)


def test_log_audit_traces_batch_inserts_rows(mock_connect):
    """log_audit_traces_batch must call execute_values with all records."""
    mock_conn_fn, mock_conn, mock_cursor = mock_connect

    with patch("repo_knowledge.postgres_store.execute_values") as mock_ev:
        store = PostgresStore()
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

        assert mock_ev.called
        rows_arg = mock_ev.call_args[0][2]
        assert len(rows_arg) == 2
        assert rows_arg[0][2] == "test_event"
        assert rows_arg[1][2] == "second_event"
        assert rows_arg[1][4] == "store"


def test_postgres_store_initialization(mock_connect):
    mock_conn_fn, mock_conn, mock_cursor = mock_connect

    # We instantiate PostgresStore. Its methods like _ensure_tables will be called lazily.
    store = PostgresStore(
        host="localhost", port=5434, user="test_user", password="test_password", database="test_db"
    )

    # Lazy initial state
    assert not store._initialized
    assert store._host == "localhost"
    assert store._db == "test_db"


def test_ensure_tables_runs_migrations(mock_connect):
    mock_conn_fn, mock_conn, mock_cursor = mock_connect

    store = PostgresStore()

    # Trigger database connection and DDL query execution
    store._ensure_tables()

    assert store._initialized
    # Verify that psycopg2.connect was called (once for checking DB, once for ensure tables)
    assert mock_conn_fn.call_count >= 2

    # Verify cursor executed CREATE TABLE statements
    executed_sql = [call[0][0].strip().lower() for call in mock_cursor.execute.call_args_list]
    assert any("create table if not exists projects" in sql for sql in executed_sql)
    assert any("create table if not exists files" in sql for sql in executed_sql)
    assert any("create table if not exists chunks" in sql for sql in executed_sql)
    assert any("create table if not exists decision_logs" in sql for sql in executed_sql)
    assert any("create table if not exists audit_logs" in sql for sql in executed_sql)


def test_upsert_project(mock_connect):
    mock_conn_fn, mock_conn, mock_cursor = mock_connect
    mock_cursor.fetchone.return_value = (42,)

    store = PostgresStore()
    project_id = store.upsert_project("my_proj", "Python")

    assert project_id == 42
    # Verify SQL execution
    exec_args = mock_cursor.execute.call_args[0]
    assert "INSERT INTO projects" in exec_args[0]
    assert exec_args[1] == ("my_proj", "Python")


def test_register_file(mock_connect):
    mock_conn_fn, mock_conn, mock_cursor = mock_connect
    mock_cursor.fetchone.return_value = (101,)

    store = PostgresStore()
    file_id = store.register_file(42, "src/main.py", "hash123", 12345.6)

    assert file_id == 101
    exec_args = mock_cursor.execute.call_args[0]
    assert "INSERT INTO files" in exec_args[0]
    assert exec_args[1] == (42, "src/main.py", "hash123", 12345.6)


def test_delete_file(mock_connect):
    mock_conn_fn, mock_conn, mock_cursor = mock_connect

    store = PostgresStore()
    store.delete_file("my_proj", "src/main.py")

    exec_args = mock_cursor.execute.call_args[0]
    assert "DELETE FROM files" in exec_args[0]
    assert exec_args[1] == ("src/main.py", "my_proj")


def test_upsert_chunks(mock_connect):
    mock_conn_fn, mock_conn, mock_cursor = mock_connect

    class FakeChunk:
        def __init__(self, language, chunk_type, symbol, content, start_line, end_line):
            self.language = language
            self.chunk_type = chunk_type
            self.symbol = symbol
            self.content = content
            self.start_line = start_line
            self.end_line = end_line

    chunk = FakeChunk("python", "class", "MyClass", "class MyClass: pass", 1, 10)

    store = PostgresStore()
    store.upsert_chunks(
        file_id=101, project="my_proj", path="src/main.py", chunks=[chunk], chunk_uuids=["uuid-xyz"]
    )

    exec_args = mock_cursor.execute.call_args[0]
    assert "INSERT INTO chunks" in exec_args[0]
    assert exec_args[1] == (
        "uuid-xyz",
        101,
        "my_proj",
        "src/main.py",
        "python",
        "class",
        "MyClass",
        "class MyClass: pass",
        1,
        10,
    )


def test_get_indexed_file_hashes(mock_connect):
    mock_conn_fn, mock_conn, mock_cursor = mock_connect
    mock_cursor.fetchall.return_value = [
        ("src/main.py", "hash_main"),
        ("src/utils.py", "hash_utils"),
    ]

    store = PostgresStore()
    hashes = store.get_indexed_file_hashes("my_proj")

    assert hashes == {"src/main.py": "hash_main", "src/utils.py": "hash_utils"}
    exec_args = mock_cursor.execute.call_args[0]
    assert "SELECT path, content_hash" in exec_args[0]
    assert exec_args[1] == ("my_proj",)


def test_log_decision(mock_connect):
    mock_conn_fn, mock_conn, mock_cursor = mock_connect

    store = PostgresStore()
    store.log_decision(
        topic="embedding_model",
        entry_name="switch_to_qwen",
        description="Switch embedding model",
        rationale="Better accuracy",
        options_considered=[{"name": "mxbai", "status": "REJECTED"}],
    )

    exec_args = mock_cursor.execute.call_args[0]
    assert "INSERT INTO decision_logs" in exec_args[0]
    assert exec_args[1][0] == "embedding_model"
    assert exec_args[1][1] == "switch_to_qwen"
    assert exec_args[1][2] == "Switch embedding model"
    assert exec_args[1][3] == "Better accuracy"
    # The last element is a Json wrapper
    assert exec_args[1][4].adapted == [{"name": "mxbai", "status": "REJECTED"}]


def test_get_decision_history(mock_connect):
    mock_conn_fn, mock_conn, mock_cursor = mock_connect
    now = datetime.now(timezone.utc)
    mock_cursor.fetchall.return_value = [
        ("embedding_model", "switch_to_qwen", "Switch", "Rationale", [{"name": "mxbai"}], now)
    ]

    store = PostgresStore()
    history = store.get_decision_history("embedding_model", limit=3, full_history=False)

    assert len(history) == 1
    assert history[0]["topic"] == "embedding_model"
    assert history[0]["name"] == "switch_to_qwen"
    assert history[0]["description"] == "Switch"
    assert history[0]["rationale"] == "Rationale"
    assert history[0]["options_considered"] == [{"name": "mxbai"}]
    assert history[0]["logged_at"] == now.isoformat()


def test_health_check_returns_true_when_ok(mock_connect):
    # Setup connection to succeed
    mock_conn_fn, mock_conn, mock_cursor = mock_connect

    store = PostgresStore()
    assert store.health_check() is True


def test_health_check_returns_false_on_failure():
    # Setup connection to fail completely
    with patch(
        "repo_knowledge.postgres_store.psycopg2.connect", side_effect=Exception("Connection failed")
    ):
        store = PostgresStore()
        assert store.health_check() is False
