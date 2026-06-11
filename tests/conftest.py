import os

import psycopg2
import pytest

from repo_knowledge.postgres_store import PostgresStore


@pytest.fixture(scope="session")
def pg_connection():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DB", "repo_knowledge_test"),
    )
    conn.autocommit = False

    # Run migrations via a fresh store instance (not using the injected connection)
    # so that the test DB schema is set up before any tests run.
    store = PostgresStore(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DB", "repo_knowledge_test"),
    )
    store._ensure_tables()

    yield conn
    conn.close()


@pytest.fixture
def db(pg_connection):
    with pg_connection.cursor() as cur:
        cur.execute("SAVEPOINT test_savepoint")
    yield pg_connection
    with pg_connection.cursor() as cur:
        cur.execute("ROLLBACK TO SAVEPOINT test_savepoint")
