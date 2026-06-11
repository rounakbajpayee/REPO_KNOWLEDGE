import os

import psycopg2
import pytest

from repo_knowledge.postgres_store import PostgresStore

# Defaults match the postgres service defined in .github/workflows/ci.yml.
# Override via env vars for local development.
_PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
_PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
_PG_USER = os.getenv("POSTGRES_USER", "postgres")
_PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
_PG_DB = os.getenv("POSTGRES_DB", "repo_knowledge_test")


@pytest.fixture(scope="session")
def pg_connection():
    conn = psycopg2.connect(
        host=_PG_HOST,
        port=_PG_PORT,
        user=_PG_USER,
        password=_PG_PASSWORD,
        database=_PG_DB,
    )
    conn.autocommit = False

    # Run migrations once for the session via a fresh store instance.
    store = PostgresStore(
        host=_PG_HOST,
        port=_PG_PORT,
        user=_PG_USER,
        password=_PG_PASSWORD,
        database=_PG_DB,
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
