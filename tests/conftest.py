import os

import psycopg2
import pytest

from repo_knowledge.postgres_store import PostgresStore


@pytest.fixture(scope="session")
def pg_connection():
    # Connect to the database specified by environment variables
    conn = psycopg2.connect(
        host=os.getenv("TEST_PG_HOST", "localhost"),
        port=int(os.getenv("TEST_PG_PORT", "5432")),
        user=os.getenv("TEST_PG_USER", "postgres"),
        password=os.getenv("TEST_PG_PASSWORD", ""),
        database=os.getenv("TEST_PG_DB", "repo_knowledge_test"),
    )
    conn.autocommit = False

    # Instantiate store and use its _ensure_tables via this connection
    # to create tables if they don't exist yet in the test DB
    # We do it by passing the connection directly
    store = PostgresStore(
        host=os.getenv("TEST_PG_HOST", "localhost"),
        port=int(os.getenv("TEST_PG_PORT", "5432")),
        user=os.getenv("TEST_PG_USER", "postgres"),
        password=os.getenv("TEST_PG_PASSWORD", ""),
        database=os.getenv("TEST_PG_DB", "repo_knowledge_test"),
        connection=conn
    )
    # The _ensure_tables method will run the DDL queries
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
