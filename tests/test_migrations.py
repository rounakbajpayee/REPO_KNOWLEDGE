import os

import psycopg2
import pytest
from alembic.config import Config

from alembic import command
from repo_knowledge.config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)

# Reuse existing test db config. We use a dedicated fixture since tables are replaced.

@pytest.fixture
def tmp_pg():
    """Provides a clean test database by ensuring all tables are dropped beforehand."""
    # We use the config credentials which should point to the test db in CI
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )

    # Drop existing tables to start clean
    with conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS audit_logs CASCADE;")
            cur.execute("DROP TABLE IF EXISTS decision_logs CASCADE;")
            cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
            cur.execute("DROP TABLE IF EXISTS files CASCADE;")
            cur.execute("DROP TABLE IF EXISTS projects CASCADE;")
            cur.execute("DROP TABLE IF EXISTS alembic_version CASCADE;")

    conn.close()

    pkg_root = os.path.dirname(os.path.dirname(__file__))
    alembic_ini_path = os.path.join(pkg_root, "alembic.ini")
    alembic_cfg = Config(alembic_ini_path)

    yield alembic_cfg

    # Cleanup after test
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )

    with conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS audit_logs CASCADE;")
            cur.execute("DROP TABLE IF EXISTS decision_logs CASCADE;")
            cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
            cur.execute("DROP TABLE IF EXISTS files CASCADE;")
            cur.execute("DROP TABLE IF EXISTS projects CASCADE;")
            cur.execute("DROP TABLE IF EXISTS alembic_version CASCADE;")

    conn.close()


def test_upgrade_creates_all_tables(tmp_pg):
    alembic_cfg = tmp_pg
    command.upgrade(alembic_cfg, "head")

    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public';"
            )
            tables = {row[0] for row in cur.fetchall()}

    conn.close()

    assert "projects" in tables
    assert "files" in tables
    assert "chunks" in tables
    assert "decision_logs" in tables
    assert "audit_logs" in tables
    assert "alembic_version" in tables


def test_downgrade_drops_all_tables(tmp_pg):
    alembic_cfg = tmp_pg

    # First upgrade to head
    command.upgrade(alembic_cfg, "head")

    # Then downgrade to base
    command.downgrade(alembic_cfg, "base")

    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public';"
            )
            tables = {row[0] for row in cur.fetchall()}

    conn.close()

    # Assert tables are gone. alembic_version may remain.
    assert "projects" not in tables
    assert "files" not in tables
    assert "chunks" not in tables
    assert "decision_logs" not in tables
    assert "audit_logs" not in tables


def test_upgrade_is_idempotent(tmp_pg):
    alembic_cfg = tmp_pg

    # Run twice
    command.upgrade(alembic_cfg, "head")
    # To test IF NOT EXISTS semantics, we drop alembic_version and run it again.

    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )
    with conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS alembic_version;")

    conn.close()

    # Try again. Will re-run 001_initial_schema.py because alembic_version is missing
    command.upgrade(alembic_cfg, "head")
    # If no exception is raised, idempotent upgrade is successful
