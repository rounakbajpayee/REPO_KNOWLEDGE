"""initial schema

Revision ID: 001
Revises:
Create Date: 2024-06-11 11:15:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Projects table
    op.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            stack VARCHAR(255),
            last_indexed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Files table (cascades deletes to chunks)
    op.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            path VARCHAR(1024) NOT NULL,
            content_hash VARCHAR(64) NOT NULL,
            file_mtime DOUBLE PRECISION NOT NULL,
            last_indexed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project_id, path)
        );
    """)

    # Chunks table
    op.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id UUID PRIMARY KEY,
            file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
            project VARCHAR(255) NOT NULL,
            path VARCHAR(1024) NOT NULL,
            language VARCHAR(64),
            chunk_type VARCHAR(64),
            symbol VARCHAR(255),
            content TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Decision history vault table
    op.execute("""
        CREATE TABLE IF NOT EXISTS decision_logs (
            id SERIAL PRIMARY KEY,
            topic VARCHAR(255) NOT NULL,
            entry_name VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            rationale TEXT NOT NULL,
            options_considered JSONB,
            logged_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # System/audit logs table
    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP WITH TIME ZONE NOT NULL,
            trace_id VARCHAR(8),
            event VARCHAR(255) NOT NULL,
            severity VARCHAR(10) NOT NULL,
            subsystem VARCHAR(64) NOT NULL,
            duration_ms INTEGER,
            payload JSONB
        );
    """)

    # BM25 full-text search: generated tsvector column + GIN index
    op.execute("""
        ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR
            GENERATED ALWAYS AS (
                to_tsvector('english',
                    coalesce(symbol, '') || ' ' ||
                    coalesce(chunk_type, '') || ' ' ||
                    coalesce(content, '')
                )
            ) STORED;
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin
        ON chunks USING GIN (content_tsv);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS audit_logs CASCADE;")
    op.execute("DROP TABLE IF EXISTS decision_logs CASCADE;")
    op.execute("DROP TABLE IF EXISTS chunks CASCADE;")
    op.execute("DROP TABLE IF EXISTS files CASCADE;")
    op.execute("DROP TABLE IF EXISTS projects CASCADE;")
