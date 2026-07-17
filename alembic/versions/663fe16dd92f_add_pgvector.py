"""add pgvector

Revision ID: 663fe16dd92f
Revises: 001
Create Date: 2026-07-17 14:51:11.356679

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '663fe16dd92f'
down_revision: Union[str, Sequence[str], None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure pgvector is available
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Add embedding column to chunks table (768 is default for nomic-embed-text)
    op.execute("ALTER TABLE chunks ADD COLUMN embedding vector(768);")

def downgrade() -> None:
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS embedding;")
    op.execute("DROP EXTENSION IF EXISTS vector;")
