"""add_vector_support

Revision ID: 0a3b3d3aa847
Revises: 4261a9480485
Create Date: 2025-03-26 09:26:15.803263

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0a3b3d3aa847"
down_revision: Union[str, None] = "4261a9480485"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable vector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Add vector column to questions table
    op.add_column("questions", sa.Column("embedding", Vector(384), nullable=False))

    # Create vector index with name and optimized settings
    op.execute(
        "CREATE INDEX questions_embedding_idx ON questions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    # Drop index (using the specific index name)
    op.execute("DROP INDEX IF EXISTS questions_embedding_idx")

    # Drop column
    op.drop_column("questions", "embedding")

    # Drop extension
    op.execute("DROP EXTENSION IF EXISTS vector")
