"""add source_key to alerts

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-08-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('alerts', sa.Column('source_key', sa.String(length=200), nullable=True))
    op.create_index('ix_alerts_source_key', 'alerts', ['source_key'])


def downgrade() -> None:
    op.drop_index('ix_alerts_source_key', table_name='alerts')
    op.drop_column('alerts', 'source_key')
