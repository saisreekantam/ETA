"""add reasoning + reasoning_status to alerts

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-08-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('alerts', sa.Column('reasoning', sa.Text(), nullable=True))
    op.add_column('alerts', sa.Column('reasoning_status', sa.String(length=20),
                                       nullable=False, server_default='unavailable'))


def downgrade() -> None:
    op.drop_column('alerts', 'reasoning_status')
    op.drop_column('alerts', 'reasoning')
