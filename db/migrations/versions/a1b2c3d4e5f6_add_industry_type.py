"""add industry_type to facilities

Revision ID: a1b2c3d4e5f6
Revises: 9c3d4e5f6a7b
Create Date: 2026-07-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '9c3d4e5f6a7b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('facilities', sa.Column('industry_type', sa.String(length=50), nullable=True))
    # the seeded demo plant is the chemical-process template (TEP zones)
    op.execute("UPDATE facilities SET industry_type = 'chemical_plant' "
               "WHERE name = 'Demo Steel & Chemical Plant'")


def downgrade() -> None:
    op.drop_column('facilities', 'industry_type')
