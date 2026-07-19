"""add sensor_saliency to zone_risk_scores

Revision ID: 9c3d4e5f6a7b
Revises: 8b2c3d4e5f6a
Create Date: 2026-07-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '9c3d4e5f6a7b'
down_revision: Union[str, Sequence[str], None] = '8b2c3d4e5f6a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('zone_risk_scores',
                  sa.Column('sensor_saliency', JSONB(), nullable=False, server_default='[]'))


def downgrade() -> None:
    op.drop_column('zone_risk_scores', 'sensor_saliency')
