"""add device ingest tokens; allow zone-less (security) alerts

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-18

"""
import secrets
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('devices', sa.Column('ingest_token', sa.String(length=64), nullable=True))
    op.alter_column('alerts', 'zone_id', existing_type=sa.UUID(), nullable=True)

    # backfill: every existing sensor device gets a token so enforcement can be strict
    conn = op.get_bind()
    for (device_id,) in conn.execute(sa.text("SELECT id FROM devices WHERE kind = 'sensor'")):
        conn.execute(sa.text("UPDATE devices SET ingest_token = :t WHERE id = :i"),
                     {"t": "isi_dev_" + secrets.token_hex(16), "i": device_id})


def downgrade() -> None:
    op.alter_column('alerts', 'zone_id', existing_type=sa.UUID(), nullable=False)
    op.drop_column('devices', 'ingest_token')
