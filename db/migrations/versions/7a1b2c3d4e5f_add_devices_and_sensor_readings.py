"""add devices and sensor readings

Revision ID: 7a1b2c3d4e5f
Revises: 4cdfe3057dce
Create Date: 2026-07-06

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '7a1b2c3d4e5f'
down_revision: Union[str, Sequence[str], None] = '4cdfe3057dce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'devices',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('facility_id', sa.UUID(), nullable=False),
        sa.Column('zone_id', sa.UUID(), nullable=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('kind', sa.String(length=20), nullable=False),
        sa.Column('source_type', sa.String(length=20), nullable=False),
        sa.Column('source', sa.String(length=500), nullable=True),
        sa.Column('metric', sa.String(length=100), nullable=True),
        sa.Column('unit', sa.String(length=20), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('last_seen', sa.DateTime(), nullable=True),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['facility_id'], ['facilities.id']),
        sa.ForeignKeyConstraint(['zone_id'], ['zones.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'sensor_readings',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('device_id', sa.UUID(), nullable=False),
        sa.Column('metric', sa.String(length=100), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['device_id'], ['devices.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_sensor_readings_device_id'), 'sensor_readings', ['device_id'])
    op.create_index(op.f('ix_sensor_readings_created_at'), 'sensor_readings', ['created_at'])


def downgrade() -> None:
    op.drop_index(op.f('ix_sensor_readings_created_at'), table_name='sensor_readings')
    op.drop_index(op.f('ix_sensor_readings_device_id'), table_name='sensor_readings')
    op.drop_table('sensor_readings')
    op.drop_table('devices')
