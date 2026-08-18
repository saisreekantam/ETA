"""
Real, time-scoped "is there an active permit for this zone right now" lookup -- the piece
that was missing for CCTV events to be genuinely permit-correlated instead of relying on
LiveMonitoring.jsx's self-reported `hasPermit` checkbox (never checked against the DB).

No code anywhere else in the app does this: scripts/demo_scenario_runner.py's permit lookup
is keyed by run_id (a synthetic benchmark run), not by (zone, status, now-in-range).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from db.models import Permit, Zone


def get_active_permits_for_zone(db: Session, facility_id: uuid.UUID | str, zone_key: str,
                                 now: datetime | None = None) -> list[Permit]:
    """Active permits for zone_key whose validity window covers `now` (default: current
    time). A null valid_from/valid_to is treated as open-ended on that side."""
    now = now or datetime.utcnow()
    fid = facility_id if isinstance(facility_id, uuid.UUID) else uuid.UUID(str(facility_id))

    zone = db.query(Zone).filter_by(facility_id=fid, key=zone_key).one_or_none()
    if zone is None:
        return []

    return (
        db.query(Permit)
        .filter(
            Permit.facility_id == fid,
            Permit.zone_id == zone.id,
            Permit.status == "active",
            (Permit.valid_from.is_(None)) | (Permit.valid_from <= now),
            (Permit.valid_to.is_(None)) | (Permit.valid_to >= now),
        )
        .all()
    )
