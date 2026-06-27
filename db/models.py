"""
SQLAlchemy models. `facilities` is the multi-tenant anchor -- every other table that
represents real-world state (zones, permits, presence, incidents, audit log, vision
sessions) carries a facility_id, because the original prototype hardcoded one synthetic
plant. A real deployment monitors many plants, each with its own zone layout, permits,
and sensors, so nothing here assumes there is only one facility.

Sensor time-series data and model weights stay on the filesystem (parquet/pt files) --
they're bulk binary/numeric data, not the kind of thing a relational DB is for. What
moves into Postgres is everything that's actually relational: permits reference zones,
violations reference permits and risk scores, incident reports reference violations, all
of which benefit from real foreign keys and querying instead of re-parsing CSVs per request.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Facility(Base):
    """The multi-tenant anchor. One row per real (or simulated-as-if-real) plant."""
    __tablename__ = "facilities"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200), unique=True)
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    timezone: Mapped[str] = mapped_column(String(50), default="Asia/Kolkata")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    zones: Mapped[list["Zone"]] = relationship(back_populates="facility")


class Zone(Base):
    """A monitored area within a facility -- the schematic plant-map boxes in the
    frontend correspond 1:1 to rows here, scoped per facility instead of the global
    fixed 7-zone layout the prototype started with."""
    __tablename__ = "zones"
    __table_args__ = (UniqueConstraint("facility_id", "key", name="uq_zone_facility_key"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id"))
    key: Mapped[str] = mapped_column(String(100))  # e.g. "reactor_zone"
    label: Mapped[str] = mapped_column(String(200))
    layout_x: Mapped[float] = mapped_column(Float, default=0)
    layout_y: Mapped[float] = mapped_column(Float, default=0)
    layout_w: Mapped[float] = mapped_column(Float, default=100)
    layout_h: Mapped[float] = mapped_column(Float, default=100)
    has_sensors: Mapped[bool] = mapped_column(Boolean, default=True)

    facility: Mapped["Facility"] = relationship(back_populates="zones")


class Run(Base):
    """One synthetic-benchmark run (or, in a live deployment, one ingestion window) --
    kept as a first-class record so eval/demo replay can look up ground truth without
    re-reading manifest.csv, and so live deployments have an analogous concept to log
    against (a 'run' becomes a rolling time window instead of a fixed synthetic episode)."""
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id"))
    zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("zones.id"))
    external_run_id: Mapped[str] = mapped_column(String(50), index=True)  # the synthetic run_id string
    scenario_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    condition: Mapped[str | None] = mapped_column(String(50), nullable=True)
    permit_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    compound_active: Mapped[bool] = mapped_column(Boolean, default=False)
    ground_truth_onset_sample: Mapped[float | None] = mapped_column(Float, nullable=True)
    sensor_data_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    zone: Mapped["Zone"] = relationship()


class Permit(Base):
    __tablename__ = "permits"

    id: Mapped[uuid.UUID] = _uuid_pk()
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id"))
    zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("zones.id"))
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    external_permit_id: Mapped[str] = mapped_column(String(50))  # e.g. "PTW-e9206897"
    permit_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20))  # active | expired | revoked
    valid_from: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class WorkerPresence(Base):
    __tablename__ = "worker_presence"

    id: Mapped[uuid.UUID] = _uuid_pk()
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id"))
    zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("zones.id"))
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    external_worker_id: Mapped[str] = mapped_column(String(50))
    entry_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ZoneRiskScore(Base):
    __tablename__ = "zone_risk_scores"

    id: Mapped[uuid.UUID] = _uuid_pk()
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id"))
    zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("zones.id"))
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    compound_risk_score: Mapped[float] = mapped_column(Float)
    baseline_risk_score: Mapped[float] = mapped_column(Float)
    contributing_sensors: Mapped[list] = mapped_column(JSONB, default=list)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PermitViolation(Base):
    __tablename__ = "permit_violations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id"))
    zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("zones.id"))
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    permit_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("permits.id"), nullable=True)
    reason: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    zone: Mapped["Zone"] = relationship()


class IncidentReport(Base):
    __tablename__ = "incident_reports"

    id: Mapped[uuid.UUID] = _uuid_pk()
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id"))
    zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("zones.id"))
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    escalation_level: Mapped[str] = mapped_column(String(20))
    report_text: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    zone: Mapped["Zone"] = relationship()


class AuditLogEntry(Base):
    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = _uuid_pk()
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id"))
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    node_name: Mapped[str] = mapped_column(String(100))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class VisionSessionRecord(Base):
    """Persisted record of a live/video vision session -- the in-memory session state
    in vision/live_inference.py is still the live source of truth while a session is
    actively running (frame buffer, thread handle), but start/stop/config and the event
    log get written here so they survive a backend restart and are queryable per facility."""
    __tablename__ = "vision_sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id"))
    zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("zones.id"))
    external_session_id: Mapped[str] = mapped_column(String(50), index=True)
    source_type: Mapped[str] = mapped_column(String(20))  # sample | upload | live
    source: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class VisionEventRecord(Base):
    __tablename__ = "vision_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("vision_sessions.id"))
    detector_name: Mapped[str] = mapped_column(String(50))
    event_type: Mapped[str] = mapped_column(String(50))
    detail: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RagDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    source_label: Mapped[str] = mapped_column(String(300))  # e.g. "DGMS (Tech.) Circular No. 05 of 2020"
    file_path: Mapped[str] = mapped_column(String(500))


class RagChunk(Base):
    """Embedding dim 384 matches all-MiniLM-L6-v2 (rag/ingest.py) -- update both together
    if the embedding model changes."""
    __tablename__ = "rag_chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rag_documents.id"))
    text: Mapped[str] = mapped_column(Text)
    citation: Mapped[str] = mapped_column(String(300))
    page: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))


class ApiKey(Base):
    """Minimal auth: a hashed key, optionally scoped to one facility (null = all
    facilities, for an admin/demo key). Checked by server/auth.py."""
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = _uuid_pk()
    key_hash: Mapped[str] = mapped_column(String(128), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    facility_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("facilities.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
