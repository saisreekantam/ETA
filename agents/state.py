"""
Shared state schema for the LangGraph pipeline.

Every node (compound_risk, permit_correlation, orchestrator) reads from and
writes to this same TypedDict. Locking this early is what lets the GNN,
permit-correlation logic, and frontend get built in parallel against a
stable contract instead of guessing each other's output shape.
"""
from __future__ import annotations

from typing import Literal, TypedDict


class PermitRecord(TypedDict):
    permit_id: str
    permit_type: Literal["hot_work", "confined_space", "electrical", "general"]
    zone: str
    valid_from: str  # ISO timestamp
    valid_to: str
    status: Literal["active", "expired", "revoked"]


class WorkerPresenceRecord(TypedDict):
    worker_id: str
    zone: str
    entry_time: str
    exit_time: str | None  # None if still present


class ZoneRiskScore(TypedDict):
    zone: str
    compound_risk_score: float  # 0-1, from the GNN
    baseline_risk_score: float  # 0-1, from the naive single-sensor threshold model
    contributing_sensors: list[str]
    contributing_faults: list[str] | None  # only populated on synthetic/eval runs


class PermitViolation(TypedDict):
    permit_id: str
    zone: str
    reason: str  # e.g. "hot_work permit active in zone with compound_risk_score=0.87"
    severity: Literal["low", "medium", "high", "critical"]


class VisionDetection(TypedDict):
    frame_id: str
    zone: str
    detections: list[str]  # e.g. ["person_no_hardhat", "person"]
    confidence: list[float]
    attention_map_path: str | None  # explainability artifact from RT-DETR


class RetrievedCitation(TypedDict):
    source: str  # e.g. "OISD-STD-105" or "Factory Act 1948 S.36"
    text: str
    score: float


class PipelineState(TypedDict):
    run_id: str
    timestamp: str

    # inputs (populated by upstream data sources / synthetic generator)
    sensor_window: dict  # {sensor_name: [values]} for the current rolling window
    permits: list[PermitRecord]
    worker_presence: list[WorkerPresenceRecord]
    vision_detections: list[VisionDetection]

    # produced by compound_risk_node
    zone_risk_scores: list[ZoneRiskScore]

    # produced by permit_correlation_node
    permit_violations: list[PermitViolation]

    # produced by orchestrator_node
    retrieved_citations: list[RetrievedCitation]
    incident_report: str | None
    escalation_level: Literal["none", "monitor", "alert", "emergency"]

    # audit trail (every node appends, never overwrites)
    audit_log: list[str]
