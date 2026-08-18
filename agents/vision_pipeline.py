"""
Routes a live CCTV hazard event (vision/live_inference.py's VisionEvent) through the SAME
permit-correlation reasoning the sensor pipeline uses, instead of the bare template-string
alert server/live_watch.py::raise_live_alert produces on its own.

Only ppe_violation/unauthorized_entry go through here (see server/live_watch.py) --
fall_detected/fire_smoke_detected are unambiguous life-safety emergencies that must alert
instantly, and stay on the fast raise_live_alert path unchanged.

SPEED: run_vision_correlation deliberately does NOT call the LLM (unlike
orchestrator_node, which it otherwise mirrors) -- permit lookup + severity/escalation
computation is all DB queries and dict logic, well under 100ms. The RAG-grounded incident
report is real, but generated ASYNCHRONOUSLY by server/alerts.py::persist_alert after the
Alert already exists (build_reasoning_prompt below hands it the exact same prompt
orchestrator_node would have used), so an operator sees the alert within ~1-2s of the
camera detecting something, with the "why" filling in a couple of seconds later -- not
gated behind an 8B-parameter local LLM round-trip before they see anything at all.

A pure-CCTV trigger has no sensor_window (the GNN needs TEP's 52 channels), so this can't
go through compound_risk_node -- instead it calls permit_correlation_node directly with
zone_risk_scores=[], which it's built to tolerate (see its empty-list handling). This is
a partial node pass, not a pipeline.invoke() -- LangGraph's compiled graph has a single
fixed entry point (compound_risk), and forcing an empty sensor_window through it would be
a lie about what was actually observed.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from agents.nodes.orchestrator_node import SEVERITY_TO_ESCALATION, build_incident_prompt
from agents.nodes.permit_correlation_node import permit_correlation_node
from agents.state import PermitRecord, PermitViolation, PipelineState, VisionDetection
from db.models import Permit
from server.permit_lookup import get_active_permits_for_zone

_SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}

# VisionEvent.event -> the vision_detections shape permit_correlation_node reads.
# VisionEvent (vision/live_inference.py) carries only 4 flat fields (timestamp, detector,
# event, detail) -- no structured labels/confidence -- so this is a lossy but faithful
# translation of "what kind of evidence this event represents", not a re-detection.
_EVENT_TO_DETECTIONS = {
    "ppe_violation": ["head"],
    "unauthorized_entry": ["person_unauthorized"],
}


def _permit_to_state_dict(permit: Permit) -> PermitRecord:
    return PermitRecord(
        permit_id=permit.external_permit_id,
        permit_type=permit.permit_type,
        zone=permit.zone.key if permit.zone else "",
        valid_from=permit.valid_from.isoformat() if permit.valid_from else "",
        valid_to=permit.valid_to.isoformat() if permit.valid_to else "",
        status=permit.status,
    )


def _translate_event(event_type: str, zone_key: str) -> VisionDetection:
    detections = _EVENT_TO_DETECTIONS[event_type]
    return VisionDetection(
        frame_id="live",
        zone=zone_key,
        detections=detections,
        confidence=[1.0] * len(detections),
        attention_map_path=None,
    )


def top_violation_of(violations: list[PermitViolation]) -> PermitViolation:
    return max(violations, key=lambda v: _SEVERITY_RANK[v["severity"]])


def build_reasoning_prompt(top_violation: PermitViolation, permits: list[PermitRecord]) -> str:
    """The RAG-grounded prompt orchestrator_node would have used -- for the async LLM
    call server/alerts.py::persist_alert kicks off after the alert already exists."""
    prompt, _citations = build_incident_prompt(top_violation, permits)
    return prompt


def run_vision_correlation(db: Session, facility_id: uuid.UUID | str, zone_key: str,
                            event_type: str) -> PipelineState:
    """event_type must be one of _EVENT_TO_DETECTIONS' keys. Returns the resulting state
    with escalation_level/permit_violations/audit_log populated -- incident_report is
    deliberately left None here (see module docstring on why the LLM call is async)."""
    active_permits = get_active_permits_for_zone(db, facility_id, zone_key)

    state: PipelineState = {
        "run_id": f"live-vision-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensor_window": {},
        "permits": [_permit_to_state_dict(p) for p in active_permits],
        "worker_presence": [],
        "vision_detections": [_translate_event(event_type, zone_key)],
        "zone_risk_scores": [],
        "flow_attention": [],
        "permit_violations": [],
        "retrieved_citations": [],
        "incident_report": None,
        "escalation_level": "none",
        "audit_log": [f"vision_pipeline: triggered by live CCTV event '{event_type}' in {zone_key}, "
                       f"{len(active_permits)} active permit(s) found for this zone"],
    }
    state.update(permit_correlation_node(state))

    if not state["permit_violations"]:
        state["audit_log"].append("vision_pipeline: no violations, no alert")
        return state

    top_violation = top_violation_of(state["permit_violations"])
    state["escalation_level"] = SEVERITY_TO_ESCALATION[top_violation["severity"]]
    state["audit_log"].append(
        f"vision_pipeline: escalation={state['escalation_level']} (reasoning generated asynchronously)")
    return state
