"""
Replays one synthetic scenario through the full agent pipeline end-to-end -- this is
what the live/recorded demo runs. Prints the audit trail (the LangGraph checkpoint
history) and the generated incident report.

Permit/presence/run metadata now comes from Postgres (db/models.py), not
manifest.csv/permits.parquet/shiftlogs.parquet directly -- those flat files are still
how the *synthetic benchmark generators* (simulator/) produce data, but db/seed.py loads
them into the DB once, and everything downstream reads from there. Sensor time-series
data stays on disk (Run.sensor_data_path) since that's bulk numeric data, not relational.

Run directly: `python -m scripts.demo_scenario_runner [run_id]`
(if run_id omitted, picks the first compound run for scenario s1_reactor_heat_removal)
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.graph import build_pipeline  # noqa: E402
from db.models import Permit, Run, WorkerPresence  # noqa: E402
from db.session import SessionLocal  # noqa: E402


def _load_vision_detections(run_id: str, zone: str, is_compound: bool, has_active_permit: bool,
                             has_presence: bool) -> list[dict]:
    """PPE detection is fully pre-baked (model inference + event classification both
    happened in vision/bake_demo_frames.py). Zone-intrusion is pre-baked for the
    expensive part only (person count, via a stock detector) -- the unauthorized-entry
    classification itself is computed live here against the run's actual permit status.

    The cached demo image always shows people (it's a fixed photo), but this run's own
    synthetic ground truth (worker_presence.has_presence) may say no one is actually
    there -- the zone-intrusion entry is gated on has_presence so we never claim visual
    evidence of a person that contradicts this run's own data."""
    manifest_path = REPO_ROOT / "data" / "ppe_vision" / "cached_outputs" / "manifest.json"
    if not manifest_path.exists():
        return []
    cache = json.loads(manifest_path.read_text())
    tag = "violation_0" if is_compound else "compliant_0"
    entry = cache.get(tag)
    if entry is None:
        return []

    detections = [{
        "frame_id": tag,
        "zone": zone,
        "detections": entry["detections"],
        "confidence": entry["confidences"],
        "attention_map_path": f"data/ppe_vision/cached_outputs/{entry['annotated_image']}",
    }]

    person_count = entry.get("person_count", 0)
    if person_count > 0 and has_presence:
        label = "person_unauthorized" if not has_active_permit else "person_authorized"
        detections.append({
            "frame_id": f"{tag}_zone_intrusion",
            "zone": zone,
            "detections": [label] * person_count,
            "confidence": entry.get("person_confidences", []),
            "attention_map_path": None,
        })
    return detections


def load_state_for_run(run_id: str) -> dict:
    db = SessionLocal()
    try:
        run = db.query(Run).filter_by(external_run_id=run_id).one_or_none()
        if run is None:
            raise KeyError(f"run_id {run_id} not found in DB -- did you run `python -m db.seed`?")
        permit = db.query(Permit).filter_by(run_id=run.id).one_or_none()
        presence = db.query(WorkerPresence).filter_by(run_id=run.id).one_or_none()
        zone_key = run.zone.key
    finally:
        db.close()

    sensor_df = pd.read_parquet(REPO_ROOT / run.sensor_data_path)
    has_permit = permit is not None and permit.status == "active"
    has_presence = presence is not None and presence.entry_time is not None
    vision_detections = _load_vision_detections(run_id, zone_key, run.compound_active, has_permit, has_presence)

    return {
        "run_id": run_id,
        "timestamp": pd.Timestamp.now().isoformat(),
        "sensor_window": sensor_df.to_dict(orient="list"),
        "permits": [{
            "permit_id": permit.external_permit_id, "permit_type": permit.permit_type,
            "zone": zone_key, "valid_from": permit.valid_from, "valid_to": permit.valid_to,
            "status": permit.status,
        }] if permit else [],
        "worker_presence": [{
            "worker_id": presence.external_worker_id, "zone": zone_key,
            "entry_time": presence.entry_time, "exit_time": presence.exit_time,
            "has_presence": has_presence,
        }] if presence else [],
        "vision_detections": vision_detections,
        "zone_risk_scores": [],
        "permit_violations": [],
        "retrieved_citations": [],
        "incident_report": None,
        "escalation_level": "none",
        "audit_log": [],
    }


def main():
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        db = SessionLocal()
        try:
            run = (db.query(Run).filter_by(scenario_id="s1_reactor_heat_removal", condition="compound")
                   .first())
            run_id = run.external_run_id
        finally:
            db.close()

    print(f"=== Running pipeline for run_id={run_id} ===\n")
    state = load_state_for_run(run_id)
    pipeline = build_pipeline()
    config = {"configurable": {"thread_id": f"demo-{uuid.uuid4().hex[:8]}"}}
    final_state = pipeline.invoke(state, config=config)

    print("--- Audit trail ---")
    for entry in final_state["audit_log"]:
        print(" -", entry)

    print("\n--- Zone risk scores ---")
    for score in final_state["zone_risk_scores"]:
        print(f"  zone={score['zone']} gnn={score['compound_risk_score']} "
              f"baseline={score['baseline_risk_score']}")

    print("\n--- Permit violations ---")
    for v in final_state["permit_violations"]:
        print(f"  [{v['severity']}] {v['reason']}")

    print(f"\n--- Escalation level: {final_state['escalation_level']} ---")
    if final_state["incident_report"]:
        print("\n--- Incident report ---")
        print(final_state["incident_report"])
        print("\n--- Citations used ---")
        for c in final_state["retrieved_citations"]:
            print(f"  - {c['source']}")


if __name__ == "__main__":
    main()
