"""
FastAPI server exposing the agent pipeline and facility data to the frontend.
Deliberately thin -- all real logic lives in agents/ and models/, this just adapts it
to HTTP for the dashboard. Reads/writes Postgres (db/models.py) for everything relational
(zones, runs, permits, results) instead of re-parsing CSVs/parquet on every request --
see db/seed.py for the one-time load of the synthetic benchmark into the DB.

Every endpoint takes an optional facility_id (defaults to the seeded demo facility) --
this is deliberately exposed even though the frontend doesn't yet have a facility
picker, so the API itself is genuinely multi-tenant rather than just the schema.

Run: `uvicorn server.main:app --reload --port 8000` (from industrial-safety-intel/)
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.graph import build_pipeline  # noqa: E402
from db.auth import verify_api_key  # noqa: E402
from db.models import AuditLogEntry, Facility, IncidentReport, PermitViolation, Run, Zone, ZoneRiskScore  # noqa: E402
from db.session import SessionLocal, get_db  # noqa: E402
from db.settings import settings  # noqa: E402
from scripts.demo_scenario_runner import load_state_for_run  # noqa: E402
from scripts.replay import compute_replay_trace  # noqa: E402
from vision.live_inference import create_session, get_session, stop_session  # noqa: E402

UPLOAD_DIR = REPO_ROOT / "data" / "vision_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEMO_FACILITY_NAME = "Demo Steel & Chemical Plant"


def require_api_key(x_api_key: str | None = Header(default=None), api_key: str | None = None):
    """Applied globally (see FastAPI(dependencies=...) below) -- every route requires
    a valid key once API_KEY_REQUIRED=true in .env. Off by default in dev so local
    iteration doesn't need a key on every curl; flip it on before exposing this anywhere
    reachable by someone other than you, since right now nothing else gates access.

    Accepts the key via the X-API-Key header (normal fetch/XHR calls) OR an api_key
    query param (the MJPEG stream is loaded via a plain <img src>, which can't set
    custom headers -- query param is the standard fallback for that case)."""
    if not settings.api_key_required:
        return
    db = SessionLocal()
    try:
        key = verify_api_key(db, x_api_key or api_key)
        if key is None:
            raise HTTPException(status_code=401, detail="Missing or invalid API key (X-API-Key header or api_key query param)")
    finally:
        db.close()


app = FastAPI(title="Industrial Safety Intelligence API", dependencies=[Depends(require_api_key)])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static/ppe_vision", StaticFiles(directory=str(REPO_ROOT / "data" / "ppe_vision" / "cached_outputs")),
          name="ppe_vision_static")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def _resolve_facility_id(db: Session, facility_id: str | None) -> uuid.UUID:
    if facility_id:
        return uuid.UUID(facility_id)
    facility = db.query(Facility).filter_by(name=DEMO_FACILITY_NAME).first()
    if facility is None:
        raise HTTPException(status_code=500, detail="No facility seeded -- run `python -m db.seed` first")
    return facility.id


@app.get("/facilities")
def get_facilities(db: Session = Depends(get_db)):
    return [{"id": str(f.id), "name": f.name, "location": f.location} for f in db.query(Facility).all()]


@app.get("/zones")
def get_zones(facility_id: str | None = None, db: Session = Depends(get_db)):
    fid = _resolve_facility_id(db, facility_id)
    zones = db.query(Zone).filter_by(facility_id=fid).all()
    return {z.key: {"label": z.label, "x": z.layout_x, "y": z.layout_y, "w": z.layout_w, "h": z.layout_h}
            for z in zones}


@app.get("/scenarios")
def get_scenarios(facility_id: str | None = None, db: Session = Depends(get_db)):
    fid = _resolve_facility_id(db, facility_id)
    runs = db.query(Run).filter_by(facility_id=fid).order_by(Run.scenario_id).all()

    by_scenario: dict[str, dict] = {}
    for run in runs:
        bucket = by_scenario.setdefault(run.scenario_id, {
            "scenario_id": run.scenario_id, "zone": run.zone.key, "permit_type": run.permit_type,
            "compound_run_ids": [], "normal_run_ids": [],
        })
        if run.condition == "compound" and len(bucket["compound_run_ids"]) < 3:
            bucket["compound_run_ids"].append(run.external_run_id)
        elif run.condition == "normal" and len(bucket["normal_run_ids"]) < 1:
            bucket["normal_run_ids"].append(run.external_run_id)
    return list(by_scenario.values())


@app.post("/run/{run_id}")
def run_scenario(run_id: str, facility_id: str | None = None, db: Session = Depends(get_db)):
    try:
        state = load_state_for_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run_id {run_id} not found")

    pipeline = get_pipeline()
    config = {"configurable": {"thread_id": f"api-{uuid.uuid4().hex[:8]}"}}
    final_state = pipeline.invoke(state, config=config)

    fid = _resolve_facility_id(db, facility_id)
    run_row = db.query(Run).filter_by(external_run_id=run_id).one_or_none()

    # Persist results -- this is what makes incident history real instead of thrown
    # away after each request. zone_id per score is looked up by key since the agent
    # state only carries zone key strings, not DB ids.
    zone_by_key = {z.key: z for z in db.query(Zone).filter_by(facility_id=fid).all()}
    for score in final_state["zone_risk_scores"]:
        zone = zone_by_key.get(score["zone"])
        if zone:
            db.add(ZoneRiskScore(
                facility_id=fid, zone_id=zone.id, run_id=run_row.id if run_row else None,
                compound_risk_score=score["compound_risk_score"], baseline_risk_score=score["baseline_risk_score"],
                contributing_sensors=score["contributing_sensors"],
            ))
    for violation in final_state["permit_violations"]:
        zone = zone_by_key.get(violation["zone"])
        if zone:
            db.add(PermitViolation(
                facility_id=fid, zone_id=zone.id, run_id=run_row.id if run_row else None,
                reason=violation["reason"], severity=violation["severity"],
            ))
    if final_state["incident_report"]:
        top_zone_key = max(final_state["zone_risk_scores"], key=lambda s: s["compound_risk_score"])["zone"]
        zone = zone_by_key.get(top_zone_key)
        if zone:
            db.add(IncidentReport(
                facility_id=fid, zone_id=zone.id, run_id=run_row.id if run_row else None,
                escalation_level=final_state["escalation_level"], report_text=final_state["incident_report"],
                citations=[c["source"] for c in final_state["retrieved_citations"]],
            ))
    for entry in final_state["audit_log"]:
        db.add(AuditLogEntry(facility_id=fid, run_id=run_row.id if run_row else None,
                              node_name=entry.split(":")[0], message=entry))
    db.commit()

    return {
        "run_id": run_id,
        "zone_risk_scores": final_state["zone_risk_scores"],
        "permit_violations": final_state["permit_violations"],
        "escalation_level": final_state["escalation_level"],
        "incident_report": final_state["incident_report"],
        "retrieved_citations": [c["source"] for c in final_state["retrieved_citations"]],
        "audit_log": final_state["audit_log"],
        "permits": final_state["permits"],
        "worker_presence": final_state["worker_presence"],
        "vision_detections": [{
            **det,
            "image_url": f"/static/ppe_vision/{Path(det['attention_map_path']).name}" if det["attention_map_path"] else None,
        } for det in final_state["vision_detections"]],
    }


@app.get("/incidents")
def get_incidents(facility_id: str | None = None, limit: int = 50, db: Session = Depends(get_db)):
    """Incident history -- the real payoff of persisting results instead of discarding
    them after each request. Not yet surfaced in the frontend; available for a future
    'history' view or for an auditor to query directly."""
    fid = _resolve_facility_id(db, facility_id)
    reports = (db.query(IncidentReport).filter_by(facility_id=fid)
               .order_by(IncidentReport.created_at.desc()).limit(limit).all())
    return [{
        "id": str(r.id), "zone": r.zone.key, "escalation_level": r.escalation_level,
        "report_text": r.report_text, "citations": r.citations, "created_at": r.created_at.isoformat(),
    } for r in reports]


@app.get("/replay/{run_id}")
def replay_scenario(run_id: str, step: int = 4):
    try:
        return compute_replay_trace(run_id, step=step)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run_id {run_id} not found")


# --- Live / video hazard detection -----------------------------------------------
# Demo mode (video file) and production mode (live=True, camera/RTSP) run the exact
# same inference code path -- see vision/live_inference.py's module docstring.

@app.get("/vision/sample-clip")
def get_sample_clip():
    """Path to the shipped sample video, so the frontend can offer it without
    requiring the user to find/upload their own footage first."""
    return {"path": "sample_demo_clip.mp4",
            "url": "/static/ppe_vision/sample_demo_clip.mp4"}


@app.post("/vision/upload")
async def upload_video(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"path": str(dest)}


@app.post("/vision/sessions")
def start_vision_session(
    zone: str,
    live: bool = False,
    video_path: str | None = None,
    camera_index: int = 0,
    rtsp_url: str | None = None,
    run_intrusion: bool = False,
    run_fall: bool = False,
    run_fire_smoke: bool = False,
    has_active_permit: bool = False,
):
    if live:
        source = rtsp_url if rtsp_url else camera_index
        mode = "live"
    else:
        if not video_path:
            raise HTTPException(status_code=400, detail="video_path required when live=false")
        source = video_path
        mode = "video"

    session = create_session(
        source=source, zone=zone, run_intrusion=run_intrusion, run_fall=run_fall,
        run_fire_smoke=run_fire_smoke, has_active_permit=has_active_permit, mode=mode, loop_video=not live,
    )
    return {"session_id": session.session_id, "mode": mode}


@app.get("/vision/sessions/{session_id}/stream")
def stream_vision_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    def generate():
        while True:
            s = get_session(session_id)
            if s is None or not s.is_running():
                break
            frame = s.latest_jpeg()
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/vision/sessions/{session_id}/events")
def get_vision_session_events(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "running": session.is_running(),
        "frames_processed": session.frames_processed,
        "error": session.error,
        "events": [{"timestamp": e.timestamp, "detector": e.detector, "event": e.event, "detail": e.detail}
                   for e in session.events],
    }


@app.post("/vision/sessions/{session_id}/stop")
def stop_vision_session(session_id: str):
    stop_session(session_id)
    return {"stopped": True}
