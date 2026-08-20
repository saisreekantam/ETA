"""
Emergency-response endpoints: the alert inbox with an acknowledgment trail, the optional
outbound webhook, and the evidence-bundle export.

Alert rows are created by run_scenario (server/main.py) whenever a pipeline pass
escalates to alert/emergency. Acknowledging records who and when -- turning "someone
probably saw it" into an auditable fact. The webhook (ALERT_WEBHOOK_URL) posts the same
payload to any external channel (Slack/Teams/incident tooling) fire-and-forget.

The evidence bundle is a self-contained printable HTML page (report + citations +
violations + scores + CCTV frame + audit trail) -- print-to-PDF gives the regulator
handoff without a PDF library dependency.
"""
from __future__ import annotations

import base64
import csv
import html
import io
import json
import threading
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.models import Alert, Facility, ZoneRiskScore
from db.session import SessionLocal, get_db
from db.settings import settings
from server.break_mode import get_break_mode, set_break_mode

REPO_ROOT = Path(__file__).resolve().parents[1]

router = APIRouter(tags=["alerts"])

DEMO_FACILITY_NAME = "Demo Steel & Chemical Plant"


def _resolve_facility_id(db: Session, facility_id: str | None) -> uuid.UUID:
    if facility_id:
        return uuid.UUID(facility_id)
    facility = db.query(Facility).filter_by(name=DEMO_FACILITY_NAME).first()
    if facility is None:
        raise HTTPException(status_code=500, detail="No facility seeded -- run `python -m db.seed` first")
    return facility.id


def fire_webhook(payload: dict):
    """POST the alert to ALERT_WEBHOOK_URL without blocking or failing the pipeline
    response -- notification delivery must never make risk detection slower."""
    url = settings.alert_webhook_url
    if not url:
        return

    def _post():
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    threading.Thread(target=_post, daemon=True).start()


class BreakModeBody(BaseModel):
    active: bool
    operator_name: str | None = None


@router.get("/break-mode")
def read_break_mode(facility_id: str | None = None, db: Session = Depends(get_db)):
    fid = _resolve_facility_id(db, facility_id)
    return get_break_mode(fid)


@router.post("/break-mode")
def update_break_mode(body: BreakModeBody, facility_id: str | None = None, db: Session = Depends(get_db)):
    """Toggled from the header ('Screen off' / 'I'm back'). While active,
    persist_alert fires an extra, urgently-flagged webhook call for alert/emergency
    escalations -- see its docstring and server/break_mode.py."""
    fid = _resolve_facility_id(db, facility_id)
    return set_break_mode(fid, body.active, body.operator_name)


def _alert_payload(a: Alert) -> dict:
    return {
        "id": str(a.id),
        "zone": a.zone.key if a.zone_id else None,
        "zone_label": a.zone.label if a.zone_id else "Facility-wide",
        "run_id": a.external_run_id,
        "level": a.level,
        "message": a.message,
        "reasoning": a.reasoning,
        "reasoning_status": a.reasoning_status,
        "created_at": a.created_at.isoformat() + "Z",
        "acknowledged_by": a.acknowledged_by,
        "acknowledged_at": a.acknowledged_at.isoformat() + "Z" if a.acknowledged_at else None,
    }


_UNAVAILABLE_PREFIX = "[LLM unavailable"


def _generate_reasoning_async(alert_id: uuid.UUID, message: str, prompt: str | None = None):
    """Fire-and-forget: the alert is already persisted and visible before this runs, so a
    slow or failed LLM call never delays the alert itself -- only fills in the 'why' a few
    seconds later. Opens its own DB session since the request's session will be closed by
    the time this completes.

    prompt: pass the RAG-grounded prompt agents/vision_pipeline.py::build_reasoning_prompt
    builds for CCTV-correlated alerts (real regulatory citations, same as the sensor
    pipeline's incident_report). Omit it for alerts with no permit-correlation context
    (device anomalies, fast-path fall/fire) -- falls back to a generic explanation."""
    def _work():
        from agents.nodes.orchestrator_node import _call_llm
        effective_prompt = prompt or (
            f"In 2-3 plain, non-technical sentences, explain why this industrial "
            f"safety alert fired and what immediate risk it represents. Do not invent "
            f"details beyond what's stated.\n\nAlert: {message}")
        text = _call_llm(effective_prompt)
        reasoning = None if text.startswith(_UNAVAILABLE_PREFIX) else text
        status = "ready" if reasoning else "unavailable"

        db = SessionLocal()
        try:
            alert = db.query(Alert).filter_by(id=alert_id).one_or_none()
            if alert:
                alert.reasoning = reasoning
                alert.reasoning_status = status
                db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    threading.Thread(target=_work, daemon=True).start()


def persist_alert(db: Session, *, facility_id: uuid.UUID, zone_id: uuid.UUID | None, level: str,
                  message: str, run_id: uuid.UUID | None = None, external_run_id: str | None = None,
                  reasoning: str | None = None, reasoning_prompt: str | None = None,
                  source: str = "pipeline", source_key: str | None = None) -> Alert:
    """The one place an Alert row gets created -- unifies what used to be two divergent
    call sites (server/main.py's inline block, server/live_watch.py::raise_live_alert)
    with different run_id handling and webhook payload shapes.

    reasoning: pass it when already available (the sensor pipeline runs the orchestrator's
    LLM call before an escalation is known, so it's ready at persist time). Omit it
    everywhere else -- the alert commits immediately with reasoning_status='pending' and a
    background call fills it in a few seconds later, so nothing (life-safety alerts,
    CCTV-correlated ones) is ever delayed by an LLM round-trip.
    reasoning_prompt: for the async case, an optional RAG-grounded prompt (see
    agents/vision_pipeline.py::build_reasoning_prompt) instead of the generic fallback.
    source_key: identifies *what ongoing hazard* this is (e.g. "vision:reactor_zone:fall_detected")
    for callers that re-check the same condition repeatedly (CCTV, device watcher) -- see
    has_open_alert. Leave None for one-shot alerts (sensor-pipeline runs, security)."""
    alert = Alert(facility_id=facility_id, zone_id=zone_id, run_id=run_id,
                  external_run_id=external_run_id, level=level, message=message,
                  reasoning=reasoning, reasoning_status="ready" if reasoning else "pending",
                  source_key=source_key)
    db.add(alert)
    db.commit()
    db.refresh(alert)

    fire_webhook({"level": level, "zone": alert.zone.key if zone_id else None,
                 "run_id": external_run_id, "message": message, "source": source})

    # Nobody's watching the screen right now -- escalate harder than the routine webhook
    # above. Same URL/mechanism, just a second, distinctly-flagged call so the receiving
    # side (Slack/WhatsApp/etc.) can route it urgently instead of as routine noise.
    if level in ("alert", "emergency"):
        break_state = get_break_mode(facility_id)
        if break_state["active"]:
            fire_webhook({
                "level": level, "zone": alert.zone.key if zone_id else None,
                "run_id": external_run_id, "message": message, "source": source,
                "urgent": True, "escalation_reason": "operator_on_break",
                "operator_name": break_state.get("operator_name"),
            })

    if reasoning is None:
        _generate_reasoning_async(alert.id, message, prompt=reasoning_prompt)
    return alert


def has_open_alert(db: Session, facility_id: uuid.UUID, source_key: str) -> bool:
    """True if this exact ongoing hazard (see persist_alert's source_key) already has an
    unacknowledged alert sitting in the inbox. Callers that re-check a persisting
    condition on a timer (CCTV detections, device anomaly polling) use this instead of
    re-raising every cooldown window regardless of whether anyone's addressed the first
    one -- an officer who hasn't acknowledged it yet doesn't need five more of the same
    thing; an officer who HAS acknowledged it (or a genuinely new occurrence after the
    hazard cleared and came back) should still get a fresh alert."""
    return db.query(Alert.id).filter_by(
        facility_id=facility_id, source_key=source_key, acknowledged_at=None,
    ).first() is not None


@router.get("/alerts")
def list_alerts(facility_id: str | None = None, unacked_only: bool = False, limit: int = 30,
                db: Session = Depends(get_db)):
    fid = _resolve_facility_id(db, facility_id)
    q = db.query(Alert).filter_by(facility_id=fid)
    if unacked_only:
        q = q.filter(Alert.acknowledged_at.is_(None))
    rows = q.order_by(Alert.created_at.desc()).limit(limit).all()
    return [_alert_payload(a) for a in rows]


@router.get("/alerts/export")
def export_alerts_csv(facility_id: str | None = None, limit: int = 5000, db: Session = Depends(get_db)):
    """Every alert this facility has raised, one row per alert, for offline analysis
    (e.g. false-positive rate over time, which zones/levels dominate, alert->ack latency).
    Where an alert came from a scored run (run_id set -- the sensor pipeline, not CCTV/
    device fast-path alerts, which have no GNN score to attach), the matching
    zone_risk_scores row is joined in so the sensor-driven numbers ride along with the
    alert that was raised from them, not just the human-readable message."""
    fid = _resolve_facility_id(db, facility_id)
    rows = (db.query(Alert).filter_by(facility_id=fid)
            .order_by(Alert.created_at.desc()).limit(limit).all())

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "alert_id", "created_at_utc", "level", "zone_key", "zone_label", "message",
        "reasoning", "reasoning_status", "acknowledged_by", "acknowledged_at_utc",
        "ack_latency_seconds", "run_id", "compound_risk_score", "baseline_risk_score",
        "contributing_sensors",
    ])
    for a in rows:
        score = None
        if a.run_id and a.zone_id:
            score = (db.query(ZoneRiskScore)
                     .filter_by(run_id=a.run_id, zone_id=a.zone_id)
                     .order_by(ZoneRiskScore.computed_at.desc()).first())
        ack_latency = (a.acknowledged_at - a.created_at).total_seconds() if a.acknowledged_at else ""
        writer.writerow([
            str(a.id), a.created_at.isoformat(), a.level,
            a.zone.key if a.zone_id else "", a.zone.label if a.zone_id else "facility-wide",
            a.message, a.reasoning or "", a.reasoning_status,
            a.acknowledged_by or "", a.acknowledged_at.isoformat() if a.acknowledged_at else "",
            ack_latency, a.external_run_id or "",
            f"{score.compound_risk_score:.4f}" if score else "",
            f"{score.baseline_risk_score:.4f}" if score else "",
            ";".join(score.contributing_sensors) if score and score.contributing_sensors else "",
        ])

    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=alerts_export.csv"},
    )


class AckBody(BaseModel):
    acknowledged_by: str


@router.post("/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: str, body: AckBody, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter_by(id=uuid.UUID(alert_id)).one_or_none()
    if alert is None:
        raise HTTPException(status_code=404, detail="alert not found")
    if alert.acknowledged_at is not None:
        return _alert_payload(alert)
    alert.acknowledged_by = body.acknowledged_by.strip() or "safety officer"
    alert.acknowledged_at = datetime.utcnow()
    db.commit()
    return _alert_payload(alert)


# --- evidence bundle ----------------------------------------------------------------

_EVIDENCE_CSS = """
body { font-family: -apple-system, 'Segoe UI', sans-serif; color: #1a2230; margin: 40px auto;
       max-width: 800px; line-height: 1.55; font-size: 14px; }
h1 { font-size: 22px; margin-bottom: 2px; }
h2 { font-size: 15px; margin: 26px 0 8px; border-bottom: 1px solid #d8dde5; padding-bottom: 4px; }
.meta { color: #55606f; font-size: 12.5px; margin-bottom: 18px; }
.badge { display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700;
         font-size: 12px; text-transform: uppercase; }
.badge.emergency { background: #fbe4e4; color: #a02020; }
.badge.alert { background: #fdeedd; color: #9a5615; }
.badge.monitor { background: #fdf6dd; color: #855f0a; }
.badge.none { background: #e0f0e9; color: #2f6f52; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e4e8ee; }
th { color: #55606f; font-weight: 600; font-size: 12px; text-transform: uppercase; }
.violation { background: #fdf1f1; border: 1px solid #f0c8c8; border-radius: 8px;
             padding: 10px 14px; margin: 8px 0; font-size: 13px; }
.report { white-space: pre-wrap; background: #f7f8fa; border: 1px solid #e4e8ee;
          border-radius: 8px; padding: 16px; }
img.frame { max-width: 420px; border-radius: 8px; border: 1px solid #d8dde5; }
ol.audit { font-size: 13px; color: #3a4454; }
.footer { margin-top: 34px; color: #8a94a3; font-size: 11.5px; border-top: 1px solid #e4e8ee;
          padding-top: 10px; }
@media print { body { margin: 10mm; } }
"""


@router.get("/runs/{run_id}/evidence", response_class=HTMLResponse)
def evidence_bundle(run_id: str, facility_id: str | None = None, db: Session = Depends(get_db)):
    """Self-contained evidence page for the most recent pipeline pass of this run."""
    # Imported here to avoid a circular import (main imports this router).
    from server.main import _read_stored_result
    from db.models import Run

    fid = _resolve_facility_id(db, facility_id)
    run_row = db.query(Run).filter_by(external_run_id=run_id).one_or_none()
    result = _read_stored_result(db, fid, run_id, run_row)
    if result is None:
        raise HTTPException(status_code=404, detail="No stored result for this run -- run the scenario first")

    esc = result["escalation_level"]
    scores = sorted(result["zone_risk_scores"], key=lambda s: -s["compound_risk_score"])
    top = scores[0] if scores else None

    frame_html = ""
    for det in result["vision_detections"]:
        p = det.get("attention_map_path")
        if p and (REPO_ROOT / p).exists():
            b64 = base64.b64encode((REPO_ROOT / p).read_bytes()).decode()
            frame_html += (f'<p><img class="frame" src="data:image/jpeg;base64,{b64}" '
                           f'alt="CCTV frame {html.escape(det["frame_id"])}"/><br/>'
                           f'<small>Frame {html.escape(det["frame_id"])} · zone {html.escape(det["zone"])} · '
                           f'detections: {html.escape(", ".join(det["detections"]))}</small></p>')

    rows = "".join(
        f"<tr><td>{html.escape(s['zone'])}</td><td>{s['compound_risk_score']:.3f}</td>"
        f"<td>{s['baseline_risk_score']:.3f}</td></tr>" for s in scores)
    violations = "".join(
        f'<div class="violation"><strong>[{html.escape(v["severity"])}]</strong> {html.escape(v["reason"])}</div>'
        for v in result["permit_violations"]) or "<p>None flagged.</p>"
    citations = "".join(f"<li>{html.escape(c)}</li>" for c in result["retrieved_citations"])
    audit = "".join(f"<li>{html.escape(a)}</li>" for a in result["audit_log"])

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Evidence bundle · {html.escape(run_id)}</title><style>{_EVIDENCE_CSS}</style></head><body>
<h1>Incident Evidence Bundle</h1>
<div class="meta">Run {html.escape(run_id)} · generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
· Industrial Safety Intelligence</div>
<p>Escalation level: <span class="badge {html.escape(esc)}">{html.escape(esc)}</span>
{f"&nbsp; Top zone: <strong>{html.escape(top['zone'])}</strong> (compound risk {top['compound_risk_score']:.2f})" if top else ""}</p>
<h2>Zone risk scores (GNN vs single-sensor baseline)</h2>
<table><tr><th>Zone</th><th>Compound risk</th><th>Baseline</th></tr>{rows}</table>
<h2>Permit violations</h2>{violations}
<h2>CCTV evidence</h2>{frame_html or "<p>No frames attached.</p>"}
<h2>Generated incident report</h2><div class="report">{html.escape(result["incident_report"] or "")}</div>
<h2>Regulatory citations</h2><ul>{citations}</ul>
<h2>Agent audit trail</h2><ol class="audit">{audit}</ol>
<div class="footer">Generated on-premise. Report text produced by a locally hosted LLM;
citations retrieved from the DGMS/OISD/Factory-Act corpus. Print to PDF for filing.</div>
</body></html>"""
    return HTMLResponse(page)
