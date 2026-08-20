"""
The live last-mile: turns what the platform merely DISPLAYED into warnings someone is
accountable for seeing. Three inputs feed the same sink:

  1. ppe_violation / unauthorized_entry (vision/live_inference.py) -- routed through
     agents/vision_pipeline.py's two-node pass (permit_correlation + orchestrator), so
     these get real permit correlation and an LLM-drafted "why" instead of a bare
     template string. Previously these lived only in the in-memory session log on the
     Live Monitoring page and vanished with the session.
  2. fall_detected / fire_smoke_detected -- unambiguous life-safety emergencies. These
     stay on the fast path below (raise_live_alert): persist immediately, LLM reasoning
     generated asynchronously afterward (see server/alerts.py::persist_alert) so paging
     someone is never delayed by an LLM round-trip.
  3. Ingested IoT sensor readings (server/devices.py) -- previously stored and charted,
     never evaluated.

All three raise persisted Alert rows (the appbar bell + acknowledgment trail) via the
same server/alerts.py::persist_alert, and fire the outbound webhook, exactly like
pipeline escalations. A per-key cooldown stops a 30fps hazard or a stuck sensor from
flooding the inbox: the first detection warns immediately, repeats within the window are
suppressed.

Device anomaly rule (deliberately simple and explainable): a reading is anomalous when
it deviates more than DEVICE_SIGMA from the device's own trailing mean (minimum history
required, tiny variance clamped). This is a per-device self-baseline -- no configuration
needed for a brand-new device beyond assigning it a zone. The trained GNN is NOT used
here: it expects the TEP simulator's 52 channels, and pretending an arbitrary device
maps onto that would be theater.
"""
from __future__ import annotations

import threading
import time
import uuid

from db.models import Device, SensorReading, Zone
from db.session import SessionLocal
from server.alerts import persist_alert

# event type -> alert level; fall/fire are life-safety, straight to emergency. Only used
# for fall/fire now -- ppe_violation/unauthorized_entry get their level from
# agents/vision_pipeline.py's orchestrator-driven escalation instead of this static map.
VISION_EVENT_LEVELS = {
    "ppe_violation": "alert",
    "unauthorized_entry": "alert",
    "fall_detected": "emergency",
    "fire_smoke_detected": "emergency",
}

# Routed through the orchestrator (agents/vision_pipeline.py) for permit correlation +
# an LLM-drafted report, instead of the fast bare-template raise_live_alert path.
_ORCHESTRATOR_ROUTED_EVENTS = {"ppe_violation", "unauthorized_entry"}

ALERT_COOLDOWN_S = 120       # per (source, zone, event) -- first hit warns, repeats wait
DEVICE_POLL_S = 30
DEVICE_SIGMA = 4.0
DEVICE_MIN_HISTORY = 12      # readings needed before a device has a usable self-baseline
DEVICE_WINDOW = 40

_last_raised: dict[str, float] = {}
_cooldown_lock = threading.Lock()


def _cooled_down(key: str) -> bool:
    now = time.monotonic()
    with _cooldown_lock:
        if now - _last_raised.get(key, -1e9) < ALERT_COOLDOWN_S:
            return False
        _last_raised[key] = now
        return True


def reset_vision_cooldowns(zone_key: str) -> None:
    """Clear this zone's vision-alert cooldowns so a freshly-started session isn't
    silently gated by a previous, unrelated session's alert history (e.g. testing Live
    camera and then Sample clip back to back on the same zone within ALERT_COOLDOWN_S --
    the second session's detections are real but would otherwise raise nothing). A new
    session starting is a deliberate re-arm; it should get its own first-hit warning.
    Cooldowns WITHIN a session's own run are unaffected -- this only fires once, at
    session start, not per-frame."""
    prefix = f"vision:{zone_key}:"
    with _cooldown_lock:
        for key in [k for k in _last_raised if k.startswith(prefix)]:
            del _last_raised[key]


def raise_live_alert(facility_id: uuid.UUID | str | None, zone_key: str, level: str,
                     message: str, cooldown_key: str) -> bool:
    """Persist an Alert (reasoning generated asynchronously, see
    server/alerts.py::persist_alert) deduplicated by cooldown_key. Returns True if an
    alert was actually raised. Never raises -- a failed warning write must not take down
    the inference/ingest path that called it."""
    if not _cooled_down(cooldown_key):
        return False
    db = SessionLocal()
    try:
        zone_q = db.query(Zone).filter_by(key=zone_key)
        if facility_id:
            zone_q = zone_q.filter_by(facility_id=uuid.UUID(str(facility_id)))
        zone = zone_q.first()
        if zone is None:
            return False
        persist_alert(db, facility_id=zone.facility_id, zone_id=zone.id, level=level,
                      message=message, source="live")
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def raise_security_alert(facility_id: uuid.UUID | str, message: str, cooldown_key: str,
                         zone_id: uuid.UUID | None = None) -> bool:
    """Facility-scoped security warning (spoofed/unregistered ingest attempts). Unlike
    hazard alerts these may have no zone -- the facility is the scope. Same cooldown
    discipline: a scripted attack hammering the endpoint warns once per window."""
    if not _cooled_down(cooldown_key):
        return False
    db = SessionLocal()
    try:
        persist_alert(db, facility_id=uuid.UUID(str(facility_id)), zone_id=zone_id,
                      level="security", message=message, source="iot-ingest")
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def _raise_vision_correlated_alert(facility_id: str | None, zone_key: str, event_type: str,
                                   cooldown_key: str, count: int = 1) -> bool:
    """ppe_violation/unauthorized_entry: run the real permit-correlation pass
    (agents/vision_pipeline.py) instead of a bare template alert -- fast (no LLM in this
    path, see that module's docstring), so the alert lands within ~1-2s of the camera
    detecting something. Only fires an Alert row if the correlation actually escalates (a
    PPE/intrusion event with no active-permit ambiguity may legitimately resolve to
    'none').

    Note the gate is ('monitor', 'alert', 'emergency'), wider than the full sensor pipeline
    (server/main.py only persists 'alert'/'emergency'). A pure-vision trigger with no
    active permit floors at severity='medium' -> escalation='monitor' (see
    agents/nodes/permit_correlation_node.py's _zone_intrusion_violations /
    _ppe_violations_independent docstrings) -- that's the COMMON case here, not an edge
    case, since there's no GNN score available to push it higher. Excluding 'monitor'
    would silently drop the majority of real live CCTV detections."""
    if not _cooled_down(cooldown_key):
        return False
    from agents.vision_pipeline import (  # local import: avoids a server.live_watch <->
        build_reasoning_prompt, run_vision_correlation, top_violation_of,  # agents.vision_pipeline
    )  # <-> server.permit_lookup import cycle risk at module load time
    db = SessionLocal()
    try:
        zone_q = db.query(Zone).filter_by(key=zone_key)
        if facility_id:
            zone_q = zone_q.filter_by(facility_id=uuid.UUID(str(facility_id)))
        zone = zone_q.first()
        if zone is None:
            return False
        state = run_vision_correlation(db, zone.facility_id, zone_key, event_type, count=count)
        if state["escalation_level"] == "none":
            return False
        top_violation = top_violation_of(state["permit_violations"])
        prompt = build_reasoning_prompt(top_violation, state["permits"])
        persist_alert(db, facility_id=zone.facility_id, zone_id=zone.id,
                      level=state["escalation_level"], message=top_violation["reason"],
                      reasoning_prompt=prompt, source="live-cctv")
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def make_vision_alert_hook(facility_id: str | None):
    """Callback for VisionSession.on_event -- maps hazard events to alerts. Bound per
    session so the alert lands on the facility the session was started under."""
    def hook(session, event):
        if event.event in _ORCHESTRATOR_ROUTED_EVENTS:
            _raise_vision_correlated_alert(
                facility_id, session.zone, event.event,
                cooldown_key=f"vision:{session.zone}:{event.event}", count=event.count,
            )
            return
        level = VISION_EVENT_LEVELS.get(event.event)
        if level is None:
            return  # system events (stream_ended etc.) are not hazards
        raise_live_alert(
            facility_id, session.zone, level,
            f"LIVE CCTV ({event.detector}): {event.detail}",
            cooldown_key=f"vision:{session.zone}:{event.event}",
        )
    return hook


def _check_devices_once():
    db = SessionLocal()
    try:
        devices = db.query(Device).filter_by(kind="sensor").all()
        for device in devices:
            if device.zone_id is None:
                continue  # nowhere to attach a warning -- zone assignment is the opt-in
            readings = (db.query(SensorReading).filter_by(device_id=device.id)
                        .order_by(SensorReading.created_at.desc()).limit(DEVICE_WINDOW).all())
            if len(readings) < DEVICE_MIN_HISTORY:
                continue
            latest, history = readings[0], readings[3:]  # skip newest 3 in the baseline
            values = [r.value for r in history]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = max(var ** 0.5, 1e-9, abs(mean) * 1e-3)
            deviation = abs(latest.value - mean) / std
            if deviation >= DEVICE_SIGMA:
                zone = db.get(Zone, device.zone_id)
                if zone is None:
                    continue
                raise_live_alert(
                    zone.facility_id, zone.key, "alert",
                    (f"IoT device '{device.name}' anomalous: {device.metric or 'value'}="
                     f"{latest.value:.2f} deviates {deviation:.1f} sigma from its trailing "
                     f"mean {mean:.2f} (n={len(values)})"),
                    cooldown_key=f"device:{device.id}",
                )
    finally:
        db.close()


def start_device_watcher():
    """Background thread polling ingested readings -- started from server startup."""
    def loop():
        while True:
            try:
                _check_devices_once()
            except Exception:
                pass  # watcher must survive transient DB hiccups
            time.sleep(DEVICE_POLL_S)

    threading.Thread(target=loop, daemon=True, name="device-watcher").start()
