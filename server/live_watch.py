"""
The live last-mile: turns what the platform merely DISPLAYED into warnings someone is
accountable for seeing. Two inputs feed the same sink:

  1. Live CCTV hazard events (vision/live_inference.py) -- ppe_violation,
     unauthorized_entry, fall_detected, fire_smoke_detected fired per-frame on real
     video. Previously these lived only in the in-memory session log on the Live
     Monitoring page and vanished with the session.
  2. Ingested IoT sensor readings (server/devices.py) -- previously stored and charted,
     never evaluated.

Both now raise persisted Alert rows (the appbar bell + acknowledgment trail) and fire
the outbound webhook, exactly like pipeline escalations. A per-key cooldown stops a
30fps hazard or a stuck sensor from flooding the inbox: the first detection warns
immediately, repeats within the window are suppressed.

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

from db.models import Alert, Device, SensorReading, Zone
from db.session import SessionLocal
from server.alerts import fire_webhook

# event type -> alert level; fall/fire are life-safety, straight to emergency
VISION_EVENT_LEVELS = {
    "ppe_violation": "alert",
    "unauthorized_entry": "alert",
    "fall_detected": "emergency",
    "fire_smoke_detected": "emergency",
}

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


def raise_live_alert(facility_id: uuid.UUID | str | None, zone_key: str, level: str,
                     message: str, cooldown_key: str) -> bool:
    """Persist an Alert + fire the webhook, deduplicated by cooldown_key. Returns True
    if an alert was actually raised. Never raises -- a failed warning write must not
    take down the inference/ingest path that called it."""
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
        db.add(Alert(facility_id=zone.facility_id, zone_id=zone.id, level=level, message=message))
        db.commit()
        fire_webhook({"level": level, "zone": zone_key, "message": message, "source": "live"})
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
        db.add(Alert(facility_id=uuid.UUID(str(facility_id)), zone_id=zone_id,
                     level="security", message=message))
        db.commit()
        fire_webhook({"level": "security", "message": message, "source": "iot-ingest"})
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
