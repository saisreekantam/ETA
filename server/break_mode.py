"""
"Screen off" / on-break state: an operator stepping away from the console for a few
minutes shouldn't mean an emergency only shows up as a banner nobody's there to see.
While a facility is in break mode, server/alerts.py::persist_alert fires an EXTRA,
distinctly-flagged webhook call (reusing the existing ALERT_WEBHOOK_URL mechanism --
no new external service, no phone-number registration) for alert/emergency-level
escalations, so whatever's listening on the other end (Slack/WhatsApp/etc.) can route it
as urgent instead of a routine notification.

In-memory, not persisted -- mirrors server/live_watch.py's cooldown dict pattern. A
facility's break-mode state resetting on server restart is the right default (an
operator should re-confirm they're away after any redeploy, not silently inherit stale
state), and this is far simpler than a new DB table + migration for something this
lightweight.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime

_state: dict[str, dict] = {}
_lock = threading.Lock()


def set_break_mode(facility_id: uuid.UUID | str, active: bool, operator_name: str | None = None) -> dict:
    key = str(facility_id)
    with _lock:
        if active:
            _state[key] = {"active": True, "operator_name": operator_name, "since": datetime.utcnow()}
        else:
            _state.pop(key, None)
        return _state.get(key, {"active": False, "operator_name": None, "since": None})


def get_break_mode(facility_id: uuid.UUID | str) -> dict:
    with _lock:
        return _state.get(str(facility_id), {"active": False, "operator_name": None, "since": None})


def is_on_break(facility_id: uuid.UUID | str) -> bool:
    return get_break_mode(facility_id)["active"]
