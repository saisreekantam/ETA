"""
Real-time hazard detection over an actual video stream -- the live counterpart to the
cached/pre-baked demo path (vision/bake_demo_frames.py). Runs the same trained PPE model
frame-by-frame against either:
  - a video file (demo mode: no camera hardware needed, deterministic, repeatable)
  - a live camera / RTSP URL (production mode: source=0 for a local webcam, or an RTSP
    URL for an actual plant CCTV camera)

Both modes use the identical inference code path -- "demo vs production" is purely a
choice of cv2.VideoCapture source, nothing else differs. That's the point: the demo is
not a simulation of the capability, it's the same capability pointed at a file instead
of a camera.

Zone-intrusion is optional and throttled (checked every INTRUSION_EVERY_N_FRAMES) since
it runs a second model pass and isn't the primary signal for this view.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from vision.detectors.fall import CLASS_NAMES as FALL_CLASS_NAMES
from vision.detectors.fall import WEIGHTS as FALL_WEIGHTS
from vision.detectors.fire_smoke import CLASS_NAMES as FIRE_SMOKE_CLASS_NAMES
from vision.detectors.fire_smoke import WEIGHTS as FIRE_SMOKE_WEIGHTS
from vision.detectors.ppe import CLASS_NAMES as PPE_CLASS_NAMES
from vision.detectors.ppe import WEIGHTS as PPE_WEIGHTS
from vision.detectors.zone_intrusion import COCO_PERSON_CLASS_ID

INTRUSION_EVERY_N_FRAMES = 5
FALL_EVERY_N_FRAMES = 3
FIRE_SMOKE_EVERY_N_FRAMES = 3

_ppe_model = None
_person_model = None
_fall_model = None
_fire_smoke_model = None
_model_lock = threading.Lock()


def _load_models():
    global _ppe_model, _person_model, _fall_model, _fire_smoke_model
    with _model_lock:
        if _ppe_model is None:
            from ultralytics import RTDETR
            _ppe_model = RTDETR(str(PPE_WEIGHTS))
        if _person_model is None:
            from ultralytics import RTDETR
            _person_model = RTDETR("rtdetr-l.pt")
        if _fall_model is None:
            from ultralytics import RTDETR
            _fall_model = RTDETR(str(FALL_WEIGHTS))
        if _fire_smoke_model is None:
            from ultralytics import RTDETR
            _fire_smoke_model = RTDETR(str(FIRE_SMOKE_WEIGHTS))


@dataclass
class VisionEvent:
    timestamp: float
    detector: str
    event: str
    detail: str


@dataclass
class VisionSession:
    source: str | int
    zone: str
    run_intrusion: bool = False
    run_fall: bool = False
    run_fire_smoke: bool = False
    has_active_permit: bool = False
    loop_video: bool = True
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    events: list[VisionEvent] = field(default_factory=list)
    frames_processed: int = 0
    error: str | None = None
    mode: str = "video"  # "video" or "live", set by caller for display purposes only

    def __post_init__(self):
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None

    def start(self):
        _load_models()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def _log(self, detector: str, event: str, detail: str):
        self.events.insert(0, VisionEvent(timestamp=time.time(), detector=detector, event=event, detail=detail))
        self.events = self.events[:50]

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"Could not open video source: {self.source}"
            self._running = False
            return

        frame_idx = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                if self.loop_video and isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._log("system", "stream_ended", "Video source ended.")
                break

            try:
                ppe_results = _ppe_model.predict(frame, conf=0.4, verbose=False)[0]
                annotated = ppe_results.plot()
                labels = [PPE_CLASS_NAMES[int(b.cls.item())] for b in ppe_results.boxes]
                if "head" in labels and "helmet" not in labels:
                    n = labels.count("head")
                    self._log("ppe", "ppe_violation", f"{n} worker(s) detected without helmets in {self.zone}")

                if self.run_intrusion and frame_idx % INTRUSION_EVERY_N_FRAMES == 0:
                    person_results = _person_model.predict(frame, conf=0.4, verbose=False)[0]
                    n_person = sum(1 for b in person_results.boxes if int(b.cls.item()) == COCO_PERSON_CLASS_ID)
                    if n_person > 0 and not self.has_active_permit:
                        self._log("zone_intrusion", "unauthorized_entry",
                                  f"{n_person} person(s) detected in {self.zone} with no active permit")

                if self.run_fall and frame_idx % FALL_EVERY_N_FRAMES == 0:
                    fall_results = _fall_model.predict(frame, conf=0.4, verbose=False)[0]
                    n_down = 0
                    # drawn directly onto `annotated` (already carrying the PPE overlay)
                    # rather than via .plot(), which would replace it -- this is what
                    # makes both detectors' boxes visible on the stream at once instead
                    # of only ever showing PPE's, which is a real gap this fixes.
                    for b in fall_results.boxes:
                        label = FALL_CLASS_NAMES[int(b.cls.item())]
                        if label != "down":
                            continue
                        n_down += 1
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, f"DOWN {float(b.conf.item()):.2f}", (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if n_down > 0:
                        self._log("fall", "fall_detected",
                                  f"{n_down} person(s) detected in a fallen/down position in {self.zone}")

                if self.run_fire_smoke and frame_idx % FIRE_SMOKE_EVERY_N_FRAMES == 0:
                    fire_smoke_results = _fire_smoke_model.predict(frame, conf=0.4, verbose=False)[0]
                    n_fire = n_smoke = 0
                    # drawn directly onto `annotated`, same reason as fall above -- multiple
                    # detectors' boxes need to coexist on one frame, only one can call .plot()
                    for b in fire_smoke_results.boxes:
                        label = FIRE_SMOKE_CLASS_NAMES[int(b.cls.item())]
                        color = (0, 140, 255) if label == "fire" else (160, 160, 160)  # orange / gray
                        if label == "fire":
                            n_fire += 1
                        else:
                            n_smoke += 1
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"{label.upper()} {float(b.conf.item()):.2f}", (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if n_fire > 0 or n_smoke > 0:
                        detail = " and ".join(p for p in [f"{n_fire} fire region(s)" if n_fire else None,
                                                           f"{n_smoke} smoke region(s)" if n_smoke else None] if p)
                        self._log("fire_smoke", "fire_smoke_detected", f"Detected {detail} in {self.zone}")
            except Exception as e:
                self.error = str(e)
                annotated = frame

            ok2, buf = cv2.imencode(".jpg", annotated)
            if ok2:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
            self.frames_processed += 1
            frame_idx += 1

        cap.release()
        self._running = False


_sessions: dict[str, VisionSession] = {}
_sessions_lock = threading.Lock()


def create_session(**kwargs) -> VisionSession:
    session = VisionSession(**kwargs)
    with _sessions_lock:
        _sessions[session.session_id] = session
    session.start()
    return session


def get_session(session_id: str) -> VisionSession | None:
    with _sessions_lock:
        return _sessions.get(session_id)


def stop_session(session_id: str):
    session = get_session(session_id)
    if session:
        session.stop()
        with _sessions_lock:
            _sessions.pop(session_id, None)
