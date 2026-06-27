"""
PPE compliance detector -- wraps the fine-tuned RT-DETR (head/helmet/person, see
vision/ppe_detect.py and vision/rtdetr_finetune.py). Raises "ppe_violation" when a head
is detected with no corresponding helmet detection in the same frame. The "person" class
from this checkpoint is unreliable (test mAP50 0.01, see vision/weights/ -- this dataset's
person annotations are too sparse) so it's deliberately not used for anything here.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

from vision.detectors.base import DetectorResult, RawDetection

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "vision" / "weights" / "rtdetr_ppe.pt"
CLASS_NAMES = {0: "head", 1: "helmet", 2: "person"}

_model = None


def _lazy_load():
    global _model
    if _model is None:
        _model = RTDETR(str(WEIGHTS))


class PPEDetector:
    name = "ppe"

    def predict(self, image_path: str, context: dict) -> DetectorResult:
        _lazy_load()
        results = _model.predict(image_path, conf=0.4, verbose=False)
        r = results[0]

        raw = [RawDetection(label=CLASS_NAMES[int(b.cls.item())], confidence=round(float(b.conf.item()), 3))
               for b in r.boxes]
        labels = [d.label for d in raw]
        n_heads_no_helmet = labels.count("head")  # if any heads at all and no helmet class present

        if n_heads_no_helmet > 0 and "helmet" not in labels:
            return DetectorResult(
                detector_name=self.name, raw_detections=raw, event="ppe_violation",
                event_detail=(f"{n_heads_no_helmet} worker(s) detected without helmets "
                               f"(RT-DETR confidence {max((d.confidence for d in raw), default=0):.2f})"),
            )
        return DetectorResult(detector_name=self.name, raw_detections=raw, event=None,
                               event_detail="PPE compliant: helmet(s) detected, no bare-head violation")
