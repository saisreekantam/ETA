"""
Fire/smoke detector -- wraps the fine-tuned RT-DETR (smoke / fire, see
vision/rtdetr_finetune_fire_smoke.py, trained on D-Fire after the first attempt on a
Roboflow set hit a real annotation-quality ceiling -- see that script's docstring).
Raises "fire_smoke_detected" if either class fires; fire and smoke are reported
separately in the detail string since a confirmed fire is a different severity than
smoke alone (smoke can be a false trigger from steam/dust, fire is unambiguous).

Test set quality (full D-Fire test split, 4302 images): smoke P=0.79 R=0.77 mAP50=0.78
mAP50-95=0.37, fire P=0.71 R=0.54 mAP50=0.59 mAP50-95=0.24 -- see
vision/weights/rtdetr_fire_smoke.pt training log for the full breakdown.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

from vision.detectors.base import DetectorResult, RawDetection

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "vision" / "weights" / "rtdetr_fire_smoke.pt"
CLASS_NAMES = {0: "smoke", 1: "fire"}

_model = None


def _lazy_load():
    global _model
    if _model is None:
        _model = RTDETR(str(WEIGHTS))


class FireSmokeDetector:
    name = "fire_smoke"

    def predict(self, image_path: str, context: dict) -> DetectorResult:
        _lazy_load()
        results = _model.predict(image_path, conf=0.4, verbose=False)
        r = results[0]

        raw = [RawDetection(label=CLASS_NAMES[int(b.cls.item())], confidence=round(float(b.conf.item()), 3))
               for b in r.boxes]
        labels = [d.label for d in raw]
        n_fire, n_smoke = labels.count("fire"), labels.count("smoke")

        if n_fire > 0 or n_smoke > 0:
            parts = []
            if n_fire:
                parts.append(f"{n_fire} fire region(s) (confidence {max(d.confidence for d in raw if d.label == 'fire'):.2f})")
            if n_smoke:
                parts.append(f"{n_smoke} smoke region(s) (confidence {max(d.confidence for d in raw if d.label == 'smoke'):.2f})")
            return DetectorResult(
                detector_name=self.name, raw_detections=raw, event="fire_smoke_detected",
                event_detail="Detected " + " and ".join(parts) + ".",
            )
        return DetectorResult(detector_name=self.name, raw_detections=raw, event=None,
                               event_detail="No fire or smoke detected.")
