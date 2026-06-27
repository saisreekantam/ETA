"""
Fall / man-down detector -- wraps the fine-tuned RT-DETR (crouching_or_bending / down /
person, see vision/rtdetr_finetune_fall.py). Raises "fall_detected" when the "down" class
(a person in a fallen/lying position) is detected. "crouching_or_bending" is reported as
context evidence (people clustering/bending over someone is itself a signal of an
incident) but does not independently raise the event -- only an actual "down" detection
does, to avoid false-alarming on people who are simply crouching to work.

Test set quality (real numbers, not placeholders): down P=0.72 R=0.71 mAP50=0.74,
crouching_or_bending P=0.68 R=0.81 mAP50=0.76, person P=0.70 R=0.55 mAP50=0.62 -- see
vision/weights/rtdetr_fall.pt training log for the full breakdown.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

from vision.detectors.base import DetectorResult, RawDetection

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "vision" / "weights" / "rtdetr_fall.pt"
CLASS_NAMES = {0: "crouching_or_bending", 1: "down", 2: "person"}

_model = None


def _lazy_load():
    global _model
    if _model is None:
        _model = RTDETR(str(WEIGHTS))


class FallDetector:
    name = "fall"

    def predict(self, image_path: str, context: dict) -> DetectorResult:
        _lazy_load()
        results = _model.predict(image_path, conf=0.4, verbose=False)
        r = results[0]

        raw = [RawDetection(label=CLASS_NAMES[int(b.cls.item())], confidence=round(float(b.conf.item()), 3))
               for b in r.boxes]
        labels = [d.label for d in raw]
        n_down = labels.count("down")

        if n_down > 0:
            down_confidences = [d.confidence for d in raw if d.label == "down"]
            bystander_note = ""
            if labels.count("crouching_or_bending") > 0:
                bystander_note = f" ({labels.count('crouching_or_bending')} bystander(s) also bending over them)"
            return DetectorResult(
                detector_name=self.name, raw_detections=raw, event="fall_detected",
                event_detail=(f"{n_down} person(s) detected in a fallen/down position"
                               f"{bystander_note} (RT-DETR confidence {max(down_confidences):.2f})"),
            )
        return DetectorResult(detector_name=self.name, raw_detections=raw, event=None,
                               event_detail="No fallen person detected.")
