"""
PPE compliance detector -- wraps the fine-tuned RT-DETR (head/helmet/person, see
vision/ppe_detect.py and vision/rtdetr_finetune.py). The "person" class from this
checkpoint is unreliable (test mAP50 0.01, see vision/weights/ -- this dataset's
person annotations are too sparse) so it's deliberately not used for anything here.

Violation logic is PER HEAD, not per frame: a bare-head box counts as a violation
unless a helmet box overlaps it. The original frame-level rule ("any helmet anywhere
=> compliant") let one helmeted worker mask every bare-headed colleague in the frame.

Two guards against the failure mode observed on live webcam input (dark hair detected
as "helmet", suppressing real violations -- out-of-distribution for the construction
-site training data):
  - helmet needs conf >= 0.60 to count, vs 0.40 for head. Validated on 250 test images:
    helmet precision is 1.000 at both thresholds, recall drops only 0.991 -> 0.987, so
    the on-dataset cost is negligible while low-confidence live hair-FPs are dropped.
  - head/helmet are mutually exclusive on one person, so a helmet box that mostly
    overlaps a head box is a contradiction; keep whichever the model is more confident
    about instead of always believing the helmet.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

from vision.detectors.base import DetectorResult, RawDetection

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "vision" / "weights" / "rtdetr_ppe.pt"
CLASS_NAMES = {0: "head", 1: "helmet", 2: "person"}

HEAD_CONF = 0.40
HELMET_CONF = 0.60
CONFLICT_IOU = 0.50  # head/helmet overlap that counts as "same head" (contradiction)
COVER_IOU = 0.20     # helmet overlap that counts as "this head is wearing it"

_model = None


def _lazy_load():
    global _model
    if _model is None:
        _model = RTDETR(str(WEIGHTS))


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def analyze_ppe_result(result) -> tuple[list[RawDetection], int, list[float]]:
    """Shared by PPEDetector (cached demo frames) and vision/live_inference.py (video)
    so both paths apply identical thresholds and per-head logic. Takes one ultralytics
    result predicted at a LOW conf (<= HEAD_CONF) and returns
    (kept raw detections, n_bare_heads_without_helmet, kept helmet confidences)."""
    heads, helmets = [], []
    for b in result.boxes:
        cls, conf = int(b.cls.item()), float(b.conf.item())
        xyxy = b.xyxy[0].tolist()
        if cls == 0 and conf >= HEAD_CONF:
            heads.append((xyxy, conf))
        elif cls == 1 and conf >= HELMET_CONF:
            helmets.append((xyxy, conf))

    # contradiction pass: a helmet mostly covering a head box is the same head seen as
    # both classes -- keep the higher-confidence reading
    for helm_box, helm_conf in list(helmets):
        for head_box, head_conf in list(heads):
            if _iou(helm_box, head_box) >= CONFLICT_IOU:
                if helm_conf >= head_conf:
                    heads = [h for h in heads if h[0] is not head_box]
                else:
                    helmets = [h for h in helmets if h[0] is not helm_box]

    uncovered = [h for h in heads
                 if not any(_iou(h[0], helm[0]) >= COVER_IOU for helm in helmets)]

    raw = ([RawDetection(label="head", confidence=round(c, 3)) for _, c in heads]
           + [RawDetection(label="helmet", confidence=round(c, 3)) for _, c in helmets])
    return raw, len(uncovered), [c for _, c in helmets]


class PPEDetector:
    name = "ppe"

    def predict(self, image_path: str, context: dict) -> DetectorResult:
        _lazy_load()
        results = _model.predict(image_path, conf=0.25, verbose=False)
        raw, n_violations, _ = analyze_ppe_result(results[0])

        if n_violations > 0:
            return DetectorResult(
                detector_name=self.name, raw_detections=raw, event="ppe_violation",
                event_detail=(f"{n_violations} worker(s) detected without helmets "
                               f"(RT-DETR confidence {max((d.confidence for d in raw), default=0):.2f})"),
            )
        return DetectorResult(detector_name=self.name, raw_detections=raw, event=None,
                               event_detail="PPE compliant: no bare-head detection lacking a helmet")
