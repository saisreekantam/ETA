"""
Fire/smoke detector -- STUB. Waiting on a sourced dataset (D-Fire, 21k images, see
project notes) to fine-tune a second RT-DETR the same way vision/rtdetr_finetune.py did
for PPE. The interface below is final; once vision/weights/rtdetr_fire_smoke.pt exists,
delete the NotImplementedError and mirror ppe.py's structure exactly -- that's the point
of the pluggable registry, this file is the only thing that needs to change.
"""
from __future__ import annotations

from vision.detectors.base import DetectorResult


class FireSmokeDetector:
    name = "fire_smoke"

    def predict(self, image_path: str, context: dict) -> DetectorResult:
        raise NotImplementedError(
            "fire/smoke model not trained yet -- see vision/weights/rtdetr_fire_smoke.pt "
            "(does not exist). Train with vision/rtdetr_finetune.py pointed at the D-Fire "
            "dataset once data/fire_smoke/raw/ is populated, then mirror vision/detectors/ppe.py."
        )
