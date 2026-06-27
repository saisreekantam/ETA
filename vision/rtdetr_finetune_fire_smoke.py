"""
Full fine-tune (no LoRA/adapters -- all weights updated) of RT-DETR on D-Fire (smoke /
fire, ~21k images, Venancio et al. 2022), for the fire/smoke CCTV vision agent
(vision/detectors/fire_smoke.py). Meant to run on a real GPU (the project's RTX 6000 Ada
boxes), not the local Mac.

Switched from the original Roboflow community dataset (data/fire_smoke/raw/) after
inspecting it visually: that dataset's boxes only covered small arbitrary patches of
visible fire/smoke, leaving most of the actually-burning frame labeled as background --
a real annotation-quality ceiling, not a training bug, confirmed by drawing ground-truth
boxes on sample images. D-Fire's boxes (verified the same way) are exhaustive --
multi-box images tile the full visible smoke plume and each distinct flame region.
Class order is the OPPOSITE of the old dataset: 0=smoke, 1=fire here.

Run: `python -m vision.rtdetr_finetune_fire_smoke` from industrial-safety-intel/, on the GPU host.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "fire_smoke" / "raw_dfire" / "data.yaml"

MODEL = "rtdetr-l.pt"
EPOCHS = 60
IMG_SIZE = 640
BATCH = 16


def main():
    model = RTDETR(MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=0,
        project=str(REPO_ROOT / "vision" / "runs"),
        name="rtdetr_fire_smoke_dfire_finetune",
        freeze=[],
        patience=15,
    )

    metrics = model.val(data=str(DATA_YAML), split="test")
    print("Test set metrics:", metrics.results_dict)


if __name__ == "__main__":
    main()
