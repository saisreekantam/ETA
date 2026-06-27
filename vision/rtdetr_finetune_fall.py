"""
Full fine-tune (no LoRA/adapters -- all weights updated) of RT-DETR on the fall-detection
dataset (crouching_or_bending / down / person), for the fall/man-down CCTV vision agent.
Meant to run on a real GPU (the project's RTX 6000 Ada boxes), not the local Mac.

The hazard-indicating class is "down" (a fallen/lying person); "crouching_or_bending" is
kept as context (people clustering/bending over someone is itself a signal) -- see
data/fall_detection/raw/data.yaml for the class-naming caveat.

Run: `python -m vision.rtdetr_finetune_fall` from industrial-safety-intel/, on the GPU host.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "fall_detection" / "raw" / "data.yaml"

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
        device=1,
        project=str(REPO_ROOT / "vision" / "runs"),
        name="rtdetr_fall_finetune",
        freeze=[],
        patience=15,
    )

    metrics = model.val(data=str(DATA_YAML), split="test")
    print("Test set metrics:", metrics.results_dict)


if __name__ == "__main__":
    main()
