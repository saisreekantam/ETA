"""
Full fine-tune (no LoRA/adapters -- all weights updated) of RT-DETR on the Hard Hat
Workers dataset (head / helmet / person), for the PPE/CCTV vision agent. Meant to run
on a real GPU (the project's RTX 6000 Ada boxes), not the local Mac.

Run: `python -m vision.rtdetr_finetune` from industrial-safety-intel/, on the GPU host.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "ppe_vision" / "raw" / "data.yaml"

MODEL = "rtdetr-l.pt"  # pretrained checkpoint, fine-tuned from here -- full fine-tune
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
        device=1,  # GPU1 is the free 48GB card at training time -- GPU0 had another job running
        project=str(REPO_ROOT / "vision" / "runs"),
        name="rtdetr_ppe_finetune",
        freeze=[],  # explicit: no frozen layers, no adapters -- full fine-tune
        patience=15,
    )

    metrics = model.val(data=str(DATA_YAML), split="test")
    print("Test set metrics:", metrics.results_dict)


if __name__ == "__main__":
    main()
