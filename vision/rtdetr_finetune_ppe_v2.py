"""
Round-2 fine-tune of the PPE RT-DETR, targeting head/helmet confusion observed live:
dark voluminous hair gets detected as "helmet", which under the frame-level violation
rule (any helmet present => compliant) suppresses real violations. Same recipe that
worked for the fire/smoke round 2:

  - CONTINUES from the deployed round-1 checkpoint (vision/weights/rtdetr_ppe.pt)
    with lr0=1e-4 AdamW + cosine -- refine, don't relearn.
  - warmup disabled (warmup_epochs=0): the fire v2 run showed ultralytics' default
    warmup_bias_lr=0.1 momentarily wrecks a converged model (mAP50 0.68 -> 0.17 at
    epoch 1) before it recovers -- pointless when resuming, so skip it here.
  - Moderate HSV jitter (helmet shells span white/yellow/red/blue; hair doesn't) plus
    default geometric augs. The main lever for hair-vs-helmet is simply more gradient
    steps on the same exhaustive annotations at a refining LR.

Run on the GPU host (worker1) from the synced dir:
  python rtdetr_finetune_ppe_v2.py
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "ppe_vision" / "raw" / "data.yaml"
ROUND1_WEIGHTS = REPO_ROOT / "vision" / "weights" / "rtdetr_ppe.pt"

EPOCHS = 30
IMG_SIZE = 640
BATCH = 16


def main():
    model = RTDETR(str(ROUND1_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=0,
        project=str(REPO_ROOT / "vision" / "runs"),
        name="rtdetr_ppe_v2",
        freeze=[],
        patience=10,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=0,
        hsv_h=0.03,
        hsv_s=0.8,
        hsv_v=0.5,
    )

    metrics = model.val(data=str(DATA_YAML), split="test")
    print("Test set metrics:", metrics.results_dict)
    for i, name in metrics.names.items():
        print(f"class {name}: mAP50={metrics.box.ap50[i]:.3f} mAP50-95={metrics.box.ap[i]:.3f}")


if __name__ == "__main__":
    main()
