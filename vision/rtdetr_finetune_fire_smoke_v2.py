"""
Round-2 fine-tune of the fire/smoke RT-DETR, targeting the fire class specifically:
round 1 (vision/rtdetr_finetune_fire_smoke.py) early-stopped at epoch ~30 with test
mAP50 fire=0.59 vs smoke=0.78. Fire boxes in D-Fire are small and their appearance
varies far more than smoke (orange/red/yellow flames, day/night, indoor/outdoor), so
this round:

  - CONTINUES from the deployed round-1 checkpoint (vision/weights/rtdetr_fire_smoke.pt)
    instead of restarting from COCO weights -- "more epochs at a lower learning rate".
  - Drops the LR two orders below the ultralytics default (lr0=1e-4, AdamW, cosine
    schedule) so the extra epochs refine rather than wipe the converged weights.
  - Turns the color augmentation up specifically for fire: hsv_h 0.015->0.05 (hue swings
    cover the orange<->red<->yellow flame range), hsv_s 0.7->0.9, hsv_v 0.4->0.6
    (saturation/brightness cover night fires vs daylight haze). Smoke is near-achromatic
    so the stronger jitter costs it little, which is the asymmetry we want.

Third option from the plan (mixing in FASDD/VisiFire) intentionally NOT done here --
different labeling conventions would need the same visual box-quality audit D-Fire got
(see round-1 docstring) before they can be trusted, which isn't a quick win.

Run on the GPU host (worker1) from the synced dir:
  python rtdetr_finetune_fire_smoke_v2.py
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "data" / "fire_smoke" / "raw_dfire" / "data.yaml"
ROUND1_WEIGHTS = REPO_ROOT / "vision" / "weights" / "rtdetr_fire_smoke.pt"

EPOCHS = 40
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
        name="rtdetr_fire_smoke_dfire_v2",
        freeze=[],
        patience=15,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=2,
        hsv_h=0.05,
        hsv_s=0.9,
        hsv_v=0.6,
    )

    metrics = model.val(data=str(DATA_YAML), split="test")
    print("Test set metrics:", metrics.results_dict)
    # per-class breakdown -- the whole point of this round is the fire class
    for i, name in metrics.names.items():
        print(f"class {name}: mAP50={metrics.box.ap50[i]:.3f} mAP50-95={metrics.box.ap[i]:.3f}")


if __name__ == "__main__":
    main()
