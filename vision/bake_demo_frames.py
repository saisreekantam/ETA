"""
Runs the active vision detectors once on a handful of staged demo images (real photos
from the Hard Hat Workers test split -- we have no actual plant CCTV footage, so these
are explicitly framed as demo frames, not a live feed) and caches detections + annotated
images. Per the plan: CCTV/PPE detection is pre-baked, not live -- this script is the
"bake" step; agent nodes consume the cached JSON as if it were a live camera call.

Bakes TWO things per image: the PPE detector's output (head/helmet -> ppe_violation) and
the zone-intrusion detector's RAW person count only (the model-inference part). The
zone-intrusion EVENT itself (unauthorized_entry vs covered-by-permit) is intentionally
NOT baked here -- it depends on which run's permit data you cross-check against, which
is run-specific, not image-specific. That cheap cross-check happens live in
scripts/demo_scenario_runner.py using this cached person count, with zero extra model
inference.

Picks one frame with a "head" detected but no "helmet" (a PPE violation -- used for
compound-positive scenarios) and one frame with "helmet" present (compliant -- used for
normal-condition scenarios), so the vision evidence is narratively tied to whether the
scenario is actually hazardous.

Run: `python -m vision.bake_demo_frames`
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = REPO_ROOT / "data" / "ppe_vision" / "raw" / "test" / "images"
TEST_LABELS_DIR = REPO_ROOT / "data" / "ppe_vision" / "raw" / "test" / "labels"
OUT_DIR = REPO_ROOT / "data" / "ppe_vision" / "cached_outputs"

# Manually curated (visually checked) from the auto-selected candidate pool -- the
# dataset mixes plenty of non-industrial photos (offices, bedrooms, handshakes) that
# happen to satisfy the class-label filter, so picking the first match isn't reliable.
# These are an actual outdoor work site (dirt lot, vehicles) and an actual construction
# site (rebar, concrete pour) respectively.
CURATED_VIOLATION = "006410_jpg.rf.bc6eb1c50dddd7b24da72dc53c34714a"


def _pick_demo_images(n_violation: int = 1, n_compliant: int = 1) -> dict[str, Path]:
    violation, compliant = [], []
    curated_path = TEST_LABELS_DIR / f"{CURATED_VIOLATION}.txt"
    if curated_path.exists():
        violation.append(curated_path)

    for label_path in TEST_LABELS_DIR.glob("*.txt"):
        classes = {int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()}
        if 0 in classes and 1 not in classes and len(violation) < n_violation and label_path not in violation:
            violation.append(label_path)
        elif 1 in classes and len(compliant) < n_compliant:
            compliant.append(label_path)
        if len(violation) >= n_violation and len(compliant) >= n_compliant:
            break

    picks = {}
    for i, lbl in enumerate(violation):
        picks[f"violation_{i}"] = TEST_IMAGES_DIR / (lbl.stem + ".jpg")
    for i, lbl in enumerate(compliant):
        picks[f"compliant_{i}"] = TEST_IMAGES_DIR / (lbl.stem + ".jpg")
    return picks


def main():
    from ultralytics import RTDETR

    from vision.detectors.ppe import WEIGHTS as PPE_WEIGHTS
    from vision.detectors.ppe import PPEDetector
    from vision.detectors.zone_intrusion import ZoneIntrusionDetector

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ppe_model = RTDETR(str(PPE_WEIGHTS))
    ppe_detector = PPEDetector()
    person_detector = ZoneIntrusionDetector()

    picks = _pick_demo_images()
    manifest = {}
    for tag, img_path in picks.items():
        ppe_result = ppe_detector.predict(str(img_path), context={})
        person_result = person_detector.predict(str(img_path), context={"has_active_permit_for_zone": True})

        annotated_path = OUT_DIR / f"{tag}.jpg"
        ppe_model.predict(str(img_path), conf=0.4, verbose=False)[0].save(filename=str(annotated_path))

        manifest[tag] = {
            "source_image": img_path.name,
            "annotated_image": annotated_path.name,
            "detections": [d.label for d in ppe_result.raw_detections],
            "confidences": [d.confidence for d in ppe_result.raw_detections],
            "has_violation": ppe_result.event == "ppe_violation",
            "person_count": len(person_result.raw_detections),
            "person_confidences": [d.confidence for d in person_result.raw_detections],
        }
        print(f"{tag}: ppe={manifest[tag]['detections']} person_count={manifest[tag]['person_count']} "
              f"-> {annotated_path.name}")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} cached detections to {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
