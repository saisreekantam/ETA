"""
Runs the active vision detectors once on a set of staged demo images (real photos
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

Bakes FIVE violation frames (a "head" detected but no "helmet" -- used for
compound-positive scenarios) and THREE compliant frames (helmets present, no bare
head -- used for normal-condition scenarios), so the vision evidence is narratively tied
to whether the scenario is actually hazardous AND different runs can show different
frames (scripts/demo_scenario_runner.py picks one per run_id) instead of every demo
reusing the same photo.

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
# dataset mixes plenty of non-industrial photos (offices, bedrooms, handshakes,
# politicians) that happen to satisfy the class-label filter, so picking the first
# match isn't reliable. Each list spans different sites and worker counts so demo
# runs don't all show the same photo.
CURATED_VIOLATIONS = [
    # outdoor work site, dirt lot with vehicles
    "006410_jpg.rf.bc6eb1c50dddd7b24da72dc53c34714a",
    # demolition site, 3 workers reviewing plans amid rubble
    "005641_jpg.rf.e8f647a4d5ddee7031cbe5cc58145fe8",
    # gravel/quarry site walkdown, 8 people, none helmeted
    "005657_jpg.rf.3a6fb96684576db0282d408ccc32b67c",
    # earthworks site, 9 people around a drawing, aggregate piles + trucks behind
    "006042_jpg.rf.ba9f28d93c57a0f3576377c3ee431fd2",
    # red-earth construction site, 4 people, crane sections in background
    "006890_jpg.rf.75cc1ffc4e48c8a9c36666886d7b58d4",
]
CURATED_COMPLIANT = [
    # tunnel construction, ~8 helmeted workers on track works
    "005298_jpg.rf.647d148af5d961d8bbc041f172247170",
    # open-pit mine, 2 workers in hi-vis + helmets, drill rigs below
    "005746_jpg.rf.3ba1a2eeeb6c932bb016c3b6c6e3a0a4",
    # pile-driving crew close-up, 3 helmeted workers in life vests
    "006052_jpg.rf.a7c01d03fd72e2fd4e2251afa4ec46db",
]


def _label_classes(label_path: Path) -> set[int]:
    return {int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()}


def _pick_demo_images(n_violation: int = 5, n_compliant: int = 3) -> dict[str, Path]:
    violation = [TEST_LABELS_DIR / f"{s}.txt" for s in CURATED_VIOLATIONS]
    compliant = [TEST_LABELS_DIR / f"{s}.txt" for s in CURATED_COMPLIANT]
    violation = [p for p in violation if p.exists()][:n_violation]
    compliant = [p for p in compliant if p.exists()][:n_compliant]

    # Fallback auto-pick if curated files are missing (e.g. a re-downloaded dataset with
    # different Roboflow hashes). Compliant requires helmets AND no bare head -- an image
    # with both is a violation, not a compliant frame.
    for label_path in sorted(TEST_LABELS_DIR.glob("*.txt")):
        if len(violation) >= n_violation and len(compliant) >= n_compliant:
            break
        if label_path in violation or label_path in compliant:
            continue
        classes = _label_classes(label_path)
        if 0 in classes and 1 not in classes and len(violation) < n_violation:
            violation.append(label_path)
        elif 1 in classes and 0 not in classes and len(compliant) < n_compliant:
            compliant.append(label_path)

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

    # sanity check: the detector's verdict must agree with each frame's intended role --
    # a "violation_*" frame the detector calls compliant would make the demo incoherent.
    for tag, entry in manifest.items():
        expected = tag.startswith("violation")
        if entry["has_violation"] != expected:
            print(f"WARNING: {tag} ({entry['source_image']}) detector verdict "
                  f"has_violation={entry['has_violation']} contradicts its role -- consider swapping it out")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} cached detections to {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
