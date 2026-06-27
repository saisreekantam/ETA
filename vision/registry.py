"""
Registry tying all vision detectors to the agent pipeline. New detectors register here
and are immediately available to compound_risk/permit_correlation without further wiring.
fire_smoke is omitted from ACTIVE_DETECTORS until its model exists -- see
vision/detectors/fire_smoke.py (training on D-Fire in progress).
"""
from __future__ import annotations

from vision.detectors.base import DetectorResult
from vision.detectors.fall import FallDetector
from vision.detectors.ppe import PPEDetector
from vision.detectors.zone_intrusion import ZoneIntrusionDetector

ACTIVE_DETECTORS = {
    "ppe": PPEDetector(),
    "zone_intrusion": ZoneIntrusionDetector(),
    "fall": FallDetector(),
}


def run_detectors(image_path: str, context: dict, detector_names: list[str] | None = None) -> list[DetectorResult]:
    names = detector_names or list(ACTIVE_DETECTORS.keys())
    return [ACTIVE_DETECTORS[name].predict(image_path, context) for name in names if name in ACTIVE_DETECTORS]
