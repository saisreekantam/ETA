"""
Registry tying all vision detectors to the agent pipeline. New detectors register here
and are immediately available to compound_risk/permit_correlation without further wiring.
"""
from __future__ import annotations

from vision.detectors.base import DetectorResult
from vision.detectors.fall import FallDetector
from vision.detectors.fire_smoke import FireSmokeDetector
from vision.detectors.ppe import PPEDetector
from vision.detectors.zone_intrusion import ZoneIntrusionDetector

ACTIVE_DETECTORS = {
    "ppe": PPEDetector(),
    "zone_intrusion": ZoneIntrusionDetector(),
    "fall": FallDetector(),
    "fire_smoke": FireSmokeDetector(),
}


def run_detectors(image_path: str, context: dict, detector_names: list[str] | None = None) -> list[DetectorResult]:
    names = detector_names or list(ACTIVE_DETECTORS.keys())
    return [ACTIVE_DETECTORS[name].predict(image_path, context) for name in names if name in ACTIVE_DETECTORS]
