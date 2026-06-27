"""
Common interface every vision detector implements, so the orchestrator/permit-correlation
logic has one consistent hook to call regardless of which model backs a given detector --
adding a new hazard category (fire/smoke, fall/man-down, ...) means dropping in a new
file here, not touching the agent pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RawDetection:
    label: str
    confidence: float


@dataclass
class DetectorResult:
    detector_name: str
    raw_detections: list[RawDetection]
    event: str | None  # semantic event this detector raises, e.g. "ppe_violation", None if benign
    event_detail: str  # human-readable explanation, used directly in violation reasons/LLM prompts


class Detector(Protocol):
    name: str

    def predict(self, image_path: str, context: dict) -> DetectorResult:
        """context carries whatever the detector needs beyond the image -- e.g.
        zone-intrusion needs the run's active permits, PPE doesn't need anything extra."""
        ...
