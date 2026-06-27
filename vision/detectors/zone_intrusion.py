"""
Zone-intrusion detector -- needs no fine-tuning or new dataset, unlike the other
detectors here. Uses a stock COCO-pretrained RT-DETR (its "person" class is robust,
unlike our PPE-finetuned checkpoint's broken person class) purely to confirm a human
is physically present in the frame, then cross-references that against the run's active
permits for the zone: a person detected with NO matching active permit in that zone is
"unauthorized_entry". This is the genuinely new logic -- person detection itself is a
commodity capability; the value is the permit/zone cross-check, which only this project's
permit system makes possible.

Known gap: the synthetic data generator (simulator/permit_shiftlog_synth.py) never
produces has_presence=True with has_permit=False -- worker presence is always
permit-linked in every scenario/negative-control pattern we generate. So the
"unauthorized_entry" branch below is real, working code, but doesn't currently fire on
any of our demo scenarios; it would matter for a real deployment where someone enters a
zone without ever filing a permit, a case our synthetic world doesn't model. Documented
rather than retrofitted, to avoid fabricating a triggering scenario just to show it off.
"""
from __future__ import annotations

from ultralytics import RTDETR

from vision.detectors.base import DetectorResult, RawDetection

COCO_PERSON_CLASS_ID = 0

_model = None


def _lazy_load():
    global _model
    if _model is None:
        _model = RTDETR("rtdetr-l.pt")  # stock COCO-pretrained checkpoint, downloaded once


class ZoneIntrusionDetector:
    name = "zone_intrusion"

    def predict(self, image_path: str, context: dict) -> DetectorResult:
        """context must provide 'zone' and 'has_active_permit_for_zone' (bool) --
        populated by the caller from the same permit records the GNN/permit-correlation
        node already use, so this detector adds no new data dependency."""
        _lazy_load()
        results = _model.predict(image_path, conf=0.4, verbose=False)
        r = results[0]

        raw = [RawDetection(label="person", confidence=round(float(b.conf.item()), 3))
               for b in r.boxes if int(b.cls.item()) == COCO_PERSON_CLASS_ID]

        zone = context.get("zone", "unknown_zone")
        has_permit = context.get("has_active_permit_for_zone", False)

        if raw and not has_permit:
            return DetectorResult(
                detector_name=self.name, raw_detections=raw, event="unauthorized_entry",
                event_detail=(f"{len(raw)} person(s) detected in {zone} with NO active "
                               f"permit authorizing presence there."),
            )
        if raw and has_permit:
            return DetectorResult(
                detector_name=self.name, raw_detections=raw, event=None,
                event_detail=f"{len(raw)} person(s) detected in {zone}, covered by an active permit.",
            )
        return DetectorResult(detector_name=self.name, raw_detections=raw, event=None,
                               event_detail=f"No person detected in {zone}.")
