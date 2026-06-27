"""
Seeds the DB from the existing flat-file synthetic benchmark and RAG corpus -- this is
the one-time migration off "read CSVs/parquet on every request" onto real tables. Sensor
time-series data stays on disk (Run.sensor_data_path points at it); everything relational
(permits, presence, runs, RAG chunks+embeddings) moves into Postgres.

Idempotent: re-running wipes and re-seeds rather than accumulating duplicates, since this
is meant for (re-)initializing a dev/demo DB, not an incremental production loader.

Run: `python -m db.seed`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from db.auth import create_api_key  # noqa: E402
from db.models import (  # noqa: E402
    ApiKey,
    Base,
    Facility,
    Permit,
    RagChunk,
    RagDocument,
    Run,
    WorkerPresence,
    Zone,
)
from db.session import SessionLocal, engine  # noqa: E402

DEMO_FACILITY_NAME = "Demo Steel & Chemical Plant"

# mirrors server/main.py's ZONE_LAYOUT -- kept in sync manually since one is DB seed
# data and the other is frontend layout metadata; if you change one, change both.
ZONE_LAYOUT = {
    "feed_zone":        {"label": "Feed Systems",     "x": 40,  "y": 220, "w": 120, "h": 90},
    "reactor_zone":      {"label": "Reactor",          "x": 200, "y": 180, "w": 130, "h": 160},
    "condenser_zone":    {"label": "Condenser",        "x": 370, "y": 140, "w": 110, "h": 90},
    "separator_zone":    {"label": "Separator",        "x": 370, "y": 260, "w": 110, "h": 90},
    "stripper_zone":     {"label": "Stripper",         "x": 520, "y": 200, "w": 110, "h": 140},
    "compressor_zone":   {"label": "Recycle Compressor", "x": 200, "y": 40, "w": 150, "h": 80},
    "control_room":      {"label": "Control Room",     "x": 660, "y": 40, "w": 110, "h": 80},
}


def _parse_dt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    ts = pd.Timestamp(value)
    return None if pd.isna(ts) else ts.to_pydatetime()


def seed():
    Base.metadata.create_all(engine)  # no-op if alembic already applied; safe either way
    db = SessionLocal()

    try:
        # TRUNCATE...CASCADE rather than ordered per-table deletes: zone_risk_scores,
        # permit_violations, incident_reports, audit_log, vision_events/sessions all
        # carry FKs back to runs/zones/permits (real rows exist here from actual /run
        # API calls during testing -- that's the incident-history feature working as
        # intended), so an ordered delete-by-table would hit FK violations. CASCADE
        # handles the dependency graph for us. facilities/api_keys are deliberately
        # NOT in this list -- see the get-or-create below, a handed-out API key must
        # keep working across reseeds.
        db.execute(text("""
            TRUNCATE runs, permits, worker_presence, zone_risk_scores, permit_violations,
                     incident_reports, audit_log, vision_events, vision_sessions, zones,
                     rag_chunks, rag_documents
            CASCADE
        """))
        db.commit()

        facility = db.query(Facility).filter_by(name=DEMO_FACILITY_NAME).one_or_none()
        if facility is None:
            facility = Facility(name=DEMO_FACILITY_NAME, location="Synthetic / TEP-derived")
            db.add(facility)
            db.flush()

        zone_by_key = {}
        for key, layout in ZONE_LAYOUT.items():
            zone = Zone(
                facility_id=facility.id, key=key, label=layout["label"],
                layout_x=layout["x"], layout_y=layout["y"], layout_w=layout["w"], layout_h=layout["h"],
                has_sensors=key != "control_room",
            )
            db.add(zone)
            zone_by_key[key] = zone
        db.flush()
        print(f"Seeded facility '{facility.name}' with {len(zone_by_key)} zones.")

        manifest = pd.read_csv(REPO_ROOT / "data" / "synthetic" / "manifest.csv")
        permits_df = pd.read_parquet(REPO_ROOT / "data" / "permits" / "permits.parquet")
        presences_df = pd.read_parquet(REPO_ROOT / "data" / "shiftlogs" / "shiftlogs.parquet")

        run_by_external_id = {}
        for _, row in manifest.iterrows():
            zone = zone_by_key[row["zone"]]
            run = Run(
                facility_id=facility.id, zone_id=zone.id, external_run_id=row["run_id"],
                scenario_id=row["scenario_id"], condition=row["condition"], permit_type=row["permit_type"],
                compound_active=bool(row["compound_active"]),
                ground_truth_onset_sample=(None if pd.isna(row["ground_truth_onset_sample"])
                                            else float(row["ground_truth_onset_sample"])),
                sensor_data_path=row["path"],
            )
            db.add(run)
            run_by_external_id[row["run_id"]] = run
        db.flush()
        print(f"Seeded {len(run_by_external_id)} runs.")

        for _, row in permits_df.iterrows():
            run = run_by_external_id.get(row["run_id"])
            if run is None:
                continue
            db.add(Permit(
                facility_id=facility.id, zone_id=run.zone_id, run_id=run.id,
                external_permit_id=row["permit_id"], permit_type=row["permit_type"],
                status=row["status"], valid_from=_parse_dt(row["valid_from"]), valid_to=_parse_dt(row["valid_to"]),
            ))

        for _, row in presences_df.iterrows():
            run = run_by_external_id.get(row["run_id"])
            if run is None:
                continue
            db.add(WorkerPresence(
                facility_id=facility.id, zone_id=run.zone_id, run_id=run.id,
                external_worker_id=row["worker_id"],
                entry_time=_parse_dt(row["entry_time"]), exit_time=_parse_dt(row["exit_time"]),
            ))
        db.commit()
        print(f"Seeded {len(permits_df)} permits and {len(presences_df)} worker-presence records.")

        chunks_path = REPO_ROOT / "rag" / "corpus" / "chunks.json"
        index_path = REPO_ROOT / "rag" / "corpus" / "index.npz"
        if chunks_path.exists() and index_path.exists():
            chunks = json.loads(chunks_path.read_text())
            embeddings = np.load(index_path)["embeddings"]

            doc_by_source = {}
            for chunk, embedding in zip(chunks, embeddings):
                source = chunk["source"]
                if source not in doc_by_source:
                    doc = RagDocument(source_label=chunk["citation"].split(" (p.")[0].split(", Section")[0],
                                       file_path=source)
                    db.add(doc)
                    db.flush()
                    doc_by_source[source] = doc
                db.add(RagChunk(
                    document_id=doc_by_source[source].id, text=chunk["text"],
                    citation=chunk["citation"], page=chunk["page"], embedding=embedding.tolist(),
                ))
            db.commit()
            print(f"Seeded {len(doc_by_source)} RAG documents, {len(chunks)} chunks with embeddings.")
        else:
            print("RAG corpus not found (rag/corpus/chunks.json + index.npz) -- skipping RAG seed.")

        # API keys are NOT wiped on reseed (unlike everything above) -- a key already
        # handed to a client/frontend must keep working across `db.seed` re-runs during
        # dev. Only create one if this facility doesn't have one yet.
        existing_key = db.query(ApiKey).filter_by(facility_id=facility.id, revoked=False).first()
        if existing_key is None:
            raw_key = create_api_key(db, name=f"{DEMO_FACILITY_NAME} demo key", facility_id=facility.id)
            print(f"\nCreated API key (shown once, store it now):\n  {raw_key}\n")
        else:
            print(f"\nFacility already has an active API key ('{existing_key.name}') -- not creating another.")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
