"""
Generates the synthetic permit-to-work and worker-presence overlay keyed to the
manifest produced by fault_injection.py. This is what makes the benchmark "compound"
in the PS1 sense (permit + sensor + zone), not just a multivariate sensor problem.

Overlap rule per run (see docs/scenario-definitions.md):
  - compound-positive runs: permit + worker presence scheduled to overlap the fault
    window in the scenario's mapped zone (the ground-truth ground-truth compound label).
  - normal / single_a / single_b runs: split across four negative-control patterns so the
    model can't just learn "permit present => risk" or "fault present => risk" in isolation:
      (a) no fault, permit present in zone, no overlap needed (permit alone isn't risk)
      (b) single fault only, no permit in zone at all
      (c) single fault only, permit present in zone (tests for permit-only over-triggering)
      (d) no fault, no permit (pure normal operation)

Run directly: `python simulator/permit_shiftlog_synth.py` (requires manifest.csv to exist).
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from simulator.fault_injection import N_SAMPLES, SYNTHETIC_DIR  # noqa: E402

SAMPLE_MINUTES = 3  # tep2py's fixed sample period


def _sample_to_iso(sample_idx: int, run_date: pd.Timestamp) -> str:
    return (run_date + pd.Timedelta(minutes=SAMPLE_MINUTES * sample_idx)).isoformat()


def _negative_control_pattern(rng: np.random.Generator) -> str:
    return rng.choice(["a_permit_no_overlap_needed", "b_no_permit", "c_permit_present", "d_no_permit"])


def generate_permit_and_presence(row: pd.Series, rng: np.random.Generator, run_date: pd.Timestamp) -> tuple[dict, dict]:
    """Returns (permit_record, worker_presence_record) as plain dicts matching agents/state.py schema."""
    zone = row["zone"]
    permit_type = row["permit_type"]
    is_compound = bool(row["compound_active"])
    onset = row["ground_truth_onset_sample"]

    if is_compound:
        # deliberately overlap: permit + worker presence span the fault window
        onset = int(onset)
        permit_from, permit_to = max(0, onset - 5), min(N_SAMPLES - 1, onset + 25)
        presence_from, presence_to = onset, min(N_SAMPLES - 1, onset + 20)
        has_permit, has_presence = True, True
    else:
        pattern = _negative_control_pattern(rng)
        mid = N_SAMPLES // 2
        if pattern == "a_permit_no_overlap_needed":
            permit_from, permit_to = max(0, mid - 10), mid + 10
            has_permit, has_presence = True, False
        elif pattern == "b_no_permit":
            has_permit, has_presence = False, False
            permit_from, permit_to = 0, 0
        elif pattern == "c_permit_present":
            permit_from, permit_to = max(0, mid - 15), min(N_SAMPLES - 1, mid + 15)
            presence_from, presence_to = mid, mid + 10
            has_permit, has_presence = True, True
        else:  # d_no_permit
            has_permit, has_presence = False, False
            permit_from, permit_to = 0, 0
        if not has_presence:
            presence_from, presence_to = 0, 0

    permit = {
        "permit_id": f"PTW-{uuid.uuid4().hex[:8]}",
        "permit_type": permit_type if has_permit else "general",
        "zone": zone,
        "valid_from": _sample_to_iso(permit_from, run_date) if has_permit else None,
        "valid_to": _sample_to_iso(permit_to, run_date) if has_permit else None,
        "status": "active" if has_permit else "expired",
        "run_id": row["run_id"],
        "has_permit": has_permit,
        # explicit sample indices (not just ISO timestamps) so any streaming/replay
        # re-inference can gate visibility exactly, without inverting timestamps back
        # through the run's synthetic base-date offset -- see scripts/replay.py.
        "from_sample": permit_from if has_permit else None,
        "to_sample": permit_to if has_permit else None,
    }
    presence = {
        "worker_id": f"W-{uuid.uuid4().hex[:6]}",
        "zone": zone,
        "entry_time": _sample_to_iso(presence_from, run_date) if has_presence else None,
        "exit_time": _sample_to_iso(presence_to, run_date) if has_presence else None,
        "run_id": row["run_id"],
        "has_presence": has_presence,
        "from_sample": presence_from if has_presence else None,
        "to_sample": presence_to if has_presence else None,
    }
    return permit, presence


def generate_overlay(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(SYNTHETIC_DIR / "manifest.csv")
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2026-01-01")

    permits, presences = [], []
    for i, row in manifest.iterrows():
        run_date = base_date + pd.Timedelta(hours=i)  # spread runs across distinct "days"
        permit, presence = generate_permit_and_presence(row, rng, run_date)
        permits.append(permit)
        presences.append(presence)

    permits_df = pd.DataFrame(permits)
    presences_df = pd.DataFrame(presences)

    out_dir = SYNTHETIC_DIR.parent  # data/
    permits_path = out_dir / "permits" / "permits.parquet"
    presences_path = out_dir / "shiftlogs" / "shiftlogs.parquet"
    permits_path.parent.mkdir(parents=True, exist_ok=True)
    presences_path.parent.mkdir(parents=True, exist_ok=True)
    permits_df.to_parquet(permits_path, index=False)
    presences_df.to_parquet(presences_path, index=False)
    print(f"Wrote {len(permits_df)} permit records to {permits_path}")
    print(f"Wrote {len(presences_df)} worker-presence records to {presences_path}")
    return permits_df, presences_df


if __name__ == "__main__":
    generate_overlay()
