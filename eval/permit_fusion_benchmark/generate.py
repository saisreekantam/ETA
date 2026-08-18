"""
A second synthetic TEP benchmark, built specifically to close the gap the classical-
baseline validation exposed in the original one: there, "compound" meant two
simultaneous process faults vs one, so the compound label was fully recoverable from
sensors alone (a plain Random Forest on sensor stats matched the GNN's perfect score
with zero permit information). Here, every condition uses the SAME SINGLE fault (same
IDV, same onset distribution, same magnitude) -- the sensor trajectory in the scored
window is statistically identical across the true-positive and fault-having negative
controls. The ONLY thing that varies is whether a permit + worker-presence record is
temporally valid DURING the scored window (the last 30 of 120 samples, matching
models/gnn/graph_builder.py's WINDOW convention) -- so a sensor-only classifier is
information-theoretically incapable of separating true positives from the fault-having
negative controls; only a model that actually reads permit/presence timing can.

Five conditions per scenario:
  true_positive              -- fault active, permit+presence VALID during the scored window
  fp_fault_no_permit         -- SAME fault, no permit/presence at all
  fp_fault_permit_no_overlap -- SAME fault, permit+presence exist but expired long before
                                 the scored window starts (valid only in the first third
                                 of the run)
  fp_permit_no_fault         -- no fault, permit+presence valid during the scored window
                                 (tests "permit alone isn't risk")
  normal                     -- no fault, no permit (pure negative control)

Run: `python -m eval.permit_fusion_benchmark.generate`
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "simulator" / "tep2py"))
from tep2py import tep2py  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "data"
N_SAMPLES = 120
WINDOW = 30  # scored window = samples[90:120), matching models/gnn/graph_builder.py
MIN_ONSET, MAX_ONSET = 20, 55  # fault is well-established (not just starting) by sample 90
SAMPLE_MINUTES = 3


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    idv: int
    zone: str
    permit_type: str


SCENARIOS = [
    Scenario("s1_reactor_heat_removal", idv=4, zone="reactor_zone", permit_type="confined_space"),
    Scenario("s2_condenser_pressure", idv=5, zone="condenser_zone", permit_type="general"),
    Scenario("s3_deferred_maintenance", idv=13, zone="reactor_zone", permit_type="confined_space"),
    Scenario("s4_feed_system", idv=6, zone="feed_zone", permit_type="hot_work"),
    Scenario("s5_common_cause_utility", idv=11, zone="reactor_zone", permit_type="general"),
]
CONDITIONS = ["true_positive", "fp_fault_no_permit", "fp_fault_permit_no_overlap", "fp_permit_no_fault", "normal"]
HAS_FAULT = {"true_positive": True, "fp_fault_no_permit": True, "fp_fault_permit_no_overlap": True,
             "fp_permit_no_fault": False, "normal": False}


def _sample_to_iso(sample_idx: int, run_date: pd.Timestamp) -> str:
    return (run_date + pd.Timedelta(minutes=SAMPLE_MINUTES * sample_idx)).isoformat()


def _permit_presence_schedule(condition: str, rng: np.random.Generator) -> dict:
    """All timing is relative to the fixed scored window [90, 120)."""
    if condition in ("true_positive", "fp_permit_no_fault"):
        # valid DURING the scored window -- jittered start so it's not a trivially fixed value
        permit_from = int(rng.integers(82, 94))
        permit_to = min(N_SAMPLES - 1, permit_from + int(rng.integers(25, 38)))
        presence_from = int(rng.integers(max(permit_from, 90), 96))
        presence_to = min(N_SAMPLES - 1, presence_from + int(rng.integers(15, 25)))
        return {"has_permit": True, "has_presence": True,
                "permit_from": permit_from, "permit_to": permit_to,
                "presence_from": presence_from, "presence_to": presence_to}
    if condition == "fp_fault_permit_no_overlap":
        # valid ONLY in the first third of the run -- long expired by sample 90
        permit_from = int(rng.integers(5, 15))
        permit_to = permit_from + int(rng.integers(15, 25))
        presence_from = permit_from + int(rng.integers(2, 6))
        presence_to = min(permit_to, presence_from + int(rng.integers(8, 15)))
        return {"has_permit": True, "has_presence": True,
                "permit_from": permit_from, "permit_to": permit_to,
                "presence_from": presence_from, "presence_to": presence_to}
    # fp_fault_no_permit, normal
    return {"has_permit": False, "has_presence": False,
            "permit_from": None, "permit_to": None, "presence_from": None, "presence_to": None}


def generate_run(scenario: Scenario, condition: str, run_index: int, rng: np.random.Generator, base_date: pd.Timestamp):
    onset = int(rng.integers(MIN_ONSET, MAX_ONSET))
    idata = np.zeros((N_SAMPLES, 20))
    has_fault = HAS_FAULT[condition]
    if has_fault:
        idata[onset:, scenario.idv - 1] = 1

    tep = tep2py(idata)
    tep.simulate()
    df = tep.process_data.reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]

    run_id = uuid.uuid4().hex[:12]
    df["sample_idx"] = np.arange(N_SAMPLES)
    df["run_id"] = run_id
    df["scenario_id"] = scenario.scenario_id
    df["condition"] = condition
    df["zone"] = scenario.zone
    df["fault_active"] = has_fault
    df["fault_onset_sample"] = onset if has_fault else None
    df["true_positive"] = condition == "true_positive"

    schedule = _permit_presence_schedule(condition, rng)
    run_date = base_date + pd.Timedelta(hours=run_index)
    permit = {
        "permit_id": f"PTW-{uuid.uuid4().hex[:8]}",
        "permit_type": scenario.permit_type if schedule["has_permit"] else "general",
        "zone": scenario.zone,
        "valid_from": _sample_to_iso(schedule["permit_from"], run_date) if schedule["has_permit"] else None,
        "valid_to": _sample_to_iso(schedule["permit_to"], run_date) if schedule["has_permit"] else None,
        "status": "active" if schedule["has_permit"] else "expired",
        "run_id": run_id,
        "has_permit": schedule["has_permit"],
        "from_sample": schedule["permit_from"],
        "to_sample": schedule["permit_to"],
    }
    presence = {
        "worker_id": f"W-{uuid.uuid4().hex[:6]}",
        "zone": scenario.zone,
        "entry_time": _sample_to_iso(schedule["presence_from"], run_date) if schedule["has_presence"] else None,
        "exit_time": _sample_to_iso(schedule["presence_to"], run_date) if schedule["has_presence"] else None,
        "run_id": run_id,
        "has_presence": schedule["has_presence"],
        "from_sample": schedule["presence_from"],
        "to_sample": schedule["presence_to"],
    }
    return df, permit, presence


def generate_dataset(n_runs_per_condition: int = 40, seed: int = 42):
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2026-01-01")
    manifest_rows, permits, presences = [], [], []
    run_index = 0

    for scenario in SCENARIOS:
        for condition in CONDITIONS:
            out_dir = OUT_DIR / scenario.scenario_id / condition
            out_dir.mkdir(parents=True, exist_ok=True)
            for _ in range(n_runs_per_condition):
                df, permit, presence = generate_run(scenario, condition, run_index, rng, base_date)
                run_index += 1
                run_id = df["run_id"].iloc[0]
                out_path = out_dir / f"{run_id}.parquet"
                df.to_parquet(out_path, index=False)
                permits.append(permit)
                presences.append(presence)
                manifest_rows.append({
                    "run_id": run_id,
                    "scenario_id": scenario.scenario_id,
                    "condition": condition,
                    "zone": scenario.zone,
                    "permit_type": scenario.permit_type,
                    "fault_active": HAS_FAULT[condition],
                    "true_positive": condition == "true_positive",
                    "path": str(out_path.relative_to(REPO_ROOT)),
                })
            print(f"[{scenario.scenario_id}/{condition}] generated {n_runs_per_condition} runs")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
    pd.DataFrame(permits).to_parquet(OUT_DIR / "permits.parquet", index=False)
    pd.DataFrame(presences).to_parquet(OUT_DIR / "presences.parquet", index=False)
    print(f"\nWrote manifest ({len(manifest)} runs) + permits + presences to {OUT_DIR}")
    return manifest


if __name__ == "__main__":
    generate_dataset()
