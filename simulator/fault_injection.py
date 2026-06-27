"""
Generates the ground-truth-labeled compound-risk benchmark on top of the TEP simulator.

For each scenario in SCENARIOS, generates runs across five conditions:
  - "normal":      no faults active (negative control)
  - "single_a":    only the scenario's first IDV active (trains/evals the naive baseline)
  - "single_b":    only the scenario's second IDV active
  - "compound":    both IDVs active together (the actual ground-truth positive)

Every run is saved as one parquet file under data/synthetic/<scenario_id>/<condition>/
plus a row in the manifest (data/synthetic/manifest.csv) used for run-level train/val/test
splitting (never split by timestep — see docs/scenario-definitions.md).

Run directly: `python simulator/fault_injection.py`
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "simulator" / "tep2py"))
from tep2py import tep2py  # noqa: E402

SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"

N_SAMPLES = 120          # 120 samples * 3 min = 360 min (6h) per run
MIN_ONSET_SAMPLE = 30    # leave a clean "normal" lead-in before fault onset
MAX_ONSET_SAMPLE = 70    # leave enough runway after onset to observe steady deviation
MAX_IDV_OFFSET = 4       # samples of jitter between fault A and fault B onset (realism)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    name: str
    idv_a: int
    idv_b: int
    zone: str
    permit_type: str


SCENARIOS: list[Scenario] = [
    Scenario("s1_reactor_heat_removal", "Reactor Heat-Removal Compound Failure",
             idv_a=4, idv_b=14, zone="reactor_zone", permit_type="confined_space"),
    Scenario("s2_condenser_pressure", "Condenser/Separator Pressure Compound Failure",
             idv_a=5, idv_b=15, zone="condenser_zone", permit_type="general"),
    Scenario("s3_deferred_maintenance", "Deferred-Maintenance + Utility-Fault Compound Risk",
             idv_a=13, idv_b=4, zone="reactor_zone", permit_type="confined_space"),
    Scenario("s4_feed_system", "Feed System Compound Fault",
             idv_a=6, idv_b=1, zone="feed_zone", permit_type="hot_work"),
    Scenario("s5_common_cause_utility", "Common-Cause Utility Degradation",
             idv_a=11, idv_b=12, zone="reactor_zone", permit_type="general"),
]

CONDITIONS = ["normal", "single_a", "single_b", "compound"]


def _build_idata(scenario: Scenario, condition: str, onset_a: int, onset_b: int) -> np.ndarray:
    idata = np.zeros((N_SAMPLES, 20))
    if condition in ("single_a", "compound"):
        idata[onset_a:, scenario.idv_a - 1] = 1
    if condition in ("single_b", "compound"):
        idata[onset_b:, scenario.idv_b - 1] = 1
    return idata


def generate_run(scenario: Scenario, condition: str, rng: np.random.Generator) -> pd.DataFrame:
    onset_a = int(rng.integers(MIN_ONSET_SAMPLE, MAX_ONSET_SAMPLE))
    offset = int(rng.integers(0, MAX_IDV_OFFSET + 1))
    onset_b = onset_a + offset

    idata = _build_idata(scenario, condition, onset_a, onset_b)
    tep = tep2py(idata)
    tep.simulate()
    df = tep.process_data.reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]

    run_id = uuid.uuid4().hex[:12]
    ground_truth_onset_sample = max(onset_a, onset_b) if condition == "compound" else None

    df["sample_idx"] = np.arange(N_SAMPLES)
    df["run_id"] = run_id
    df["scenario_id"] = scenario.scenario_id
    df["condition"] = condition
    df["zone"] = scenario.zone
    df["fault_a_active"] = (df["sample_idx"] >= onset_a) & (condition in ("single_a", "compound"))
    df["fault_b_active"] = (df["sample_idx"] >= onset_b) & (condition in ("single_b", "compound"))
    df["compound_active"] = condition == "compound"
    df["ground_truth_onset_sample"] = ground_truth_onset_sample

    return df


def generate_dataset(n_runs_per_condition: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    manifest_rows = []

    for scenario in SCENARIOS:
        for condition in CONDITIONS:
            out_dir = SYNTHETIC_DIR / scenario.scenario_id / condition
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_runs_per_condition):
                df = generate_run(scenario, condition, rng)
                run_id = df["run_id"].iloc[0]
                out_path = out_dir / f"{run_id}.parquet"
                df.to_parquet(out_path, index=False)
                manifest_rows.append({
                    "run_id": run_id,
                    "scenario_id": scenario.scenario_id,
                    "condition": condition,
                    "zone": scenario.zone,
                    "permit_type": scenario.permit_type,
                    "compound_active": condition == "compound",
                    "ground_truth_onset_sample": df["ground_truth_onset_sample"].iloc[0],
                    "path": str(out_path.relative_to(REPO_ROOT)),
                })
            print(f"[{scenario.scenario_id}/{condition}] generated {n_runs_per_condition} runs")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SYNTHETIC_DIR / "manifest.csv", index=False)
    print(f"\nWrote manifest with {len(manifest)} runs to {SYNTHETIC_DIR / 'manifest.csv'}")
    return manifest


if __name__ == "__main__":
    generate_dataset()
