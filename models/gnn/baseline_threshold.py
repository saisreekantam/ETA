"""
Naive single-sensor threshold baseline, now zone-aware (one threshold check per zone's
own sensor cluster) so it's a fair apples-to-apples comparison against the multi-zone
GNN -- still deliberately context-free: no permit, no presence, no cross-zone signal,
just "did any sensor belonging to this zone cross a z-score threshold." That's the
strawman the problem statement names ("single-sensor baselines"). Fits per-sensor
mean/std from "normal" condition runs, then flags each zone as alerted at the first
timestep where any of ITS OWN sensors' z-score exceeds THRESHOLD_SIGMA for
MIN_CONSECUTIVE consecutive samples.

Run directly: `python -m models.gnn.baseline_threshold` (requires data/synthetic/manifest.csv).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.gnn.graph_builder import CLUSTER_TO_ZONE, SENSOR_CLUSTERS, ZONE_VOCAB

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"

THRESHOLD_SIGMA = 4.0
MIN_CONSECUTIVE = 2

SENSOR_COLS = [f"XMEAS({i})" for i in range(1, 42)] + [f"XMV({i})" for i in range(1, 12)]
ZONE_TO_SENSOR_COLS = {zone: SENSOR_CLUSTERS[cluster] for cluster, zone in CLUSTER_TO_ZONE.items()}


def fit_normal_stats(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor mean/std computed from 'normal' condition runs only."""
    normal_runs = manifest[manifest["condition"] == "normal"]
    frames = [pd.read_parquet(REPO_ROOT / p) for p in normal_runs["path"]]
    all_normal = pd.concat(frames, ignore_index=True)
    stats = pd.DataFrame({
        "mean": all_normal[SENSOR_COLS].mean(),
        "std": all_normal[SENSOR_COLS].std().replace(0, 1e-6),
    })
    return stats


def _first_alert_sample(exceeds: np.ndarray) -> int | None:
    run_len = 0
    for idx, flag in enumerate(exceeds):
        run_len = run_len + 1 if flag else 0
        if run_len >= MIN_CONSECUTIVE:
            return idx - MIN_CONSECUTIVE + 1
    return None


def score_run(df: pd.DataFrame, stats: pd.DataFrame) -> dict:
    z = (df[SENSOR_COLS] - stats["mean"]) / stats["std"]
    result = {"run_id": df["run_id"].iloc[0]}

    zone_max_z = {}
    for zone, cols in ZONE_TO_SENSOR_COLS.items():
        zone_z = z[cols].abs()
        exceeds = (zone_z > THRESHOLD_SIGMA).any(axis=1).to_numpy()
        alert_sample = _first_alert_sample(exceeds)
        result[f"did_alert__{zone}"] = alert_sample is not None
        result[f"alert_sample__{zone}"] = alert_sample
        result[f"max_abs_zscore__{zone}"] = float(zone_z.to_numpy().max())
        zone_max_z[zone] = result[f"max_abs_zscore__{zone}"]

    # control_room has no sensors -- always zero, same convention as the GNN
    result["did_alert__control_room"] = False
    result["alert_sample__control_room"] = None
    result["max_abs_zscore__control_room"] = 0.0

    result["did_alert"] = any(result[f"did_alert__{z}"] for z in ZONE_TO_SENSOR_COLS)
    result["top_zone"] = max(zone_max_z, key=zone_max_z.get)
    result["max_abs_zscore"] = max(zone_max_z.values())
    return result


def run_baseline(manifest_path: Path | None = None) -> pd.DataFrame:
    manifest_path = manifest_path or SYNTHETIC_DIR / "manifest.csv"
    manifest = pd.read_csv(manifest_path)

    stats = fit_normal_stats(manifest)

    results = []
    for _, row in manifest.iterrows():
        df = pd.read_parquet(REPO_ROOT / row["path"])
        result = score_run(df, stats)
        result["scenario_id"] = row["scenario_id"]
        result["condition"] = row["condition"]
        result["zone"] = row["zone"]
        result["compound_active"] = row["compound_active"]
        result["ground_truth_onset_sample"] = row["ground_truth_onset_sample"]
        results.append(result)

    out = pd.DataFrame(results)
    out_path = SYNTHETIC_DIR.parent / "baseline_predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} baseline predictions to {out_path}")
    return out


if __name__ == "__main__":
    preds = run_baseline()
    compound = preds[preds["compound_active"]]
    print(f"\nBaseline alert rate on compound runs (any zone): {compound['did_alert'].mean():.2%} "
          f"({compound['did_alert'].sum()}/{len(compound)})")
    print(f"Baseline top-zone localization accuracy: {(compound['top_zone'] == compound['zone']).mean():.2%}")
