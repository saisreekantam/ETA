"""
Computes a time-animated replay trace for one run: at each cutoff sample, the GNN's
score for every zone (permit/presence visibility properly gated by their actual
from_sample/to_sample -- see the lead-time methodology note in eval/metrics.py; this
reuses the same fix) and the baseline's trailing-window max z-score per zone. This is
what lets the dashboard animate "watch the model lock onto the true zone over time"
instead of showing a single static snapshot.

Honest framing per eval/metrics.py: the GNN does not detect earlier than the baseline
on this benchmark once gating is correct. What the replay actually shows well is the
model's STABILITY and ZONE-SPECIFICITY over time, not earlier lead time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.gnn.baseline_threshold import THRESHOLD_SIGMA, ZONE_TO_SENSOR_COLS  # noqa: E402
from models.gnn.graph_builder import WINDOW, ZONE_VOCAB, build_graph  # noqa: E402
from models.gnn.model import CompoundRiskGNN  # noqa: E402

_model = None
_norm_stats = None
_baseline_stats = None
SENSOR_COLS = [f"XMEAS({i})" for i in range(1, 42)] + [f"XMV({i})" for i in range(1, 12)]


def _lazy_load():
    global _model, _norm_stats, _baseline_stats
    if _model is None:
        _model = CompoundRiskGNN()
        _model.load_state_dict(torch.load(REPO_ROOT / "models" / "gnn" / "checkpoint.pt"))
        _model.eval()
        _norm_stats = torch.load(REPO_ROOT / "models" / "gnn" / "norm_stats.pt")

        normal_runs = pd.read_csv(REPO_ROOT / "data" / "synthetic" / "manifest.csv")
        normal_runs = normal_runs[normal_runs["condition"] == "normal"]
        frames = [pd.read_parquet(REPO_ROOT / p) for p in normal_runs["path"]]
        all_normal = pd.concat(frames, ignore_index=True)
        _baseline_stats = (all_normal[SENSOR_COLS].mean(), all_normal[SENSOR_COLS].std().replace(0, 1e-6))


def _gated_record(record: dict, cutoff: int) -> dict:
    gated = dict(record)
    from_sample = record.get("from_sample")
    if from_sample is None or pd.isna(from_sample):
        return gated
    gated_key = "has_permit" if "has_permit" in record else "has_presence"
    gated[gated_key] = bool(record[gated_key]) and cutoff >= from_sample
    return gated


def compute_replay_trace(run_id: str, step: int = 2) -> dict:
    _lazy_load()
    manifest = pd.read_csv(REPO_ROOT / "data" / "synthetic" / "manifest.csv").set_index("run_id")
    permits = pd.read_parquet(REPO_ROOT / "data" / "permits" / "permits.parquet").set_index("run_id")
    presences = pd.read_parquet(REPO_ROOT / "data" / "shiftlogs" / "shiftlogs.parquet").set_index("run_id")

    row = manifest.loc[run_id]
    sensor_df = pd.read_parquet(REPO_ROOT / row["path"])
    permit_record = permits.loc[run_id].to_dict()
    presence_record = presences.loc[run_id].to_dict()
    permit_zone = permit_record["zone"]

    cutoffs = list(range(WINDOW + 1, len(sensor_df), step))
    trace = {zone: {"gnn": [], "baseline_alert": []} for zone in ZONE_VOCAB}

    base_mean, base_std = _baseline_stats

    for cutoff in cutoffs:
        gated_permit = pd.Series(_gated_record(permit_record, cutoff))
        gated_presence = pd.Series(_gated_record(presence_record, cutoff))
        window_df = sensor_df.iloc[:cutoff]

        graph = build_graph(window_df, gated_permit, gated_presence, permit_zone)
        for ntype, (mean, std) in _norm_stats.items():
            graph[ntype].x = (graph[ntype].x - mean) / std
        with torch.no_grad():
            batch = next(iter(DataLoader([graph], batch_size=1)))
            probs = torch.sigmoid(_model(batch)).tolist()

        trailing = window_df.iloc[-WINDOW:]
        for i, zone in enumerate(ZONE_VOCAB):
            trace[zone]["gnn"].append(round(probs[i], 4))
            if zone in ZONE_TO_SENSOR_COLS:
                cols = ZONE_TO_SENSOR_COLS[zone]
                z = (trailing[cols] - base_mean[cols]) / base_std[cols]
                trace[zone]["baseline_alert"].append(bool((z.abs() > THRESHOLD_SIGMA).any().any()))
            else:
                trace[zone]["baseline_alert"].append(False)

    return {
        "run_id": run_id,
        "cutoffs": cutoffs,
        "ground_truth_onset_sample": None if pd.isna(row["ground_truth_onset_sample"]) else int(row["ground_truth_onset_sample"]),
        "true_zone": row["zone"],
        "trace": trace,
    }


if __name__ == "__main__":
    import json
    result = compute_replay_trace(sys.argv[1] if len(sys.argv) > 1 else "8e39f3138d6e", step=5)
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
    print("reactor_zone gnn trace:", result["trace"]["reactor_zone"]["gnn"])
