"""
HAI 23.05 data plumbing -- a second, independent real ICS testbed dataset (same physical
boiler+turbine+water-treatment+auxiliary rig as 22.04, later collection run with a harder,
more boiler-concentrated attack campaign including 8 new "internal-point" control-logic
attacks 22.04 didn't have). Kept as its own self-contained module tree rather than
parametrizing eval/hai_validation/graph_builder.py etc., so the already-reported 22.04
results stay reproducible untouched.

Two format differences from 22.04 that this module absorbs:
  1. The attack label lives in a SEPARATE label-testN.csv file (timestamp,label), not an
     embedded Attack column -- _load_file joins them positionally (both files are
     confirmed same length, same start, strictly row-aligned; label-testN.csv's own
     timestamp column is truncated to the minute in the raw release, so positional join
     is the only reliable option, not a timestamp merge).
  2. 23.05 adds 7 new "x1001_*/x1002_*/x1003_*" columns -- internal DCS register outputs
     that the new AE01-AE08 attacks target. All of AE01-AE08's Target Controller values
     (P1-PC/P1-LC/P1-TC/P1-CC/P1-HC) are P1 boiler controllers (attack_labels_2305.py),
     so these columns are routed into the P1 zone alongside the original P1_* tags.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

HAI_DIR = Path(
    "/private/tmp/claude-501/-Users-sreekantamsaivenkat-Desktop-ETA-de6e49cd-c958-4a48-81b4-d4c20504d739"
    "/scratchpad/hai"
)

WINDOW = 60
STRIDE = 10

ZONE_VOCAB = ["P1", "P2", "P3", "P4"]
ZONE_FLOW_EDGES = [("P3", "P1"), ("P1", "P2"), ("P2", "P1"), ("P2", "P4")]

TRAIN_FILES = ["hai23-train1.csv", "hai23-test1.csv"]  # train1 = normal-only, test1 = 14 attacks (all P1)
VAL_FILES = ["hai23-test2a.csv"]   # Aug 17-18, 28 attacks
TEST_FILES = ["hai23-test2b.csv"]  # Aug 19, 10 attacks

_SUBSYSTEM_COLS: dict[str, list[str]] | None = None


def subsystem_columns() -> dict[str, list[str]]:
    global _SUBSYSTEM_COLS
    if _SUBSYSTEM_COLS is None:
        header = pd.read_csv(HAI_DIR / "hai23-train1.csv", nrows=0).columns.tolist()
        cols = {z: [] for z in ZONE_VOCAB}
        for c in header:
            if c == "timestamp":
                continue
            prefix = c.split("_")[0]
            if prefix in cols:
                cols[prefix].append(c)
            elif prefix.startswith("x100"):
                cols["P1"].append(c)  # internal boiler DCS registers -- see module docstring
        _SUBSYSTEM_COLS = cols
    return _SUBSYSTEM_COLS


def max_cluster_sensors() -> int:
    return max(len(v) for v in subsystem_columns().values())


def _label_filename(data_filename: str) -> str:
    return data_filename.replace("hai23-test", "hai23-label-test")


def load_file(name: str) -> pd.DataFrame:
    """Returns a df with timestamp + all sensor columns + Attack (0 for train*.csv,
    joined from the matching label file for test*.csv)."""
    df = pd.read_csv(HAI_DIR / name)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if name.startswith("hai23-train"):
        df["Attack"] = 0
    else:
        lab = pd.read_csv(HAI_DIR / _label_filename(name))
        assert len(lab) == len(df), f"{name}: data/label row count mismatch"
        df["Attack"] = lab["label"].to_numpy()
    return df


def _zone_onehot(zone: str) -> list[float]:
    return [1.0 if zone == z else 0.0 for z in ZONE_VOCAB]


def build_graph(window_df: pd.DataFrame) -> HeteroData:
    cols = subsystem_columns()
    max_sensors = max_cluster_sensors()

    cluster_seqs, sensor_feats, sensor_cols_flat = [], [], []
    for zone in ZONE_VOCAB:
        vals = window_df[cols[zone]].to_numpy(dtype=np.float32)
        pad = max_sensors - vals.shape[1]
        cluster_seqs.append(np.pad(vals, ((0, 0), (0, pad))))
        for c in cols[zone]:
            v = window_df[c].to_numpy(dtype=np.float32)
            sensor_feats.append([v.mean(), v.std(), v[-1] - v[0], v.min(), v.max()])
            sensor_cols_flat.append((zone, c))
    cluster_seqs = np.stack(cluster_seqs)
    sensor_feats = np.array(sensor_feats, dtype=np.float32)
    zone_feats = np.array([_zone_onehot(z) for z in ZONE_VOCAB], dtype=np.float32)

    data = HeteroData()
    data["sensor_cluster"].x = torch.tensor(cluster_seqs, dtype=torch.float32)
    data["sensor"].x = torch.tensor(sensor_feats, dtype=torch.float32)
    data["zone"].x = torch.tensor(zone_feats, dtype=torch.float32)
    data["plant"].x = torch.ones((1, 1), dtype=torch.float32)

    s_src = list(range(len(sensor_cols_flat)))
    s_dst = [ZONE_VOCAB.index(z) for z, _ in sensor_cols_flat]
    data["sensor", "feeds", "sensor_cluster"].edge_index = torch.tensor([s_src, s_dst], dtype=torch.long)
    data["sensor_cluster", "aggregates", "sensor"].edge_index = torch.tensor([s_dst, s_src], dtype=torch.long)

    cluster_idx = list(range(len(ZONE_VOCAB)))
    data["sensor_cluster", "monitors", "zone"].edge_index = torch.tensor([cluster_idx, cluster_idx], dtype=torch.long)
    data["zone", "monitored_by", "sensor_cluster"].edge_index = torch.tensor([cluster_idx, cluster_idx], dtype=torch.long)

    all_zones = list(range(len(ZONE_VOCAB)))
    data["zone", "reports_to", "plant"].edge_index = torch.tensor([all_zones, [0] * len(all_zones)], dtype=torch.long)
    data["plant", "oversees", "zone"].edge_index = torch.tensor([[0] * len(all_zones), all_zones], dtype=torch.long)

    flow_src = [ZONE_VOCAB.index(a) for a, b in ZONE_FLOW_EDGES]
    flow_dst = [ZONE_VOCAB.index(b) for a, b in ZONE_FLOW_EDGES]
    data["zone", "flows_to", "zone"].edge_index = torch.tensor([flow_src, flow_dst], dtype=torch.long)
    data["zone", "flows_from", "zone"].edge_index = torch.tensor([flow_dst, flow_src], dtype=torch.long)

    return data
