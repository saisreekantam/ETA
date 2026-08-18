"""
Real-world validation of the compound-risk architecture (GATv2 + temporal GRU over a
heterogeneous typed graph) on the HAI 22.04 dataset -- a real hardware-in-the-loop ICS
testbed (boiler + turbine + water-treatment + auxiliary loop), NOT a simulation like TEP.

This is a deliberately narrower check than the production pipeline: HAI has no permit,
shift-log or worker-presence data (no public dataset combining sensors with human/permit
activity exists -- same gap documented for TEP in docs/main.pdf Sec 3.1), and it only
publishes ONE global Attack label per timestamp, not a per-subsystem one. So this module
drops the permit/presence/worker node types entirely and predicts a single pooled
graph-level score instead of one score per zone. What survives unchanged is the
architectural bet under test: does a heterogeneous typed graph with a GRU-encoded
temporal cluster embedding and 2-hop GATv2 message passing generalize to a real testbed's
real induced-fault dynamics, against a matched per-subsystem z-score threshold baseline.

Four subsystems in the raw CSV columns (by prefix), used as both "cluster" and "zone":
  P1 -- boiler process (44 channels)
  P2 -- turbine process (24 channels)
  P3 -- water-treatment process (7 channels)
  P4 -- HIL-simulated auxiliary/hydro loop (11 channels)

Zone flow edges approximate the documented physical topology (boiler produces steam that
drives the turbine; water treatment supplies boiler feedwater; the auxiliary loop is
coupled to the turbine) -- this is a reasonable reading of the public HAI technical
documentation, not a verified P&ID, and is noted as such.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

REPO_ROOT = Path(__file__).resolve().parents[2]
HAI_DIR = Path(
    "/private/tmp/claude-501/-Users-sreekantamsaivenkat-Desktop-ETA-de6e49cd-c958-4a48-81b4-d4c20504d739"
    "/scratchpad/hai"
)

WINDOW = 60   # 60 samples = 60s of context at HAI's 1Hz sampling rate
STRIDE = 10   # window stride when enumerating training/eval windows

ZONE_VOCAB = ["P1", "P2", "P3", "P4"]  # boiler, turbine, water-treatment, auxiliary loop
ZONE_FLOW_EDGES = [
    ("P3", "P1"),  # water treatment -> boiler feedwater
    ("P1", "P2"),  # boiler steam -> turbine
    ("P2", "P1"),  # condensate return / recycle
    ("P2", "P4"),  # turbine <-> auxiliary/hydro loop coupling
]

_SUBSYSTEM_COLS: dict[str, list[str]] | None = None


def subsystem_columns() -> dict[str, list[str]]:
    """Lazy: reads one raw CSV's header to group columns by P1/P2/P3/P4 prefix."""
    global _SUBSYSTEM_COLS
    if _SUBSYSTEM_COLS is None:
        header = pd.read_csv(HAI_DIR / "train1.csv", nrows=0).columns.tolist()
        cols = {z: [] for z in ZONE_VOCAB}
        for c in header:
            if c in ("timestamp", "Attack"):
                continue
            prefix = c.split("_")[0]
            if prefix in cols:
                cols[prefix].append(c)
        _SUBSYSTEM_COLS = cols
    return _SUBSYSTEM_COLS


def max_cluster_sensors() -> int:
    return max(len(v) for v in subsystem_columns().values())


def _zone_onehot(zone: str) -> list[float]:
    return [1.0 if zone == z else 0.0 for z in ZONE_VOCAB]


def build_graph(window_df: pd.DataFrame) -> HeteroData:
    """window_df: WINDOW consecutive rows of raw HAI sensor columns (no Attack/timestamp)."""
    cols = subsystem_columns()
    max_sensors = max_cluster_sensors()

    cluster_seqs = []
    sensor_feats = []
    sensor_cols_flat: list[tuple[str, str]] = []
    for zone in ZONE_VOCAB:
        vals = window_df[cols[zone]].to_numpy(dtype=np.float32)
        pad = max_sensors - vals.shape[1]
        cluster_seqs.append(np.pad(vals, ((0, 0), (0, pad))))
        for c in cols[zone]:
            v = window_df[c].to_numpy(dtype=np.float32)
            sensor_feats.append([v.mean(), v.std(), v[-1] - v[0], v.min(), v.max()])
            sensor_cols_flat.append((zone, c))
    cluster_seqs = np.stack(cluster_seqs)  # [4, WINDOW, max_sensors]
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
