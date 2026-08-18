"""
Same CompoundRiskGNN architecture and node/edge schema as models/gnn/graph_builder.py --
reused UNCHANGED (models.gnn.model.CompoundRiskGNN) -- but with one deliberate feature
semantics fix that this benchmark's design REQUIRES: has_permit/has_presence now mean
"currently valid during the scored window" (computed from valid_from/to vs the window's
sample range) instead of "a permit record exists somewhere in the run". The original
build_graph never computed this -- has_permit was a static whole-run boolean, which is
exactly why the original benchmark couldn't have tested permit-timing fusion even if its
labels had been designed to require it. Every feature vector SHAPE is identical to the
original (permit=5, presence=1, worker=2), so CompoundRiskGNN's weights/IN_DIMS need no
change at all -- only what the numbers mean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from models.gnn.graph_builder import (
    CLUSTER_NAMES, CLUSTER_TO_ZONE, PERMIT_TYPE_VOCAB, SENSOR_CLUSTERS, SENSOR_NODE_COLS,
    WINDOW, ZONE_FLOW_EDGES, ZONE_VOCAB, MAX_CLUSTER_SENSORS, _zone_onehot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
N_SAMPLES = 120


def _permit_type_onehot(permit_type: str) -> list[float]:
    return [1.0 if permit_type == p else 0.0 for p in PERMIT_TYPE_VOCAB]


def _overlaps_window(from_sample, to_sample, window_start: int, window_end: int) -> bool:
    if from_sample is None or to_sample is None or pd.isna(from_sample) or pd.isna(to_sample):
        return False
    return float(from_sample) <= window_end and float(to_sample) >= window_start


def build_graph(sensor_df: pd.DataFrame, permit: pd.Series, presence: pd.Series, permit_zone: str) -> HeteroData:
    n_rows = len(sensor_df)
    window_start, window_end = n_rows - WINDOW, n_rows - 1
    window_df = sensor_df.iloc[-WINDOW:]

    cluster_seqs = []
    for cluster in CLUSTER_NAMES:
        vals = window_df[SENSOR_CLUSTERS[cluster]].to_numpy(dtype=np.float32)
        if len(vals) < WINDOW:
            vals = np.concatenate([np.repeat(vals[:1], WINDOW - len(vals), axis=0), vals])
        pad = MAX_CLUSTER_SENSORS - vals.shape[1]
        cluster_seqs.append(np.pad(vals, ((0, 0), (0, pad))))
    cluster_seqs = np.stack(cluster_seqs)

    sensor_node_feats = []
    for _, col in SENSOR_NODE_COLS:
        v = window_df[col].to_numpy(dtype=np.float32)
        sensor_node_feats.append([v.mean(), v.std(), v[-1] - v[0], v.min(), v.max()])
    sensor_node_feats = np.array(sensor_node_feats, dtype=np.float32)

    # THE key change: "active" = record exists AND its validity window overlaps the
    # scored window -- not "a record exists somewhere in the 120-sample run".
    permit_active = bool(permit["has_permit"]) and _overlaps_window(
        permit.get("from_sample"), permit.get("to_sample"), window_start, window_end)
    presence_active = bool(presence["has_presence"]) and _overlaps_window(
        presence.get("from_sample"), presence.get("to_sample"), window_start, window_end)

    permit_feat = np.array(
        [float(permit_active)] + _permit_type_onehot(permit["permit_type"] if permit_active else "none"),
        dtype=np.float32,
    )
    presence_feat = np.array([float(presence_active)], dtype=np.float32)

    # dwell_frac: fraction of the SCORED WINDOW (not the whole run) actually covered by
    # presence -- consistent with "active" now meaning window-relative, not run-relative
    dwell_frac = 0.0
    if presence_active:
        p_from, p_to = float(presence["from_sample"]), float(presence["to_sample"])
        overlap = min(p_to, window_end) - max(p_from, window_start) + 1
        dwell_frac = max(0.0, overlap) / WINDOW
    worker_feat = np.array([float(presence_active), dwell_frac], dtype=np.float32)

    zone_feats = np.array([_zone_onehot(z) for z in ZONE_VOCAB], dtype=np.float32)

    data = HeteroData()
    data["sensor_cluster"].x = torch.tensor(cluster_seqs, dtype=torch.float32)
    data["sensor"].x = torch.tensor(sensor_node_feats, dtype=torch.float32)
    data["permit"].x = torch.tensor(permit_feat, dtype=torch.float32).unsqueeze(0)
    data["presence"].x = torch.tensor(presence_feat, dtype=torch.float32).unsqueeze(0)
    data["worker"].x = torch.tensor(worker_feat, dtype=torch.float32).unsqueeze(0)
    data["plant"].x = torch.ones((1, 1), dtype=torch.float32)
    data["zone"].x = torch.tensor(zone_feats, dtype=torch.float32)

    s_src = list(range(len(SENSOR_NODE_COLS)))
    s_dst = [CLUSTER_NAMES.index(cluster) for cluster, _ in SENSOR_NODE_COLS]
    data["sensor", "feeds", "sensor_cluster"].edge_index = torch.tensor([s_src, s_dst], dtype=torch.long)
    data["sensor_cluster", "aggregates", "sensor"].edge_index = torch.tensor([s_dst, s_src], dtype=torch.long)

    src = list(range(len(CLUSTER_NAMES)))
    dst = [ZONE_VOCAB.index(CLUSTER_TO_ZONE[c]) for c in CLUSTER_NAMES]
    data["sensor_cluster", "monitors", "zone"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data["zone", "monitored_by", "sensor_cluster"].edge_index = torch.tensor([dst, src], dtype=torch.long)

    permit_zone_idx = ZONE_VOCAB.index(permit_zone)
    data["permit", "authorizes", "zone"].edge_index = torch.tensor([[0], [permit_zone_idx]], dtype=torch.long)
    data["zone", "authorized_by", "permit"].edge_index = torch.tensor([[permit_zone_idx], [0]], dtype=torch.long)
    data["presence", "occupies", "zone"].edge_index = torch.tensor([[0], [permit_zone_idx]], dtype=torch.long)
    data["zone", "occupied_by", "presence"].edge_index = torch.tensor([[permit_zone_idx], [0]], dtype=torch.long)
    data["worker", "works_in", "zone"].edge_index = torch.tensor([[0], [permit_zone_idx]], dtype=torch.long)
    data["zone", "staffed_by", "worker"].edge_index = torch.tensor([[permit_zone_idx], [0]], dtype=torch.long)

    all_zones = list(range(len(ZONE_VOCAB)))
    data["zone", "reports_to", "plant"].edge_index = torch.tensor([all_zones, [0] * len(all_zones)], dtype=torch.long)
    data["plant", "oversees", "zone"].edge_index = torch.tensor([[0] * len(all_zones), all_zones], dtype=torch.long)

    flow_src = [ZONE_VOCAB.index(a) for a, b in ZONE_FLOW_EDGES]
    flow_dst = [ZONE_VOCAB.index(b) for a, b in ZONE_FLOW_EDGES]
    data["zone", "flows_to", "zone"].edge_index = torch.tensor([flow_src, flow_dst], dtype=torch.long)
    data["zone", "flows_from", "zone"].edge_index = torch.tensor([flow_dst, flow_src], dtype=torch.long)

    return data


def zone_label_vector(zone: str, true_positive: bool) -> list[int]:
    return [1 if (true_positive and z == zone) else 0 for z in ZONE_VOCAB]


def load_all_graphs():
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    permits = pd.read_parquet(DATA_DIR / "permits.parquet").set_index("run_id")
    presences = pd.read_parquet(DATA_DIR / "presences.parquet").set_index("run_id")

    graphs, labels = [], []
    for _, row in manifest.iterrows():
        sensor_df = pd.read_parquet(REPO_ROOT / row["path"])
        permit = permits.loc[row["run_id"]]
        presence = presences.loc[row["run_id"]]
        graph = build_graph(sensor_df, permit, presence, row["zone"])
        graphs.append(graph)
        labels.append(zone_label_vector(row["zone"], bool(row["true_positive"])))

    return graphs, labels, manifest
