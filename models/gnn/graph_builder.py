"""
Builds one heterogeneous graph per run covering ALL 7 plant zones simultaneously, fusing
sensor telemetry with permit and worker-presence context. This is the multi-zone upgrade
over the original single-zone-per-run design: each sensor cluster now connects only to
its OWN zone (not a shared single zone node), zones are linked by directed process-flow
edges (feed -> reactor -> condenser -> separator -> stripper/compressor recycle), and the
model predicts a compound-risk score PER ZONE per graph -- not one scalar per run. This
is what turns "geospatial evidence quality" into a real, checkable metric (does the
model's top-scoring zone match the zone the fault was actually injected into) instead of
a placeholder, and it's what lets the dashboard heatmap show genuine multi-zone variation.

Node types: sensor_cluster (6, one per instrumented zone), permit (1), presence (1), zone (7).
Edges: sensor_cluster<->zone ("monitors", 1:1), permit<->zone ("authorizes"),
presence<->zone ("occupies"), zone<->zone ("flows_to", process adjacency).
control_room has no sensor cluster (no physical hazard, per docs/scenario-definitions.md)
and no process-flow edges -- it has no message-passing input, so its score is always ~0
by construction, same as a real control room with no hazardous process inside it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"

WINDOW = 30  # last 30 samples (90 min) of the 120-sample run used as the "current" observation

# one sensor cluster per instrumented zone (1:1, not shared) -- control_room has none
CLUSTER_TO_ZONE = {
    "reactor": "reactor_zone",
    "condenser": "condenser_zone",
    "separator": "separator_zone",
    "feed": "feed_zone",
    "stripper": "stripper_zone",
    "compressor": "compressor_zone",
}
SENSOR_CLUSTERS = {
    "reactor": ["XMEAS(7)", "XMEAS(8)", "XMEAS(9)", "XMEAS(21)", "XMV(10)"],
    "condenser": ["XMEAS(22)", "XMV(11)"],
    "separator": ["XMEAS(11)", "XMEAS(12)", "XMEAS(13)", "XMEAS(14)"],
    "feed": ["XMEAS(1)", "XMEAS(2)", "XMEAS(3)", "XMEAS(4)", "XMEAS(6)",
             "XMV(1)", "XMV(2)", "XMV(3)", "XMV(4)"],
    "stripper": ["XMEAS(15)", "XMEAS(16)", "XMEAS(17)", "XMEAS(18)", "XMEAS(19)", "XMV(8)", "XMV(9)"],
    "compressor": ["XMEAS(5)", "XMEAS(10)", "XMEAS(20)", "XMV(5)", "XMV(6)", "XMV(7)"],
}
CLUSTER_NAMES = list(SENSOR_CLUSTERS.keys())

ZONE_VOCAB = ["reactor_zone", "condenser_zone", "separator_zone", "stripper_zone",
              "compressor_zone", "feed_zone", "control_room"]
PERMIT_TYPE_VOCAB = ["hot_work", "confined_space", "electrical", "general"]

# directed process-flow adjacency (physical topology, lets risk propagate across zones)
ZONE_FLOW_EDGES = [
    ("feed_zone", "reactor_zone"),
    ("reactor_zone", "condenser_zone"),
    ("condenser_zone", "separator_zone"),
    ("separator_zone", "stripper_zone"),
    ("separator_zone", "compressor_zone"),
    ("compressor_zone", "reactor_zone"),  # recycle loop
]


def _zone_onehot(zone: str) -> list[float]:
    return [1.0 if zone == z else 0.0 for z in ZONE_VOCAB]


def _permit_type_onehot(permit_type: str) -> list[float]:
    return [1.0 if permit_type == p else 0.0 for p in PERMIT_TYPE_VOCAB]


def build_graph(sensor_df: pd.DataFrame, permit: pd.Series, presence: pd.Series, permit_zone: str) -> HeteroData:
    """permit_zone is the zone the run's permit/worker-presence record applies to (the
    scenario's designated hazard zone) -- sensor data covers all zones regardless."""
    window_df = sensor_df.iloc[-WINDOW:]

    sensor_feats = []
    for cluster in CLUSTER_NAMES:
        cols = SENSOR_CLUSTERS[cluster]
        vals = window_df[cols].to_numpy()
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        slope = vals[-1] - vals[0]
        feat = np.concatenate([mean, std, slope])
        sensor_feats.append(feat)
    max_len = max(len(f) for f in sensor_feats)
    sensor_feats = np.stack([np.pad(f, (0, max_len - len(f))) for f in sensor_feats])

    permit_feat = np.array(
        [float(bool(permit["has_permit"]))] + _permit_type_onehot(
            permit["permit_type"] if permit["has_permit"] else "none"
        ),
        dtype=np.float32,
    )
    presence_feat = np.array([float(bool(presence["has_presence"]))], dtype=np.float32)
    zone_feats = np.array([_zone_onehot(z) for z in ZONE_VOCAB], dtype=np.float32)

    data = HeteroData()
    data["sensor_cluster"].x = torch.tensor(sensor_feats, dtype=torch.float32)
    data["permit"].x = torch.tensor(permit_feat, dtype=torch.float32).unsqueeze(0)
    data["presence"].x = torch.tensor(presence_feat, dtype=torch.float32).unsqueeze(0)
    data["zone"].x = torch.tensor(zone_feats, dtype=torch.float32)

    # sensor_cluster <-> zone, strictly 1:1 (cluster i monitors only its own zone)
    src = list(range(len(CLUSTER_NAMES)))
    dst = [ZONE_VOCAB.index(CLUSTER_TO_ZONE[c]) for c in CLUSTER_NAMES]
    data["sensor_cluster", "monitors", "zone"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data["zone", "monitored_by", "sensor_cluster"].edge_index = torch.tensor([dst, src], dtype=torch.long)

    permit_zone_idx = ZONE_VOCAB.index(permit_zone)
    data["permit", "authorizes", "zone"].edge_index = torch.tensor([[0], [permit_zone_idx]], dtype=torch.long)
    data["zone", "authorized_by", "permit"].edge_index = torch.tensor([[permit_zone_idx], [0]], dtype=torch.long)
    data["presence", "occupies", "zone"].edge_index = torch.tensor([[0], [permit_zone_idx]], dtype=torch.long)
    data["zone", "occupied_by", "presence"].edge_index = torch.tensor([[permit_zone_idx], [0]], dtype=torch.long)

    flow_src = [ZONE_VOCAB.index(a) for a, b in ZONE_FLOW_EDGES]
    flow_dst = [ZONE_VOCAB.index(b) for a, b in ZONE_FLOW_EDGES]
    data["zone", "flows_to", "zone"].edge_index = torch.tensor([flow_src, flow_dst], dtype=torch.long)
    data["zone", "flows_from", "zone"].edge_index = torch.tensor([flow_dst, flow_src], dtype=torch.long)

    return data


def zone_label_vector(zone: str, compound_active: bool) -> list[int]:
    """Length-7 vector, 1.0 only at the injected zone's index when compound_active."""
    return [1 if (compound_active and z == zone) else 0 for z in ZONE_VOCAB]


def load_all_graphs() -> tuple[list[HeteroData], list[list[int]], pd.DataFrame]:
    """Returns (graphs, per-zone label vectors, manifest) for every run in the manifest."""
    manifest = pd.read_csv(SYNTHETIC_DIR / "manifest.csv")
    permits = pd.read_parquet(REPO_ROOT / "data" / "permits" / "permits.parquet").set_index("run_id")
    presences = pd.read_parquet(REPO_ROOT / "data" / "shiftlogs" / "shiftlogs.parquet").set_index("run_id")

    graphs, labels = [], []
    for _, row in manifest.iterrows():
        sensor_df = pd.read_parquet(REPO_ROOT / row["path"])
        permit = permits.loc[row["run_id"]]
        presence = presences.loc[row["run_id"]]
        graph = build_graph(sensor_df, permit, presence, row["zone"])
        graphs.append(graph)
        labels.append(zone_label_vector(row["zone"], bool(row["compound_active"])))

    return graphs, labels, manifest


if __name__ == "__main__":
    graphs, labels, manifest = load_all_graphs()
    n_pos = sum(any(lbl) for lbl in labels)
    print(f"Built {len(graphs)} graphs, {n_pos} with a positive zone label")
    print(graphs[0])
    print("labels[0]:", labels[0], "labels[compound idx]:", labels[manifest.index[manifest.condition == 'compound'][0]])
