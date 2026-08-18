"""
Same architecture as models/gnn/model.py's CompoundRiskGNN (GRU temporal encoder over
each subsystem's raw window -> 2 rounds of residual HeteroConv/GATv2Conv message passing)
minus the permit/presence/worker node types HAI doesn't have. Predicts one logit PER ZONE
(P1/P2/P3/P4), matching CompoundRiskGNN's per-zone design -- attack_labels.py recovers
real per-subsystem ground truth from HAI's technical documentation (not published in the
raw CSVs, which only carry one pooled Attack column), so, unlike the first pass at this
validation, there's real per-zone supervision to predict against here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv

from eval.hai_validation.graph_builder import ZONE_VOCAB, max_cluster_sensors

NODE_TYPES = ["sensor_cluster", "sensor", "zone", "plant"]
EDGE_TYPES = [
    ("sensor", "feeds", "sensor_cluster"),
    ("sensor_cluster", "aggregates", "sensor"),
    ("sensor_cluster", "monitors", "zone"),
    ("zone", "monitored_by", "sensor_cluster"),
    ("zone", "reports_to", "plant"),
    ("plant", "oversees", "zone"),
    ("zone", "flows_to", "zone"),
    ("zone", "flows_from", "zone"),
]

IN_DIMS = {"sensor": 5, "zone": len(ZONE_VOCAB), "plant": 1}
HIDDEN = 32


class HaiRiskGNN(nn.Module):
    def __init__(self, hidden: int = HIDDEN, heads: int = 2):
        super().__init__()
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(IN_DIMS[ntype], hidden) for ntype in IN_DIMS
        })
        self.temporal_encoder = nn.GRU(max_cluster_sensors(), hidden, batch_first=True)

        def make_conv():
            return HeteroConv({
                etype: GATv2Conv(hidden, hidden // heads, heads=heads, add_self_loops=False)
                for etype in EDGE_TYPES
            }, aggr="sum")

        self.conv1 = make_conv()
        self.conv2 = make_conv()
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x_dict = {ntype: F.relu(self.input_proj[ntype](data[ntype].x)) for ntype in IN_DIMS}
        _, h_n = self.temporal_encoder(data["sensor_cluster"].x)
        x_dict["sensor_cluster"] = F.relu(h_n.squeeze(0))
        edge_index_dict = {etype: data[etype].edge_index for etype in EDGE_TYPES}

        h1 = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(h1[k] + x_dict[k]) for k in h1}
        h2 = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: h2[k] + x_dict[k] for k in h2}

        zone_embeddings = x_dict["zone"]  # [n_zones_in_batch, hidden]
        logits = self.head(zone_embeddings).squeeze(-1)  # one logit per zone node
        return logits
