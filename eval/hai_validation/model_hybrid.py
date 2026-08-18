"""
HaiRiskGNN with one change: the "zone" node's input is [one-hot identity, PCA
reconstruction-error score] instead of bare one-hot (see dataset_hybrid.py). Tests
whether combining the graph's cross-zone relational reasoning (confirmed to matter by
model_no_graph.py's ablation) with PCA's strong per-zone anomaly signal (confirmed to
beat the graph alone on zone localization) beats either one individually.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv

from eval.hai_validation.graph_builder import ZONE_VOCAB, max_cluster_sensors
from eval.hai_validation.model import EDGE_TYPES, HIDDEN

NODE_TYPES = ["sensor_cluster", "sensor", "zone", "plant"]
IN_DIMS = {"sensor": 5, "zone": len(ZONE_VOCAB) + 1, "plant": 1}


class HybridRiskGNN(nn.Module):
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

        zone_embeddings = x_dict["zone"]
        logits = self.head(zone_embeddings).squeeze(-1)
        return logits
