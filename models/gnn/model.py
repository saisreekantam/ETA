"""
Heterogeneous GAT over all 7 plant zones at once: sensor-cluster nodes monitor their own
zone 1:1, permit/presence attach to whichever zone they apply to, and zone<->zone edges
follow real process-flow adjacency (feed -> reactor -> condenser -> separator ->
stripper/compressor recycle). Predicts a compound-risk logit PER ZONE node, not one
scalar per graph -- this is what makes zone-localization ("did it flag the right zone")
a real, checkable output instead of a placeholder.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv

NODE_TYPES = ["sensor_cluster", "permit", "presence", "zone"]
EDGE_TYPES = [
    ("sensor_cluster", "monitors", "zone"),
    ("zone", "monitored_by", "sensor_cluster"),
    ("permit", "authorizes", "zone"),
    ("zone", "authorized_by", "permit"),
    ("presence", "occupies", "zone"),
    ("zone", "occupied_by", "presence"),
    ("zone", "flows_to", "zone"),
    ("zone", "flows_from", "zone"),
]

IN_DIMS = {"sensor_cluster": 27, "permit": 5, "presence": 1, "zone": 7}
HIDDEN = 32


class CompoundRiskGNN(nn.Module):
    def __init__(self, hidden: int = HIDDEN, heads: int = 2):
        super().__init__()
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(IN_DIMS[ntype], hidden) for ntype in NODE_TYPES
        })

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
        x_dict = {ntype: F.relu(self.input_proj[ntype](data[ntype].x)) for ntype in NODE_TYPES}
        edge_index_dict = {etype: data[etype].edge_index for etype in EDGE_TYPES}

        # Residual connections matter here regardless of topology: without carrying each
        # layer's output forward, a 2nd conv recomputes "zone" purely from leaf embeddings
        # that themselves only depended on the *previous* zone state -- verified empirically
        # in the original single-zone design that without residuals the model collapses to
        # predicting the constant class prior. With zone<->zone edges now present, 2 hops
        # also lets risk genuinely propagate from one zone into its process-flow neighbors.
        h1 = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(h1[k] + x_dict[k]) for k in h1}
        h2 = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: h2[k] + x_dict[k] for k in h2}

        zone_embeddings = x_dict["zone"]  # [7, hidden] per graph -- one row per zone
        logits = self.head(zone_embeddings).squeeze(-1)  # [7] per graph
        return logits
