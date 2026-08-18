"""
Ablation: same GRU temporal encoder per subsystem and the same zone-identity input
projection as HaiRiskGNN, but with BOTH HeteroConv/GATv2 rounds deleted entirely -- no
sensor_cluster->zone edge, no zone<->zone flow edge, no plant node. Each zone is scored
independently from nothing but its own 60s window + its own one-hot identity. This
isolates exactly one variable: does letting zones see each other and their own raw
per-channel sensor nodes through graph attention actually help, or would an equally-sized
non-relational per-zone classifier do just as well? Same hidden width and head shape as
HaiRiskGNN so the comparison is "graph vs no graph", not "big model vs small model".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from eval.hai_validation.graph_builder import ZONE_VOCAB, max_cluster_sensors
from eval.hai_validation.model import HIDDEN

NODE_TYPES = ["sensor_cluster", "zone"]  # only these two are actually read
IN_DIMS = {"zone": len(ZONE_VOCAB)}


class NoGraphAblationGNN(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(IN_DIMS[ntype], hidden) for ntype in IN_DIMS
        })
        self.temporal_encoder = nn.GRU(max_cluster_sensors(), hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data) -> torch.Tensor:
        zone_id = F.relu(self.input_proj["zone"](data["zone"].x))
        _, h_n = self.temporal_encoder(data["sensor_cluster"].x)
        cluster_emb = F.relu(h_n.squeeze(0))
        combined = F.relu(cluster_emb + zone_id)  # no message passing -- pure per-zone read
        logits = self.head(combined).squeeze(-1)
        return logits
