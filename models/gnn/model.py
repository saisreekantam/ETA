"""
Heterogeneous GAT over all 7 plant zones at once: sensor-cluster nodes monitor their own
zone 1:1, individual sensor nodes feed their cluster, permit/presence/worker attach to
whichever zone they apply to, a global plant master node connects to every zone, and
zone<->zone edges follow real process-flow adjacency (feed -> reactor -> condenser ->
separator -> stripper/compressor recycle). Predicts a compound-risk logit PER ZONE node,
not one scalar per graph -- this is what makes zone-localization ("did it flag the right
zone") a real, checkable output instead of a placeholder.

TEMPORAL: sensor_cluster inputs are raw [WINDOW, MAX_CLUSTER_SENSORS] sequences encoded
by a GRU before message passing (the old design mean-pooled the window into 27 summary
stats, which can't tell a slow drift from a sudden step with the same mean).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv

from models.gnn.graph_builder import MAX_CLUSTER_SENSORS

NODE_TYPES = ["sensor_cluster", "sensor", "permit", "presence", "worker", "plant", "zone"]
EDGE_TYPES = [
    ("sensor", "feeds", "sensor_cluster"),
    ("sensor_cluster", "aggregates", "sensor"),
    ("sensor_cluster", "monitors", "zone"),
    ("zone", "monitored_by", "sensor_cluster"),
    ("permit", "authorizes", "zone"),
    ("zone", "authorized_by", "permit"),
    ("presence", "occupies", "zone"),
    ("zone", "occupied_by", "presence"),
    ("worker", "works_in", "zone"),
    ("zone", "staffed_by", "worker"),
    ("zone", "reports_to", "plant"),
    ("plant", "oversees", "zone"),
    ("zone", "flows_to", "zone"),
    ("zone", "flows_from", "zone"),
]

# sensor_cluster is intentionally absent: its input is a sequence handled by the GRU
IN_DIMS = {"sensor": 5, "permit": 5, "presence": 1, "worker": 2, "plant": 1, "zone": 7}
HIDDEN = 32


class CompoundRiskGNN(nn.Module):
    def __init__(self, hidden: int = HIDDEN, heads: int = 2):
        super().__init__()
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(IN_DIMS[ntype], hidden) for ntype in IN_DIMS
        })
        # temporal encoder: the 30-sample window per cluster, not its mean -- the final
        # hidden state is the cluster's node embedding entering message passing
        self.temporal_encoder = nn.GRU(MAX_CLUSTER_SENSORS, hidden, batch_first=True)

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
        _, h_n = self.temporal_encoder(data["sensor_cluster"].x)  # [1, n_clusters, hidden]
        x_dict["sensor_cluster"] = F.relu(h_n.squeeze(0))
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

    @torch.no_grad()
    def flow_attention(self, data) -> list[dict]:
        """Mean-over-heads GATv2 attention on the zone->zone process-flow edges at the
        2nd conv layer (whose input is the post-conv1 state, i.e. after sensor/permit
        evidence has reached each zone) -- 'which upstream neighbor is this zone
        listening to'. Single-graph input only; used for the dashboard's animated
        risk-propagation pipes, not for training. Note GAT attention normalizes over
        each TARGET's incoming flow edges, so a zone with one upstream neighbor always
        shows 1.0 -- the frontend scales by source risk to make the display meaningful."""
        from models.gnn.graph_builder import ZONE_VOCAB

        x_dict = {ntype: F.relu(self.input_proj[ntype](data[ntype].x)) for ntype in IN_DIMS}
        _, h_n = self.temporal_encoder(data["sensor_cluster"].x)
        x_dict["sensor_cluster"] = F.relu(h_n.squeeze(0))
        edge_index_dict = {etype: data[etype].edge_index for etype in EDGE_TYPES}
        h1 = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(h1[k] + x_dict[k]) for k in h1}

        conv = self.conv2.convs[("zone", "flows_to", "zone")]
        edge_index = data["zone", "flows_to", "zone"].edge_index
        _, (ei, alpha) = conv(x_dict["zone"], edge_index, return_attention_weights=True)
        alpha = alpha.mean(dim=1)  # [n_edges], heads averaged

        return [{
            "src": ZONE_VOCAB[int(ei[0, i])],
            "dst": ZONE_VOCAB[int(ei[1, i])],
            "attention": round(float(alpha[i]), 4),
        } for i in range(ei.size(1))]
