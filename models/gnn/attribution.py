"""
Gradient-based saliency for a single zone's risk logit -- replaces the earlier
placeholder (`contributing_sensors` was just the first 5 sensor column names,
not actually computed from the model). This is a vanilla-gradient saliency: backprop
the target zone's logit to the sensor_cluster/permit/presence input tensors and rank by
gradient magnitude. Cheap (one backward pass per explanation) and honest about what it
is -- not Integrated Gradients or SHAP, just "which inputs the logit is locally most
sensitive to," which is enough to answer "why did the agent flag this zone" in the demo.
"""
from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader

from models.gnn.graph_builder import CLUSTER_TO_ZONE, SENSOR_CLUSTERS, ZONE_VOCAB
from models.gnn.model import CompoundRiskGNN

CLUSTER_NAMES = list(SENSOR_CLUSTERS.keys())


def explain_zone(model: CompoundRiskGNN, graph, zone: str, top_k: int = 3) -> dict:
    """graph must already be normalized (same stats used at training time)."""
    zone_idx = ZONE_VOCAB.index(zone)

    batch = next(iter(DataLoader([graph], batch_size=1)))
    for ntype in ("sensor_cluster", "permit", "presence"):
        batch[ntype].x = batch[ntype].x.clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(batch)
    logits[zone_idx].backward()

    sensor_grad = batch["sensor_cluster"].x.grad.abs().sum(dim=1)  # [6]
    permit_grad = batch["permit"].x.grad.abs().sum().item()
    presence_grad = batch["presence"].x.grad.abs().sum().item()

    cluster_scores = {name: sensor_grad[i].item() for i, name in enumerate(CLUSTER_NAMES)}
    ranked_clusters = sorted(cluster_scores.items(), key=lambda kv: -kv[1])[:top_k]

    contributing_sensors = []
    for cluster_name, _ in ranked_clusters:
        contributing_sensors.extend(SENSOR_CLUSTERS[cluster_name])

    return {
        "top_sensor_clusters": [{"cluster": c, "zone": CLUSTER_TO_ZONE[c], "saliency": round(s, 4)}
                                  for c, s in ranked_clusters],
        "contributing_sensors": contributing_sensors[:5],
        "permit_saliency": round(permit_grad, 4),
        "presence_saliency": round(presence_grad, 4),
    }
