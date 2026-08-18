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

from models.gnn.graph_builder import CLUSTER_TO_ZONE, SENSOR_CLUSTERS, SENSOR_NODE_COLS, ZONE_VOCAB
from models.gnn.model import CompoundRiskGNN

CLUSTER_NAMES = list(SENSOR_CLUSTERS.keys())


def explain_zone(model: CompoundRiskGNN, graph, zone: str, top_k: int = 3) -> dict:
    """graph must already be normalized (same stats used at training time)."""
    zone_idx = ZONE_VOCAB.index(zone)

    batch = next(iter(DataLoader([graph], batch_size=1)))
    for ntype in ("sensor_cluster", "sensor", "permit", "presence", "worker"):
        batch[ntype].x = batch[ntype].x.clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(batch)
    logits[zone_idx].backward()

    # sensor_cluster.x is [n_clusters, WINDOW, channels] -- collapse everything but the
    # node axis to get one saliency per cluster
    cluster_grad = batch["sensor_cluster"].x.grad.abs().flatten(1).sum(dim=1)  # [6]
    sensor_grad = batch["sensor"].x.grad.abs().sum(dim=1)  # [33], one per physical channel
    permit_grad = batch["permit"].x.grad.abs().sum().item()
    presence_grad = (batch["presence"].x.grad.abs().sum()
                     + batch["worker"].x.grad.abs().sum()).item()

    cluster_scores = {name: cluster_grad[i].item() for i, name in enumerate(CLUSTER_NAMES)}
    ranked_clusters = sorted(cluster_scores.items(), key=lambda kv: -kv[1])[:top_k]

    # per-sensor nodes give genuinely individual instrument saliency now, instead of
    # dumping every sensor of the top clusters
    ranked_sensors = sorted(
        ((col, sensor_grad[i].item()) for i, (_, col) in enumerate(SENSOR_NODE_COLS)),
        key=lambda kv: -kv[1],
    )
    contributing_sensors = [col for col, _ in ranked_sensors[:5]]
    # normalized to the top sensor (=1.0) so the frontend can show "XMEAS(7) at 83%"
    # without knowing anything about gradient scales
    top_mag = ranked_sensors[0][1] if ranked_sensors and ranked_sensors[0][1] > 0 else 1.0
    sensor_saliency = [{"sensor": col, "saliency": round(mag / top_mag, 4)}
                       for col, mag in ranked_sensors[:5]]

    return {
        "top_sensor_clusters": [{"cluster": c, "zone": CLUSTER_TO_ZONE[c], "saliency": round(s, 4)}
                                  for c, s in ranked_clusters],
        "contributing_sensors": contributing_sensors,
        "sensor_saliency": sensor_saliency,
        "permit_saliency": round(permit_grad, 4),
        "presence_saliency": round(presence_grad, 4),
    }


# node types that make up each "evidence type" a counterfactual can remove -- worker
# carries dwell-time detail beyond presence's plain flag, so it travels with presence
_COUNTERFACTUAL_GROUPS = [
    ("sensor_evidence", ("sensor_cluster", "sensor")),
    ("active_permit", ("permit",)),
    ("worker_presence", ("presence", "worker")),
]


def compute_counterfactuals(model: CompoundRiskGNN, graph, zone: str) -> list[dict]:
    """'Would this zone still be flagged without X' for each evidence type, answered by
    actually re-scoring rather than guessing from saliency. Since node features are
    z-score normalized at inference time ((x - mean) / std -- see graph_builder / model
    loading), zeroing a node type's tensor IS "this input at its training-set average":
    no separate normal-condition baseline to fetch, no new stats to maintain.

    Cheap: one forward pass per evidence type (no backward), on top of the one
    explain_zone already does -- the model is tiny (67K params) and already loaded.
    graph must already be normalized, same as explain_zone."""
    zone_idx = ZONE_VOCAB.index(zone)

    baseline_batch = next(iter(DataLoader([graph], batch_size=1)))
    with torch.no_grad():
        baseline_score = torch.sigmoid(model(baseline_batch))[zone_idx].item()

    results = []
    for label, node_types in _COUNTERFACTUAL_GROUPS:
        batch = next(iter(DataLoader([graph], batch_size=1)))
        for nt in node_types:
            batch[nt].x = torch.zeros_like(batch[nt].x)
        with torch.no_grad():
            score = torch.sigmoid(model(batch))[zone_idx].item()
        results.append({
            "removed_factor": label,
            "score_with": round(baseline_score, 4),
            "score_without": round(score, 4),
            "delta": round(baseline_score - score, 4),
        })

    return sorted(results, key=lambda r: -r["delta"])
