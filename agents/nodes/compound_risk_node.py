"""
Compound Risk Detection Engine node. Loads the trained multi-zone heterogeneous GAT
(models/gnn) and the zone-aware naive baseline, scores ALL 7 plant zones in one
inference call, and writes per-zone scores into state -- this is what lets the
dashboard heatmap show genuine plant-wide variation instead of one zone lighting up.
Each zone's score also carries real gradient-based attribution (models/gnn/attribution),
not a placeholder list of sensor names.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from agents.state import PipelineState, ZoneRiskScore
from models.gnn.attribution import explain_zone
from models.gnn.baseline_threshold import THRESHOLD_SIGMA, ZONE_TO_SENSOR_COLS
from models.gnn.graph_builder import ZONE_VOCAB, build_graph
from models.gnn.model import CompoundRiskGNN

REPO_ROOT = Path(__file__).resolve().parents[2]

SENSOR_COLS = [f"XMEAS({i})" for i in range(1, 42)] + [f"XMV({i})" for i in range(1, 12)]

_model = None
_norm_stats = None
_baseline_stats = None


def _lazy_load():
    global _model, _norm_stats, _baseline_stats
    if _model is None:
        _model = CompoundRiskGNN()
        _model.load_state_dict(torch.load(REPO_ROOT / "models" / "gnn" / "checkpoint.pt"))
        _model.eval()
        _norm_stats = torch.load(REPO_ROOT / "models" / "gnn" / "norm_stats.pt")

        normal_runs = pd.read_csv(REPO_ROOT / "data" / "synthetic" / "manifest.csv")
        normal_runs = normal_runs[normal_runs["condition"] == "normal"]
        frames = [pd.read_parquet(REPO_ROOT / p) for p in normal_runs["path"]]
        all_normal = pd.concat(frames, ignore_index=True)
        _baseline_stats = (all_normal[SENSOR_COLS].mean(), all_normal[SENSOR_COLS].std().replace(0, 1e-6))


def _build_normalized_graph(sensor_df: pd.DataFrame, permit: dict, presence: dict, permit_zone: str):
    # graph_builder.build_graph expects the raw training-data shape (has_permit /
    # has_presence booleans); the agents/state.py PipelineState schema uses status/
    # entry_time instead -- translate here rather than changing the trained model's
    # expected input shape.
    permit_features = pd.Series({
        "has_permit": permit.get("status") == "active",
        "permit_type": permit.get("permit_type", "general"),
    })
    presence_features = pd.Series({
        "has_presence": bool(presence.get("has_presence", presence.get("entry_time") is not None)),
    })
    graph = build_graph(sensor_df, permit_features, presence_features, permit_zone)
    for ntype, (mean, std) in _norm_stats.items():
        graph[ntype].x = (graph[ntype].x - mean) / std
    return graph


def _baseline_scores_by_zone(sensor_df: pd.DataFrame) -> dict[str, float]:
    mean, std = _baseline_stats
    scores = {}
    for zone, cols in ZONE_TO_SENSOR_COLS.items():
        z = (sensor_df[cols] - mean[cols]) / std[cols]
        scores[zone] = float((z.abs() > THRESHOLD_SIGMA).any(axis=1).mean())
    scores["control_room"] = 0.0
    return scores


def compound_risk_node(state: PipelineState) -> dict:
    _lazy_load()
    sensor_df = pd.DataFrame(state["sensor_window"])
    permit_zone = state["permits"][0]["zone"] if state["permits"] else "reactor_zone"
    permit = state["permits"][0] if state["permits"] else {"status": "expired", "permit_type": "general"}
    presence = state["worker_presence"][0] if state["worker_presence"] else {"has_presence": False}

    graph = _build_normalized_graph(sensor_df, permit, presence, permit_zone)
    with torch.no_grad():
        batch = next(iter(DataLoader([graph], batch_size=1)))
        probs = torch.sigmoid(_model(batch)).tolist()

    baseline_scores = _baseline_scores_by_zone(sensor_df)

    zone_risk_scores = []
    for i, zone in enumerate(ZONE_VOCAB):
        explanation = explain_zone(_model, graph, zone) if zone != "control_room" else {
            "contributing_sensors": [], "permit_saliency": 0.0, "presence_saliency": 0.0,
        }
        zone_risk_scores.append(ZoneRiskScore(
            zone=zone,
            compound_risk_score=round(probs[i], 4),
            baseline_risk_score=round(baseline_scores.get(zone, 0.0), 4),
            contributing_sensors=explanation["contributing_sensors"],
            contributing_faults=None,
        ))

    top = max(zone_risk_scores, key=lambda s: s["compound_risk_score"])
    audit_entry = (f"compound_risk_node: top_zone={top['zone']} gnn_score={top['compound_risk_score']} "
                    f"baseline_score={top['baseline_risk_score']} "
                    f"(scored all {len(zone_risk_scores)} zones)")
    return {"zone_risk_scores": zone_risk_scores, "audit_log": state["audit_log"] + [audit_entry]}
