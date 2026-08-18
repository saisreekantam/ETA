"""
Per-run, per-zone flat feature extraction for TEP synthetic data -- the classical-baseline
counterpart of models/gnn/graph_builder.py's build_graph(), used the same way
eval/hai_validation/classical_features.py was for HAI: same last-WINDOW=30-sample slice,
same [mean, std, last-first delta, min, max] per channel the GNN's "sensor" nodes see, so
comparisons stay apples-to-apples with what the graph model is actually given.

Two feature sets per run, one row per zone (6 sensored zones; control_room is skipped --
it has no sensor cluster and both the GNN and z-score baseline score it ~0 by
construction):
  sensor_only     -- just the per-channel stats, matches the z-score baseline's inputs
  sensor_plus_ctx -- sensor stats + permit (has_permit, 4-way type one-hot) + presence
                     (has_presence, dwell_frac), zeroed for every zone except the run's
                     permit_zone -- matches exactly what the GNN's permit/presence/worker
                     nodes carry, so a classifier given this is "same information, no
                     graph" rather than "less information than the GNN".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.gnn.graph_builder import (
    CLUSTER_TO_ZONE, PERMIT_TYPE_VOCAB, RUN_SAMPLES, SENSOR_CLUSTERS, WINDOW,
)

SCORED_ZONES = list(CLUSTER_TO_ZONE.values())  # 6 zones with a sensor cluster; excludes control_room
ZONE_TO_CLUSTER = {z: c for c, z in CLUSTER_TO_ZONE.items()}


def _sensor_stats(window_df: pd.DataFrame, zone: str) -> np.ndarray:
    cols = SENSOR_CLUSTERS[ZONE_TO_CLUSTER[zone]]
    feats = []
    for col in cols:
        v = window_df[col].to_numpy(dtype=np.float32)
        feats.extend([v.mean(), v.std(), v[-1] - v[0], v.min(), v.max()])
    return np.array(feats, dtype=np.float32)


def _permit_presence_feat(permit: pd.Series, presence: pd.Series) -> np.ndarray:
    has_permit = float(bool(permit["has_permit"]))
    permit_onehot = [1.0 if (has_permit and permit["permit_type"] == p) else 0.0 for p in PERMIT_TYPE_VOCAB]
    has_presence = float(bool(presence["has_presence"]))
    dwell_frac = 0.0
    if has_presence:
        w_from, w_to = presence.get("from_sample"), presence.get("to_sample")
        if w_from is not None and w_to is not None and not (pd.isna(w_from) or pd.isna(w_to)):
            dwell_frac = (float(w_to) - float(w_from)) / RUN_SAMPLES
    return np.array([has_permit, *permit_onehot, has_presence, dwell_frac], dtype=np.float32)


def extract_run_features(sensor_df: pd.DataFrame, permit: pd.Series, presence: pd.Series, permit_zone: str):
    """Returns {zone: (sensor_only_feats, sensor_plus_ctx_feats)} for all 6 sensored zones."""
    window_df = sensor_df.iloc[-WINDOW:]
    ctx = _permit_presence_feat(permit, presence)
    zero_ctx = np.zeros_like(ctx)
    out = {}
    for zone in SCORED_ZONES:
        s = _sensor_stats(window_df, zone)
        this_ctx = ctx if zone == permit_zone else zero_ctx
        out[zone] = (s, np.concatenate([s, this_ctx]))
    return out
