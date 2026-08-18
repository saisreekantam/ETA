"""
23.05 counterpart of baseline.py (z-score) and train_classical.py's PCA baseline,
scoped down to just these two -- P2/P3 have near-zero positive examples in this
dataset's real attack campaign (see attack_labels_2305.py), so the richer classical
suite (Isolation Forest/Random Forest/ablation/hybrid) wouldn't have anything meaningful
to learn from on those zones and isn't worth re-running here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eval.hai_validation.v2305.common import STRIDE, WINDOW, ZONE_VOCAB, load_file, subsystem_columns
from eval.hai_validation.v2305.dataset_2305 import _zone_labels_for_file

THRESHOLD_SIGMA = 4.0
MIN_UNIQUE_VALUES = 10
PCA_COMPONENTS = 10
SEED = 0


def extract_zone_features(df: pd.DataFrame, stride: int = STRIDE) -> tuple[dict, np.ndarray]:
    cols = subsystem_columns()
    ends = np.arange(WINDOW, len(df) + 1, stride)
    row_idx = ends - 1
    out = {}
    for zone in ZONE_VOCAB:
        block = df[cols[zone]]
        roll = block.rolling(window=WINDOW)
        mean = roll.mean().to_numpy()[row_idx]
        std = roll.std().to_numpy()[row_idx]
        mn = roll.min().to_numpy()[row_idx]
        mx = roll.max().to_numpy()[row_idx]
        first = block.shift(WINDOW - 1).to_numpy()[row_idx]
        last = block.to_numpy()[row_idx]
        feats = np.concatenate([mean, std, last - first, mn, mx], axis=1).astype(np.float32)
        out[zone] = np.nan_to_num(feats, nan=0.0)
    return out, row_idx


def fit_zscore_stats(normal_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = subsystem_columns()
    stats = {}
    for zone in ZONE_VOCAB:
        zcols = [c for c in cols[zone] if normal_df[c].nunique() >= MIN_UNIQUE_VALUES]
        stats[zone] = pd.DataFrame({
            "cols": zcols,
            "mean": normal_df[zcols].mean().to_numpy(),
            "std": normal_df[zcols].std().replace(0, 1e-6).to_numpy(),
        })
    return stats


def score_zscore(df: pd.DataFrame, zone_labels: np.ndarray, zscore_stats: dict, stride: int = STRIDE) -> pd.DataFrame:
    rows = []
    zone_z = {}
    for zone in ZONE_VOCAB:
        s = zscore_stats[zone]
        zcols = s["cols"].tolist()
        z = ((df[zcols] - s["mean"].to_numpy()) / s["std"].to_numpy()).abs().max(axis=1)
        zone_z[zone] = z
    ends = np.arange(WINDOW, len(df) + 1, stride)
    for i, end in enumerate(ends):
        row = {"window_end": end}
        for zi, zone in enumerate(ZONE_VOCAB):
            window = zone_z[zone].iloc[end - WINDOW:end]
            max_z = float(window.max())
            row[f"max_z__{zone}"] = max_z
            row[f"did_alert__{zone}"] = max_z > THRESHOLD_SIGMA
            row[f"label__{zone}"] = float(zone_labels[end - 1, zi])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_pca_models(normal_feats: dict) -> dict:
    models = {}
    for zone in ZONE_VOCAB:
        scaler = StandardScaler().fit(normal_feats[zone])
        normal_x = scaler.transform(normal_feats[zone])
        n_components = min(PCA_COMPONENTS, normal_x.shape[1] - 1, normal_x.shape[0] - 1)
        pca = PCA(n_components=max(n_components, 1), random_state=SEED).fit(normal_x)
        models[zone] = (scaler, pca)
    return models


def pca_reconstruction_error(models: dict, feats: dict) -> dict:
    out = {}
    for zone, (scaler, pca) in models.items():
        x = scaler.transform(feats[zone])
        recon = pca.inverse_transform(pca.transform(x))
        out[zone] = np.square(x - recon).mean(axis=1).astype(np.float32)
    return out
