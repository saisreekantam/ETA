"""
Vectorized per-window, per-zone feature extraction for the classical baselines
(PCA reconstruction, Isolation Forest, Random Forest) -- same five per-channel stats the
GNN's "sensor" nodes see (mean, std, last-first delta, min, max), computed with pandas
.rolling() over the whole file at once instead of graph_builder.build_graph's per-window
Python loop. That loop is fine for HeteroData objects built once per __getitem__ during
GNN training, but re-running it per window per classical model would be needless -- these
models want one flat feature matrix per zone, extracted once and reused across all three.

Row alignment matches dataset.py/baseline.py exactly: a window ending at row index `end`
(0-based, exclusive range end) is windows.iloc[end-WINDOW:end], keyed by its last row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.hai_validation.graph_builder import STRIDE, WINDOW, ZONE_VOCAB, subsystem_columns


def extract_zone_features(df: pd.DataFrame, stride: int = STRIDE) -> dict[str, np.ndarray]:
    """Returns {zone: [n_windows, n_channels*5]} float32 arrays, row order = window end
    indices WINDOW-1, WINDOW-1+stride, ... (0-based row of the window's last sample)."""
    cols = subsystem_columns()
    ends = np.arange(WINDOW, len(df) + 1, stride)  # matches dataset.py's `end` (exclusive)
    row_idx = ends - 1  # 0-based index of the window's last sample

    out = {}
    for zone in ZONE_VOCAB:
        zcols = cols[zone]
        block = df[zcols]
        roll = block.rolling(window=WINDOW)
        mean = roll.mean().to_numpy()[row_idx]
        std = roll.std().to_numpy()[row_idx]
        mn = roll.min().to_numpy()[row_idx]
        mx = roll.max().to_numpy()[row_idx]
        first = block.shift(WINDOW - 1).to_numpy()[row_idx]
        last = block.to_numpy()[row_idx]
        delta = last - first
        feats = np.concatenate([mean, std, delta, mn, mx], axis=1).astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0)
        out[zone] = feats
    return out, row_idx
