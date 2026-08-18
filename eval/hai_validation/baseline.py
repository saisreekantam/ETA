"""
Same matched baseline as models/gnn/baseline_threshold.py: per-subsystem z-score fit on
normal-only data, alarm when any channel in a subsystem crosses THRESHOLD_SIGMA. Stats
are fit on train1.csv only (HAI's genuinely attack-free file), never on test1/test2 even
though those are in the GNN's training pool -- the baseline must not see any attack
window during fitting, matched-fairness requires it stay a pure "what does normal look
like" model exactly like the TEP baseline.

Per-zone: each subsystem's max z-score and did_alert are kept separate (not pooled),
scored against the real per-zone ground truth attack_labels.py reconstructs -- matching
the GNN's per-zone output now that real per-subsystem supervision exists (see
dataset.py's docstring for how that ground truth was recovered).

CHANNEL FILTER (real-data wrinkle TEP's clean simulation never surfaces): 24 of HAI's 86
tags are constant-valued status/digital bits during the entire attack-free train1.csv
(pump running-flags, mode switches, trip-exchange bits, etc). A z-score is undefined on a
zero-variance series -- the code guards with a 1e-6 floor, but that turns any bit flip at
all into an enormous z-score, and the baseline alarms on literally 100% of windows
regardless of label, which we saw happen. This isn't a bug in the port: it's the same
"per-channel threshold has no way to represent structure" flaw the paper argues, just
sharper on real telemetry than on TEP's clean continuous variables. Plants don't run
z-score alarms on digital status bits anyway (those get discrete state-change alarms) --
so, matching real operational practice, the baseline here is fit only over tags with at
least MIN_UNIQUE_VALUES distinct readings in train1.csv, i.e. genuinely continuous
analog measurements. The excluded bits are NOT hidden from the GNN, which sees all 86
channels through the sensor/sensor_cluster nodes same as before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.hai_validation.dataset import TEST_FILES, VAL_FILES, _load_file, _zone_labels_for_file
from eval.hai_validation.graph_builder import HAI_DIR, STRIDE, WINDOW, ZONE_VOCAB, subsystem_columns

THRESHOLD_SIGMA = 4.0
MIN_UNIQUE_VALUES = 10


def fit_normal_stats() -> pd.DataFrame:
    df = pd.read_csv(HAI_DIR / "train1.csv")
    cols = [c for cs in subsystem_columns().values() for c in cs]
    continuous_cols = [c for c in cols if df[c].nunique() >= MIN_UNIQUE_VALUES]
    stats = pd.DataFrame({
        "mean": df[continuous_cols].mean(),
        "std": df[continuous_cols].std().replace(0, 1e-6),
    })
    stats.attrs["continuous_cols"] = continuous_cols
    return stats


def score_windows(filenames: list[str], stats: pd.DataFrame, stride: int = STRIDE) -> pd.DataFrame:
    cols = subsystem_columns()
    continuous = set(stats.attrs["continuous_cols"])
    zone_cols = {zone: [c for c in cs if c in continuous] for zone, cs in cols.items()}
    rows = []
    for fname in filenames:
        df = _load_file(fname)
        zone_labels = _zone_labels_for_file(fname, df["timestamp"])
        z = pd.DataFrame(index=df.index)
        for zone in ZONE_VOCAB:
            zcols = zone_cols[zone]
            z[zone] = ((df[zcols] - stats["mean"][zcols]) / stats["std"][zcols]).abs().max(axis=1)

        for end in range(WINDOW, len(df) + 1, stride):
            window = z.iloc[end - WINDOW:end]
            row = {"file": fname, "window_end": end}
            for zi, zone in enumerate(ZONE_VOCAB):
                max_z = float(window[zone].max())
                row[f"max_z__{zone}"] = max_z
                row[f"did_alert__{zone}"] = max_z > THRESHOLD_SIGMA
                row[f"label__{zone}"] = float(zone_labels[end - 1, zi])
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    stats = fit_normal_stats()
    val = score_windows(VAL_FILES, stats)
    test = score_windows(TEST_FILES, stats)
    for name, d in [("val", val), ("test", test)]:
        for zone in ZONE_VOCAB:
            pos = d[d[f"label__{zone}"] == 1]
            if len(pos) == 0:
                print(f"{name}/{zone}: no positive windows")
                continue
            print(f"{name}/{zone}: alert rate on attack windows = {pos[f'did_alert__{zone}'].mean():.2%} "
                  f"({int(pos[f'did_alert__{zone}'].sum())}/{len(pos)}), "
                  f"overall alert rate = {d[f'did_alert__{zone}'].mean():.2%}")
    test.to_parquet(HAI_DIR / "baseline_test_predictions.parquet", index=False)
    val.to_parquet(HAI_DIR / "baseline_val_predictions.parquet", index=False)
