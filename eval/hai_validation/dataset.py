"""
Windowing over the raw HAI CSVs. Splits are by FILE, not by window, so no two windows
from the same contiguous recording session can land on both sides of a split boundary --
the same run-level-split discipline models/gnn/train.py uses for TEP, adapted to HAI's
"one CSV = one continuous recording session" structure instead of TEP's "one CSV = one
120-sample episode".

  train pool : train1.csv (all-normal) + test1.csv + test2.csv  (has the labeled attacks
               the model needs to learn from; HAI's train*.csv files are attack-free by
               design, only test*.csv contains induced-attack windows)
  val        : test3.csv
  test        : test4.csv (held out entirely until final evaluation)

A window's PER-ZONE label vector (length len(ZONE_VOCAB) = 4, one slot per P1/P2/P3/P4)
has a 1 at zone z if attack_labels.py's reconstructed catalog places an attack targeting
z with an interval covering the window's LAST sample's timestamp -- "is subsystem z under
attack right now, given the last 60s of history", matching the per-zone real-time framing
models/gnn/graph_builder.py's zone_label_vector uses for TEP. train1.csv (no attacks were
ever conducted during its collection) gets an all-zero label vector without needing to
look anything up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from eval.hai_validation.attack_labels import attack_intervals
from eval.hai_validation.graph_builder import HAI_DIR, STRIDE, WINDOW, ZONE_VOCAB, build_graph, subsystem_columns

TRAIN_FILES = ["train1.csv", "test1.csv", "test2.csv"]
VAL_FILES = ["test3.csv"]
TEST_FILES = ["test4.csv"]


def _load_file(name: str) -> pd.DataFrame:
    cols = ["timestamp", "Attack"] + [c for cs in subsystem_columns().values() for c in cs]
    df = pd.read_csv(HAI_DIR / name, usecols=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _zone_labels_for_file(name: str, timestamps: pd.Series) -> np.ndarray:
    """[n_rows, len(ZONE_VOCAB)] binary matrix; all-zero for files with no attack catalog."""
    labels = np.zeros((len(timestamps), len(ZONE_VOCAB)), dtype=np.float32)
    if name not in ("test1.csv", "test2.csv", "test3.csv", "test4.csv"):
        return labels
    ts = timestamps.to_numpy()
    for start, end, subsystems in attack_intervals(name):
        mask = (ts >= start.to_numpy()) & (ts <= end.to_numpy())
        for z in subsystems:
            labels[mask, ZONE_VOCAB.index(z)] = 1.0
    return labels


class HaiWindowDataset(Dataset):
    def __init__(self, filenames: list[str], stride: int = STRIDE):
        self.frames = [_load_file(f) for f in filenames]
        self.zone_labels = [
            _zone_labels_for_file(f, df["timestamp"]) for f, df in zip(filenames, self.frames)
        ]
        self.index: list[tuple[int, int]] = []  # (frame_idx, window_end_row_exclusive)
        for fi, df in enumerate(self.frames):
            n = len(df)
            for end in range(WINDOW, n + 1, stride):
                self.index.append((fi, end))

    def __len__(self) -> int:
        return len(self.index)

    def zone_label_matrix(self) -> np.ndarray:
        """[n_windows, len(ZONE_VOCAB)]"""
        return np.stack([self.zone_labels[fi][end - 1] for fi, end in self.index])

    def labels(self) -> np.ndarray:
        """Global (any-zone) label per window, for reporting/back-compat."""
        return (self.zone_label_matrix().sum(axis=1) > 0).astype(int)

    def __getitem__(self, i: int):
        fi, end = self.index[i]
        df = self.frames[fi]
        window_df = df.iloc[end - WINDOW:end]
        graph = build_graph(window_df)
        graph["zone"].y = torch.tensor(self.zone_labels[fi][end - 1], dtype=torch.float32)
        return graph


if __name__ == "__main__":
    train_ds = HaiWindowDataset(TRAIN_FILES)
    val_ds = HaiWindowDataset(VAL_FILES)
    test_ds = HaiWindowDataset(TEST_FILES)
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        zl = ds.zone_label_matrix()
        print(f"{name}: {len(ds)} windows, any-zone positive {(zl.sum(axis=1)>0).sum()} "
              f"({(zl.sum(axis=1)>0).mean():.3%}); per-zone positive counts: "
              f"{dict(zip(ZONE_VOCAB, zl.sum(axis=0).astype(int)))}")
