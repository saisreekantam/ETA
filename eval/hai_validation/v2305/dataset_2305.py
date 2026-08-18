"""
23.05 counterpart of eval/hai_validation/dataset.py. Same file-level split discipline
(no window straddles a split boundary), same per-zone real-ground-truth labeling, just
against v2305/common.py's file layout and attack_labels_2305.py's catalog.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from eval.hai_validation.v2305.attack_labels_2305 import attack_intervals
from eval.hai_validation.v2305.common import STRIDE, WINDOW, ZONE_VOCAB, build_graph, load_file


def _zone_labels_for_file(name: str, timestamps) -> np.ndarray:
    labels = np.zeros((len(timestamps), len(ZONE_VOCAB)), dtype=np.float32)
    if name.startswith("hai23-train"):
        return labels
    ts = timestamps.to_numpy()
    for start, end, subsystems in attack_intervals(name):
        mask = (ts >= start.to_numpy()) & (ts <= end.to_numpy())
        for z in subsystems:
            labels[mask, ZONE_VOCAB.index(z)] = 1.0
    return labels


class HaiWindowDataset2305(Dataset):
    def __init__(self, filenames: list[str], stride: int = STRIDE):
        self.frames = [load_file(f) for f in filenames]
        self.zone_labels = [
            _zone_labels_for_file(f, df["timestamp"]) for f, df in zip(filenames, self.frames)
        ]
        self.index: list[tuple[int, int]] = []
        for fi, df in enumerate(self.frames):
            n = len(df)
            for end in range(WINDOW, n + 1, stride):
                self.index.append((fi, end))

    def __len__(self) -> int:
        return len(self.index)

    def zone_label_matrix(self) -> np.ndarray:
        return np.stack([self.zone_labels[fi][end - 1] for fi, end in self.index])

    def __getitem__(self, i: int):
        fi, end = self.index[i]
        df = self.frames[fi]
        window_df = df.iloc[end - WINDOW:end]
        graph = build_graph(window_df)
        graph["zone"].y = torch.tensor(self.zone_labels[fi][end - 1], dtype=torch.float32)
        return graph


if __name__ == "__main__":
    from eval.hai_validation.v2305.common import TEST_FILES, TRAIN_FILES, VAL_FILES

    for name, files in [("train", TRAIN_FILES), ("val", VAL_FILES), ("test", TEST_FILES)]:
        ds = HaiWindowDataset2305(files)
        zl = ds.zone_label_matrix()
        print(f"{name}: {len(ds)} windows, any-zone positive {(zl.sum(axis=1)>0).sum()} "
              f"({(zl.sum(axis=1)>0).mean():.3%}); per-zone positive counts: "
              f"{dict(zip(ZONE_VOCAB, zl.sum(axis=0).astype(int)))}")
