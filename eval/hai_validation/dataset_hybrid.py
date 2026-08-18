"""
HaiWindowDataset variant that appends each zone's precomputed PCA reconstruction-error
score (precompute_pca_features.py) onto the "zone" node's input feature vector -- turning
the zone's one-hot identity (dim len(ZONE_VOCAB)) into [one-hot, pca_score] (dim
len(ZONE_VOCAB)+1). This is the hybrid architecture's only change from HaiRiskGNN: the
graph still does its own 2-hop GATv2 cross-zone reasoning, but each zone starts from a
richer per-zone anomaly signal instead of bare identity, following model_hybrid.py's
IN_DIMS["zone"] = len(ZONE_VOCAB) + 1.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import torch

from eval.hai_validation.dataset import HaiWindowDataset
from eval.hai_validation.graph_builder import STRIDE, WINDOW

PCA_SCORES_PATH = Path(__file__).resolve().parent / "checkpoints_classical" / "pca_scores_by_file.joblib"


class HaiWindowDatasetHybrid(HaiWindowDataset):
    def __init__(self, filenames: list[str], stride: int = STRIDE):
        super().__init__(filenames, stride)
        scores_by_file = joblib.load(PCA_SCORES_PATH)
        self.pca_scores = [scores_by_file[f] for f in filenames]  # [n_windows_in_file, N_ZONES]
        self.stride = stride

    def __getitem__(self, i: int):
        fi, end = self.index[i]
        graph = super().__getitem__(i)
        k = (end - WINDOW) // self.stride
        pca_row = torch.tensor(self.pca_scores[fi][k], dtype=torch.float32).unsqueeze(-1)  # [N_ZONES, 1]
        graph["zone"].x = torch.cat([graph["zone"].x, pca_row], dim=-1)
        return graph
