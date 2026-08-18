"""
Precomputes the per-zone PCA reconstruction-error score for EVERY window in every split
(train pool, val, test), using the exact same normal-only-fit models train_classical.py's
standalone PCA baseline uses. Computed once, up front, and saved -- so the hybrid model's
dataset (dataset_hybrid.py) can just look scores up by (file, window index) instead of
re-running PCA transforms inside __getitem__ on every epoch.

Run: `python -m eval.hai_validation.precompute_pca_features` (after train_classical.py
has written pca_models.joblib).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from eval.hai_validation.classical_features import extract_zone_features
from eval.hai_validation.dataset import TEST_FILES, TRAIN_FILES, VAL_FILES, _load_file
from eval.hai_validation.graph_builder import ZONE_VOCAB
from eval.hai_validation.train_classical import OUT_DIR as CLASSICAL_OUT_DIR
from eval.hai_validation.train_classical import pca_reconstruction_error

OUT_PATH = Path(__file__).resolve().parent / "checkpoints_classical" / "pca_scores_by_file.joblib"


def main():
    pca_models = joblib.load(CLASSICAL_OUT_DIR / "pca_models.joblib")

    scores_by_file: dict[str, np.ndarray] = {}
    all_files = list(dict.fromkeys(TRAIN_FILES + VAL_FILES + TEST_FILES))
    for fname in all_files:
        df = _load_file(fname)
        zone_feats, row_idx = extract_zone_features(df)
        zone_scores = pca_reconstruction_error(pca_models, zone_feats)
        # [n_windows, N_ZONES] in ZONE_VOCAB order, aligned with HaiWindowDataset's own
        # (file, window_end) index for this file
        scores_by_file[fname] = np.stack([zone_scores[z] for z in ZONE_VOCAB], axis=1)
        print(f"{fname}: {scores_by_file[fname].shape[0]} windows")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scores_by_file, OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
