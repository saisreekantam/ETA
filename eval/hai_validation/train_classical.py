"""
Three more per-zone baselines, all reusing classical_features.py's vectorized window
features and dataset.py's real per-zone attack labels, scored with the exact same
protocol as the GNN/ablation/z-score comparison in evaluate.py:

  pca            -- unsupervised. StandardScaler + PCA fit on train1.csv (genuinely
                     attack-free) only, per zone. Anomaly score = reconstruction error.
                     The original ICS anomaly-detection baseline (Mathur & Tippenhauer's
                     SWaT work) predates deep learning entirely.
  isolation_forest -- unsupervised. sklearn IsolationForest fit on the same train1-only
                     normal features per zone. Score = -sklearn's anomaly score (higher
                     = more anomalous, to match the sign convention of the other scores).
  random_forest  -- supervised. Fit per zone on the FULL train pool (train1+test1+test2)
                     with real attack labels, same features the GNN's sensor nodes see --
                     tests whether the graph/GRU architecture beats a standard supervised
                     classifier given identical information, no graph required.

Run: `python -m eval.hai_validation.train_classical`
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from eval.hai_validation.classical_features import extract_zone_features
from eval.hai_validation.dataset import TEST_FILES, TRAIN_FILES, _load_file, _zone_labels_for_file
from eval.hai_validation.graph_builder import HAI_DIR, ZONE_VOCAB

OUT_DIR = Path(__file__).resolve().parent / "checkpoints_classical"
PCA_COMPONENTS = 10
SEED = 0


def _features_and_labels(filenames: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Concatenates per-zone features/labels across multiple files, in file order --
    matches HaiWindowDataset's own file-concatenation order exactly."""
    feats_by_zone = {z: [] for z in ZONE_VOCAB}
    labels_by_zone = {z: [] for z in ZONE_VOCAB}
    for fname in filenames:
        df = _load_file(fname)
        zone_feats, row_idx = extract_zone_features(df)
        zone_labels_full = _zone_labels_for_file(fname, df["timestamp"])
        for zi, zone in enumerate(ZONE_VOCAB):
            feats_by_zone[zone].append(zone_feats[zone])
            labels_by_zone[zone].append(zone_labels_full[row_idx, zi])
    return (
        {z: np.concatenate(v, axis=0) for z, v in feats_by_zone.items()},
        {z: np.concatenate(v, axis=0) for z, v in labels_by_zone.items()},
    )


def fit_pca_models(normal_feats: dict[str, np.ndarray]) -> dict[str, tuple[StandardScaler, PCA]]:
    """Fits the per-zone StandardScaler+PCA pair used by both the standalone PCA baseline
    and the hybrid GNN's engineered feature (precompute_pca_features.py) -- one fit
    function so both consumers are guaranteed to use identical normal-only statistics."""
    models = {}
    for zone in ZONE_VOCAB:
        scaler = StandardScaler().fit(normal_feats[zone])
        normal_x = scaler.transform(normal_feats[zone])
        n_components = min(PCA_COMPONENTS, normal_x.shape[1] - 1, normal_x.shape[0] - 1)
        pca = PCA(n_components=max(n_components, 1), random_state=SEED).fit(normal_x)
        models[zone] = (scaler, pca)
    return models


def pca_reconstruction_error(models: dict, feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for zone, (scaler, pca) in models.items():
        x = scaler.transform(feats[zone])
        recon = pca.inverse_transform(pca.transform(x))
        out[zone] = np.square(x - recon).mean(axis=1).astype(np.float32)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting features: train1.csv (normal-only, for PCA/IsolationForest)...")
    normal_feats, _ = _features_and_labels(["train1.csv"])

    print("Extracting features: train pool (train1+test1+test2, for RandomForest)...")
    train_feats, train_labels = _features_and_labels(TRAIN_FILES)

    print("Extracting features: test4.csv (held-out)...")
    test_feats, test_labels = _features_and_labels(TEST_FILES)

    pca_models = fit_pca_models(normal_feats)
    pca_scores = pca_reconstruction_error(pca_models, test_feats)
    joblib.dump(pca_models, OUT_DIR / "pca_models.joblib")

    results = {"pca": {}, "isolation_forest": {}, "random_forest": {}, "labels": {}}
    for zone in ZONE_VOCAB:
        results["labels"][zone] = test_labels[zone]

        scaler, _ = pca_models[zone]
        normal_x = scaler.transform(normal_feats[zone])
        test_x = scaler.transform(test_feats[zone])
        results["pca"][zone] = pca_scores[zone]

        iso = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=-1).fit(normal_x)
        iso_score = -iso.score_samples(test_x)  # higher = more anomalous
        results["isolation_forest"][zone] = iso_score

        rf_x = scaler.transform(train_feats[zone])
        rf_y = train_labels[zone]
        if rf_y.sum() == 0:
            rf_score = np.zeros(len(test_x), dtype=np.float32)
        else:
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=12, class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            ).fit(rf_x, rf_y)
            rf_score = rf.predict_proba(test_x)[:, 1]
        results["random_forest"][zone] = rf_score

        print(f"{zone}: pca recon_range=[{pca_scores[zone].min():.3f},{pca_scores[zone].max():.3f}] "
              f"iso_range=[{iso_score.min():.3f},{iso_score.max():.3f}] "
              f"rf_positive_train={int(rf_y.sum())}")

    joblib.dump(results, OUT_DIR / "test_scores.joblib")
    print(f"\nWrote {OUT_DIR / 'test_scores.joblib'}")


if __name__ == "__main__":
    main()
