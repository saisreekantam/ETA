"""
GNN vs z-score baseline vs PCA on HAI 23.05's held-out test split (hai23-test2b.csv,
Aug 19 2022, 10 attacks). z-score/PCA are fit on hai23-train1.csv (normal-only).

Run: `python -m eval.hai_validation.v2305.evaluate_2305` (after train_2305.py has run).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from eval.hai_validation.v2305.classical_2305 import (
    extract_zone_features, fit_pca_models, fit_zscore_stats, pca_reconstruction_error, score_zscore,
)
from eval.hai_validation.v2305.common import TEST_FILES, ZONE_VOCAB, load_file
from eval.hai_validation.v2305.dataset_2305 import _zone_labels_for_file
from eval.hai_validation.v2305.train_2305 import OUT_DIR as GNN_OUT_DIR

RESULTS_PATH = Path(__file__).resolve().parent / "results_2305.json"
N_ZONES = len(ZONE_VOCAB)


def precision_recall_f1(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def fnr_at_matched_fpr(scores, y_true, target_fprs):
    neg_scores = np.sort(scores[y_true == 0])
    rows = []
    for target_fpr in target_fprs:
        if len(neg_scores) == 0:
            continue
        k = int(round((1 - target_fpr) * (len(neg_scores) - 1)))
        threshold = neg_scores[k]
        y_pred = (scores >= threshold).astype(int)
        achieved_fpr = ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fnr = fn / max((y_true == 1).sum(), 1)
        rows.append({"target_fpr": target_fpr, "achieved_fpr": round(float(achieved_fpr), 3),
                      "fnr": round(float(fnr), 3), "threshold": round(float(threshold), 4)})
    return rows


def roc_auc(scores, y_true):
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    return float(np.trapezoid(tpr, fpr))


def zone_localization(model_zone_scores: dict, zone_label_mat):
    single_zone_mask = zone_label_mat.sum(axis=1) == 1
    true_zone_idx = zone_label_mat[single_zone_mask].argmax(axis=1)
    out = {"n_single_zone_windows": int(single_zone_mask.sum())}
    for name, scores in model_zone_scores.items():
        top = scores[single_zone_mask].argmax(axis=1)
        out[f"{name}_top_zone_accuracy"] = round(float((top == true_zone_idx).mean()), 3)
    return out


def main():
    gnn_data = torch.load(GNN_OUT_DIR / "test_predictions.pt")
    gnn_probs_flat = gnn_data["probs"].numpy()
    y_flat = gnn_data["y"].numpy().astype(int)
    n_windows = len(gnn_probs_flat) // N_ZONES
    gnn_y = y_flat.reshape(n_windows, N_ZONES)

    normal_df = load_file("hai23-train1.csv")
    test_df = load_file(TEST_FILES[0])
    test_zone_labels = _zone_labels_for_file(TEST_FILES[0], test_df["timestamp"])

    zscore_stats = fit_zscore_stats(normal_df)
    base_df = score_zscore(test_df, test_zone_labels, zscore_stats)
    base_scores = np.stack([base_df[f"max_z__{z}"].to_numpy() for z in ZONE_VOCAB], axis=1)
    base_alert = np.stack([base_df[f"did_alert__{z}"].to_numpy() for z in ZONE_VOCAB], axis=1).astype(int)
    base_y = np.stack([base_df[f"label__{z}"].to_numpy() for z in ZONE_VOCAB], axis=1).astype(int)

    normal_feats, _ = extract_zone_features(normal_df)
    test_feats, _ = extract_zone_features(test_df)
    pca_models = fit_pca_models(normal_feats)
    pca_scores_by_zone = pca_reconstruction_error(pca_models, test_feats)
    pca_scores = np.stack([pca_scores_by_zone[z] for z in ZONE_VOCAB], axis=1)

    assert gnn_y.shape == base_y.shape == pca_scores.shape, \
        f"shape mismatch: gnn={gnn_y.shape} base={base_y.shape} pca={pca_scores.shape}"
    assert (gnn_y == base_y).all(), "GNN vs baseline per-zone label mismatch"

    scores_flat = {
        "gnn": gnn_probs_flat,
        "baseline_single_sensor": base_scores.reshape(-1),
        "pca": pca_scores.reshape(-1),
    }
    preds_flat = {
        "gnn": (gnn_probs_flat > 0.5).astype(int),
        "baseline_single_sensor": base_alert.reshape(-1),
        "pca": None,
    }

    target_fprs = [0.05, 0.10, 0.20]
    accuracy, fnr, auc = {}, {}, {}
    for name in scores_flat:
        if preds_flat[name] is not None:
            p, r, f1 = precision_recall_f1(y_flat, preds_flat[name])
            accuracy[name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
        else:
            accuracy[name] = "no natural 0.5-style cutoff (unsupervised continuous score)"
        fnr[name] = fnr_at_matched_fpr(scores_flat[name], y_flat, target_fprs)
        auc[name] = round(roc_auc(scores_flat[name], y_flat), 4)

    zone_scores = {
        "gnn": gnn_probs_flat.reshape(n_windows, N_ZONES),
        "baseline_single_sensor": base_scores,
        "pca": pca_scores,
    }

    results = {
        "n_test_windows": int(n_windows),
        "n_zone_slots": int(n_windows * N_ZONES),
        "n_attack_positive_slots": int(y_flat.sum()),
        "per_zone_prevalence": {z: round(float(gnn_y[:, i].mean()), 4) for i, z in enumerate(ZONE_VOCAB)},
        "note_p2_p3_train_support": "P2 has 0 and P3 has 0 positive training windows in this split's "
                                     "train pool (see dataset_2305.py output) -- per-zone results for "
                                     "P2/P3 test zero-shot generalization, not learned detection",
        "accuracy_at_default_threshold": accuracy,
        "fnr_at_matched_fpr": fnr,
        "zone_localization": zone_localization(zone_scores, gnn_y),
        "roc_auc": auc,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
