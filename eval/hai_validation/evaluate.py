"""
Final comparison on the held-out test4.csv, now scored PER ZONE (P1/P2/P3/P4) against
the real per-subsystem ground truth attack_labels.py recovers -- the properly-matched
version of this validation, superseding the first pass's pooled single-label comparison.

Six-way comparison:
  gnn                     -- HaiRiskGNN (GRU + GATv2 heterogeneous graph, train.py)
  no_graph_ablation       -- same GRU + zone identity, no message passing (train_no_graph.py)
  baseline_single_sensor  -- fixed per-zone 4-sigma z-score threshold (baseline.py)
  pca                     -- unsupervised PCA reconstruction error, fit on normal-only data
  isolation_forest        -- unsupervised Isolation Forest, fit on normal-only data
  random_forest           -- supervised Random Forest on the same window-summary features
  (pca/isolation_forest/random_forest: train_classical.py)
  hybrid                  -- HybridRiskGNN: full graph + PCA score fed into each zone's
                             own input feature (train_hybrid.py) -- graph's cross-zone
                             reasoning on top of PCA's per-zone anomaly signal

Reports, mirroring eval/metrics.py's structure:
  1. precision/recall/F1 at 0.5 threshold, flattened over all zone-slots
  2. FNR at matched false-positive-rate operating points (5/10/20%), threshold swept on
     each model's own continuous per-zone score
  3. zone localization: among windows with exactly one attacked zone, does the model's
     highest-scoring zone match the zone actually under attack
  4. ROC-AUC, flattened over all zone-slots

Run: `python -m eval.hai_validation.evaluate` (after train.py, train_no_graph.py, and
baseline.py have run).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from eval.hai_validation.baseline import TEST_FILES, fit_normal_stats, score_windows
from eval.hai_validation.graph_builder import ZONE_VOCAB
from eval.hai_validation.train import OUT_DIR as GNN_OUT_DIR
from eval.hai_validation.train_classical import OUT_DIR as CLASSICAL_OUT_DIR
from eval.hai_validation.train_hybrid import OUT_DIR as HYBRID_OUT_DIR
from eval.hai_validation.train_no_graph import OUT_DIR as ABLATION_OUT_DIR

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
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
    """Among windows with exactly ONE attacked zone, does each model's top-scoring zone
    match it. model_zone_scores: {name: [n_windows, N_ZONES] score array}."""
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

    ablation_data = torch.load(ABLATION_OUT_DIR / "test_predictions.pt")
    ablation_probs_flat = ablation_data["probs"].numpy()
    ablation_y_flat = ablation_data["y"].numpy().astype(int)

    hybrid_data = torch.load(HYBRID_OUT_DIR / "test_predictions.pt")
    hybrid_probs_flat = hybrid_data["probs"].numpy()
    hybrid_y_flat = hybrid_data["y"].numpy().astype(int)

    stats = fit_normal_stats()
    base_df = score_windows(TEST_FILES, stats)
    base_scores = np.stack([base_df[f"max_z__{z}"].to_numpy() for z in ZONE_VOCAB], axis=1)
    base_alert = np.stack([base_df[f"did_alert__{z}"].to_numpy() for z in ZONE_VOCAB], axis=1).astype(int)
    base_y = np.stack([base_df[f"label__{z}"].to_numpy() for z in ZONE_VOCAB], axis=1).astype(int)
    base_scores_flat = base_scores.reshape(-1)
    base_alert_flat = base_alert.reshape(-1)
    base_y_flat = base_y.reshape(-1)

    assert (gnn_y == base_y).all(), "GNN vs baseline per-zone label mismatch"
    assert (ablation_y_flat == y_flat).all(), "ablation vs GNN per-zone label mismatch"
    assert (hybrid_y_flat == y_flat).all(), "hybrid vs GNN per-zone label mismatch"

    classical = joblib.load(CLASSICAL_OUT_DIR / "test_scores.joblib")
    classical_y = np.stack([classical["labels"][z] for z in ZONE_VOCAB], axis=1).astype(int)
    assert classical_y.shape == gnn_y.shape, f"classical label shape mismatch: {classical_y.shape} vs {gnn_y.shape}"
    assert (classical_y == gnn_y).all(), "classical baselines vs GNN per-zone label mismatch"

    pca_flat = np.stack([classical["pca"][z] for z in ZONE_VOCAB], axis=1).reshape(-1)
    iso_flat = np.stack([classical["isolation_forest"][z] for z in ZONE_VOCAB], axis=1).reshape(-1)
    rf_flat = np.stack([classical["random_forest"][z] for z in ZONE_VOCAB], axis=1).reshape(-1)

    scores_flat = {
        "gnn": gnn_probs_flat,
        "no_graph_ablation": ablation_probs_flat,
        "baseline_single_sensor": base_scores_flat,
        "pca": pca_flat,
        "isolation_forest": iso_flat,
        "random_forest": rf_flat,
        "hybrid": hybrid_probs_flat,
    }
    preds_flat = {
        "gnn": (gnn_probs_flat > 0.5).astype(int),
        "no_graph_ablation": (ablation_probs_flat > 0.5).astype(int),
        "baseline_single_sensor": base_alert_flat,
        "pca": None,  # unsupervised, continuous scores only -- no natural 0/1 cutoff
        "isolation_forest": None,
        "random_forest": (rf_flat > 0.5).astype(int),
        "hybrid": (hybrid_probs_flat > 0.5).astype(int),
    }

    target_fprs = [0.05, 0.10, 0.20]
    accuracy, fnr, auc = {}, {}, {}
    for name in scores_flat:
        if preds_flat[name] is not None:
            p, r, f1 = precision_recall_f1(y_flat, preds_flat[name])
            accuracy[name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
        else:
            accuracy[name] = "no natural 0.5-style cutoff (unsupervised continuous score) -- see fnr_at_matched_fpr/roc_auc instead"
        fnr[name] = fnr_at_matched_fpr(scores_flat[name], y_flat, target_fprs)
        auc[name] = round(roc_auc(scores_flat[name], y_flat), 4)

    zone_scores = {
        "gnn": gnn_probs_flat.reshape(n_windows, N_ZONES),
        "no_graph_ablation": ablation_probs_flat.reshape(n_windows, N_ZONES),
        "baseline_single_sensor": base_scores,
        "pca": pca_flat.reshape(n_windows, N_ZONES),
        "isolation_forest": iso_flat.reshape(n_windows, N_ZONES),
        "random_forest": rf_flat.reshape(n_windows, N_ZONES),
        "hybrid": hybrid_probs_flat.reshape(n_windows, N_ZONES),
    }

    results = {
        "n_test_windows": int(n_windows),
        "n_zone_slots": int(n_windows * N_ZONES),
        "n_attack_positive_slots": int(y_flat.sum()),
        "per_zone_prevalence": {z: round(float(gnn_y[:, i].mean()), 4) for i, z in enumerate(ZONE_VOCAB)},
        "accuracy_at_default_threshold": {
            "note": "flattened over all zone-slots (n_windows x 4 zones); gnn/no_graph_ablation "
                     "at sigmoid>0.5, baseline at native fixed 4-sigma per-zone alert",
            **accuracy,
        },
        "fnr_at_matched_fpr": {
            "note": "threshold swept on each model's own continuous per-zone score, flattened over zone-slots",
            **fnr,
        },
        "zone_localization": zone_localization(zone_scores, gnn_y),
        "roc_auc": auc,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
