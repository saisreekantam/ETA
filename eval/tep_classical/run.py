"""
Classical baselines on the SAME TEP synthetic benchmark and SAME run-level train/val/test
split models/gnn/train.py uses (seed=0, stratified by scenario_id+condition, 70/15/15) --
so these numbers are directly comparable to the paper's reported Table 5/6 (GNN: P=1.00
R=1.00 F1=1.00; naive z-score baseline: P=0.37 R=1.00 F1=0.54).

Models, fit PER ZONE (zones have different channel counts, so no shared-width pooling):
  pca               -- unsupervised, sensor-only, fit on normal-condition runs only
  isolation_forest  -- unsupervised, sensor-only, fit on normal-condition runs only
  random_forest_sensor_only  -- supervised, same info as the z-score baseline (sensors only)
  random_forest_full_context -- supervised, sensor + permit + presence flattened together
                      -- same information the GNN gets, no graph/message-passing at all.
                      This is the key comparison: does the graph's relational structure
                      beat a flat classifier given identical inputs?

All scored at each run's TRUE hazard zone, matching eval/metrics.py's headline comparison.

Run: `python -m eval.tep_classical.run`
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eval.tep_classical.features import SCORED_ZONES, extract_run_features

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 0
PCA_COMPONENTS = 5
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def load_all_features():
    manifest = pd.read_csv(REPO_ROOT / "data" / "synthetic" / "manifest.csv")
    permits = pd.read_parquet(REPO_ROOT / "data" / "permits" / "permits.parquet").set_index("run_id")
    presences = pd.read_parquet(REPO_ROOT / "data" / "shiftlogs" / "shiftlogs.parquet").set_index("run_id")

    sensor_only = {z: [] for z in SCORED_ZONES}
    full_context = {z: [] for z in SCORED_ZONES}
    zone_labels = {z: [] for z in SCORED_ZONES}
    for _, row in manifest.iterrows():
        sensor_df = pd.read_parquet(REPO_ROOT / row["path"])
        permit, presence = permits.loc[row["run_id"]], presences.loc[row["run_id"]]
        feats = extract_run_features(sensor_df, permit, presence, row["zone"])
        for zone in SCORED_ZONES:
            so, fc = feats[zone]
            sensor_only[zone].append(so)
            full_context[zone].append(fc)
            zone_labels[zone].append(1.0 if (bool(row["compound_active"]) and zone == row["zone"]) else 0.0)

    sensor_only = {z: np.stack(v) for z, v in sensor_only.items()}
    full_context = {z: np.stack(v) for z, v in full_context.items()}
    zone_labels = {z: np.array(v, dtype=np.float32) for z, v in zone_labels.items()}
    return manifest, sensor_only, full_context, zone_labels


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


def main():
    print("Extracting features for all 800 runs...")
    manifest, sensor_only, full_context, zone_labels = load_all_features()
    strata = manifest["scenario_id"] + "_" + manifest["condition"]

    idx = np.arange(len(manifest))
    train_idx, rest_idx = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=strata)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=SEED, stratify=strata.iloc[rest_idx])
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    normal_mask = (manifest["condition"] == "normal").to_numpy()
    normal_train_idx = np.intersect1d(train_idx, np.where(normal_mask)[0])
    print(f"normal-condition train runs (for PCA/IsolationForest fit): {len(normal_train_idx)}")

    test_manifest = manifest.iloc[test_idx].reset_index(drop=True)
    n_test = len(test_idx)
    true_zone = test_manifest["zone"].tolist()
    compound_active = test_manifest["compound_active"].to_numpy()

    # per-zone model fit + per-zone test scores
    pca_scores, iso_scores, rf_so_scores, rf_fc_scores = {}, {}, {}, {}
    for zone in SCORED_ZONES:
        so, fc, y = sensor_only[zone], full_context[zone], zone_labels[zone]

        scaler = StandardScaler().fit(so[normal_train_idx])
        normal_scaled = scaler.transform(so[normal_train_idx])
        n_components = min(PCA_COMPONENTS, normal_scaled.shape[1] - 1, normal_scaled.shape[0] - 1)
        pca = PCA(n_components=max(n_components, 1), random_state=SEED).fit(normal_scaled)
        iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(normal_scaled)

        test_scaled = scaler.transform(so[test_idx])
        recon = pca.inverse_transform(pca.transform(test_scaled))
        pca_scores[zone] = np.square(test_scaled - recon).mean(axis=1)
        iso_scores[zone] = -iso.score_samples(test_scaled)

        if y[train_idx].sum() == 0:
            # this zone is never the injected hazard anywhere in the manifest (e.g.
            # separator/stripper/compressor_zone) -- no positive class to learn from,
            # same convention as models/gnn/baseline_threshold.py's control_room handling
            rf_so_scores[zone] = np.zeros(len(test_idx), dtype=np.float32)
            rf_fc_scores[zone] = np.zeros(len(test_idx), dtype=np.float32)
        else:
            rf_so = RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced", random_state=SEED, n_jobs=-1,
            ).fit(so[train_idx], y[train_idx])
            rf_so_scores[zone] = rf_so.predict_proba(so[test_idx])[:, 1]

            rf_fc = RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced", random_state=SEED, n_jobs=-1,
            ).fit(fc[train_idx], y[train_idx])
            rf_fc_scores[zone] = rf_fc.predict_proba(fc[test_idx])[:, 1]

    def at_true_zone(score_dict):
        return np.array([score_dict[z][i] for i, z in enumerate(true_zone)])

    def full_grid(score_dict):
        return np.stack([score_dict[z] for z in SCORED_ZONES], axis=1)  # [n_test, n_zones]

    scores_at_true_zone = {
        "pca": at_true_zone(pca_scores),
        "isolation_forest": at_true_zone(iso_scores),
        "random_forest_sensor_only": at_true_zone(rf_so_scores),
        "random_forest_full_context": at_true_zone(rf_fc_scores),
    }
    zone_grids = {
        "pca": full_grid(pca_scores),
        "isolation_forest": full_grid(iso_scores),
        "random_forest_sensor_only": full_grid(rf_so_scores),
        "random_forest_full_context": full_grid(rf_fc_scores),
    }

    y_true = np.array([zone_labels[z][test_idx[i]] for i, z in enumerate(true_zone)]).astype(int)

    target_fprs = [0.05, 0.10, 0.20]
    accuracy, fnr, auc, zone_top_acc = {}, {}, {}, {}
    for name, scores in scores_at_true_zone.items():
        auc[name] = round(roc_auc(scores, y_true), 4)
        fnr[name] = fnr_at_matched_fpr(scores, y_true, target_fprs)
        if name in ("pca", "isolation_forest"):
            accuracy[name] = "unsupervised continuous score, no natural 0.5-style cutoff"
        else:
            pred = (scores > 0.5).astype(int)
            p, r, f1 = precision_recall_f1(y_true, pred)
            accuracy[name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}

        grid = zone_grids[name][compound_active]
        top = grid.argmax(axis=1)
        true_top = np.array([SCORED_ZONES.index(z) for z in np.array(true_zone)[compound_active]])
        zone_top_acc[name] = round(float((top == true_top).mean()), 3) if compound_active.sum() > 0 else None

    results = {
        "n_test_runs": int(n_test),
        "n_compound_positive": int(y_true.sum()),
        "note": "scored at each run's true hazard zone, matching eval/metrics.py's compute_accuracy_comparison; "
                 "reference numbers from the paper: GNN P=1.00 R=1.00 F1=1.00, "
                 "naive z-score baseline P=0.37 R=1.00 F1=0.54, GNN zone-localization=100%, baseline=23.3%",
        "accuracy_at_default_threshold": accuracy,
        "fnr_at_matched_fpr": fnr,
        "zone_localization_top_accuracy": zone_top_acc,
        "n_compound_test_runs_for_localization": int(compound_active.sum()),
        "roc_auc": auc,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
