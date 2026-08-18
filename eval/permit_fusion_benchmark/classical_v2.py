"""
Classical baselines for the permit-fusion benchmark, same run-level train/val/test split
as train_v2.py. Feature extraction mirrors eval/tep_classical/features.py's per-zone
[mean, std, delta, min, max] over the scored WINDOW.

  z_score                    -- naive per-channel 4-sigma threshold (same as
                                 models/gnn/baseline_threshold.py), sensor-only
  pca / isolation_forest      -- unsupervised, sensor-only, fit on normal-condition runs
  random_forest_sensor_only   -- supervised, sensor-only -- should now be near-chance at
                                 separating true_positive from the fault-having negative
                                 controls, since their sensor windows are statistically
                                 identical by construction (see generate.py)
  random_forest_full_context  -- supervised, sensor + WINDOW-RELATIVE permit/presence
                                 features (graph_builder_v2.py's overlap-aware semantics)
                                 -- should succeed, same information the GNN has, no graph

Run: `python -m eval.permit_fusion_benchmark.classical_v2` (after train_v2.py has run,
for the shared test_idx / train_idx split).
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

from eval.permit_fusion_benchmark.graph_builder_v2 import DATA_DIR, N_SAMPLES, _overlaps_window
from models.gnn.graph_builder import CLUSTER_TO_ZONE, PERMIT_TYPE_VOCAB, SENSOR_CLUSTERS, WINDOW

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 0
PCA_COMPONENTS = 5
THRESHOLD_SIGMA = 4.0
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

SCORED_ZONES = list(CLUSTER_TO_ZONE.values())
ZONE_TO_CLUSTER = {z: c for c, z in CLUSTER_TO_ZONE.items()}


def _sensor_stats(window_df: pd.DataFrame, zone: str) -> np.ndarray:
    cols = SENSOR_CLUSTERS[ZONE_TO_CLUSTER[zone]]
    feats = []
    for col in cols:
        v = window_df[col].to_numpy(dtype=np.float32)
        feats.extend([v.mean(), v.std(), v[-1] - v[0], v.min(), v.max()])
    return np.array(feats, dtype=np.float32)


def _permit_ctx_feat(permit: pd.Series, presence: pd.Series) -> np.ndarray:
    window_start, window_end = N_SAMPLES - WINDOW, N_SAMPLES - 1
    permit_active = bool(permit["has_permit"]) and _overlaps_window(
        permit.get("from_sample"), permit.get("to_sample"), window_start, window_end)
    presence_active = bool(presence["has_presence"]) and _overlaps_window(
        presence.get("from_sample"), presence.get("to_sample"), window_start, window_end)
    onehot = [1.0 if (permit_active and permit["permit_type"] == p) else 0.0 for p in PERMIT_TYPE_VOCAB]
    return np.array([float(permit_active), *onehot, float(presence_active)], dtype=np.float32)


def load_all_features():
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    permits = pd.read_parquet(DATA_DIR / "permits.parquet").set_index("run_id")
    presences = pd.read_parquet(DATA_DIR / "presences.parquet").set_index("run_id")

    sensor_only = {z: [] for z in SCORED_ZONES}
    full_context = {z: [] for z in SCORED_ZONES}
    zone_labels = {z: [] for z in SCORED_ZONES}
    raw_sensor_full = {z: [] for z in SCORED_ZONES}  # for z-score baseline (full-run, all channels)
    for _, row in manifest.iterrows():
        sensor_df = pd.read_parquet(REPO_ROOT / row["path"])
        window_df = sensor_df.iloc[-WINDOW:]
        permit, presence = permits.loc[row["run_id"]], presences.loc[row["run_id"]]
        ctx = _permit_ctx_feat(permit, presence)
        for zone in SCORED_ZONES:
            so = _sensor_stats(window_df, zone)
            sensor_only[zone].append(so)
            full_context[zone].append(np.concatenate([so, ctx]))
            zone_labels[zone].append(1.0 if (bool(row["true_positive"]) and zone == row["zone"]) else 0.0)
            zmax = window_df[SENSOR_CLUSTERS[ZONE_TO_CLUSTER[zone]]].to_numpy(dtype=np.float32)
            raw_sensor_full[zone].append(zmax)

    sensor_only = {z: np.stack(v) for z, v in sensor_only.items()}
    full_context = {z: np.stack(v) for z, v in full_context.items()}
    zone_labels = {z: np.array(v, dtype=np.float32) for z, v in zone_labels.items()}
    return manifest, sensor_only, full_context, zone_labels, raw_sensor_full


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
    print("Extracting features for all 1000 runs...")
    manifest, sensor_only, full_context, zone_labels, raw_sensor_full = load_all_features()
    strata = manifest["scenario_id"] + "_" + manifest["condition"]

    idx = np.arange(len(manifest))
    train_idx, rest_idx = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=strata)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=SEED, stratify=strata.iloc[rest_idx])
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    normal_mask = (manifest["condition"] == "normal").to_numpy()
    normal_train_idx = np.intersect1d(train_idx, np.where(normal_mask)[0])

    test_manifest = manifest.iloc[test_idx].reset_index(drop=True)
    true_zone = test_manifest["zone"].tolist()
    is_true_positive = test_manifest["true_positive"].to_numpy()

    pca_scores, iso_scores, zscore_scores, rf_so_scores, rf_fc_scores = {}, {}, {}, {}, {}
    for zone in SCORED_ZONES:
        so, fc, y = sensor_only[zone], full_context[zone], zone_labels[zone]
        raw = raw_sensor_full[zone]

        scaler = StandardScaler().fit(so[normal_train_idx])
        normal_scaled = scaler.transform(so[normal_train_idx])
        n_components = min(PCA_COMPONENTS, normal_scaled.shape[1] - 1, normal_scaled.shape[0] - 1)
        pca = PCA(n_components=max(n_components, 1), random_state=SEED).fit(normal_scaled)
        iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(normal_scaled)
        test_scaled = scaler.transform(so[test_idx])
        recon = pca.inverse_transform(pca.transform(test_scaled))
        pca_scores[zone] = np.square(test_scaled - recon).mean(axis=1)
        iso_scores[zone] = -iso.score_samples(test_scaled)

        raw_normal = np.concatenate([raw[i] for i in normal_train_idx], axis=0)
        ch_mean, ch_std = raw_normal.mean(axis=0), raw_normal.std(axis=0)
        ch_std[ch_std == 0] = 1e-6
        zs = []
        for i in test_idx:
            z = np.abs((raw[i] - ch_mean) / ch_std)
            zs.append(float(z.max()))
        zscore_scores[zone] = np.array(zs, dtype=np.float32)

        if y[train_idx].sum() == 0:
            rf_so_scores[zone] = np.zeros(len(test_idx), dtype=np.float32)
            rf_fc_scores[zone] = np.zeros(len(test_idx), dtype=np.float32)
        else:
            rf_so = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1).fit(so[train_idx], y[train_idx])
            rf_so_scores[zone] = rf_so.predict_proba(so[test_idx])[:, 1]
            rf_fc = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1).fit(fc[train_idx], y[train_idx])
            rf_fc_scores[zone] = rf_fc.predict_proba(fc[test_idx])[:, 1]

    def at_true_zone(score_dict):
        return np.array([score_dict[z][i] for i, z in enumerate(true_zone)])

    def full_grid(score_dict):
        return np.stack([score_dict[z] for z in SCORED_ZONES], axis=1)

    scores_at_true_zone = {
        "z_score": at_true_zone(zscore_scores),
        "pca": at_true_zone(pca_scores),
        "isolation_forest": at_true_zone(iso_scores),
        "random_forest_sensor_only": at_true_zone(rf_so_scores),
        "random_forest_full_context": at_true_zone(rf_fc_scores),
    }
    zone_grids = {
        "z_score": full_grid(zscore_scores), "pca": full_grid(pca_scores),
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
        elif name == "z_score":
            pred = (scores > THRESHOLD_SIGMA).astype(int)
            p, r, f1 = precision_recall_f1(y_true, pred)
            accuracy[name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
        else:
            pred = (scores > 0.5).astype(int)
            p, r, f1 = precision_recall_f1(y_true, pred)
            accuracy[name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}

        grid = zone_grids[name][is_true_positive]
        top = grid.argmax(axis=1)
        true_top = np.array([SCORED_ZONES.index(z) for z in np.array(true_zone)[is_true_positive]])
        zone_top_acc[name] = round(float((top == true_top).mean()), 3) if is_true_positive.sum() > 0 else None

    # diagnostic: can sensor-only tell "any fault" apart from "no fault", vs "true_positive
    # specifically" among the fault-having conditions -- confirms the intended structure
    fault_conditions = {"true_positive", "fp_fault_no_permit", "fp_fault_permit_no_overlap"}
    has_fault = test_manifest["condition"].isin(fault_conditions).to_numpy()
    rf_so_any_fault_auc = roc_auc(at_true_zone(rf_so_scores), has_fault.astype(int))
    among_fault = has_fault
    rf_so_among_fault_auc = roc_auc(
        at_true_zone(rf_so_scores)[among_fault], is_true_positive[among_fault].astype(int)
    ) if among_fault.sum() > 0 else None

    results = {
        "n_test_runs": int(len(test_idx)),
        "n_true_positive": int(y_true.sum()),
        "note": "scored at each run's true hazard zone; z_score/pca/isolation_forest/random_forest_sensor_only "
                 "see ONLY sensor window stats (same info as z_score baseline); random_forest_full_context also "
                 "sees window-relative permit/presence overlap (same info as the GNN)",
        "diagnostic_sensor_only_rf": {
            "auc_any_fault_vs_no_fault": round(rf_so_any_fault_auc, 4),
            "auc_true_positive_vs_other_fault_conditions": round(rf_so_among_fault_auc, 4) if rf_so_among_fault_auc is not None else None,
            "interpretation": "high first number + near-0.5 second number confirms sensor-only CAN detect "
                                "'a fault exists' but CANNOT distinguish true_positive from the other two "
                                "fault-having conditions, as designed",
        },
        "accuracy_at_default_threshold": accuracy,
        "fnr_at_matched_fpr": fnr,
        "zone_localization_top_accuracy": zone_top_acc,
        "n_true_positive_test_runs_for_localization": int(is_true_positive.sum()),
        "roc_auc": auc,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
