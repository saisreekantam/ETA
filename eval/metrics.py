"""
Computes the five judged metrics, comparing the multi-zone GNN against the zone-aware
naive single-sensor baseline on the SAME held-out test split (run-level, no timestep
leakage by construction -- see models/gnn/train.py's stratified run split).

1. Compound detection accuracy vs. single-sensor baseline (precision/recall/F1), scored
   at each run's true hazard zone.
2. Prediction lead time (streaming re-inference at increasing cutoffs, compared to the
   baseline's first-alert-sample at the same zone, both relative to ground_truth_onset_sample)
3. False negative rate at matched false-positive-rate operating points
4. Zone-localization accuracy: does the model's TOP-scoring zone match the zone the
   fault was actually injected into -- now a real, checkable number, not a placeholder.
5. RAG citation-rate is computed separately by the orchestrator agent, not here.

Run directly: `python -m eval.metrics` (requires gnn_predictions.parquet and
baseline_predictions.parquet to already exist).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from models.gnn.graph_builder import ZONE_VOCAB, build_graph, load_all_graphs
from models.gnn.model import CompoundRiskGNN

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def fnr_at_matched_fpr(scores: np.ndarray, y_true: np.ndarray, target_fprs: list[float]) -> list[dict]:
    """Sweeps the decision threshold to hit each target FPR, reports FNR there."""
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


def _baseline_true_zone_alert(baseline_preds: pd.DataFrame, run_ids: pd.Series, zones: pd.Series) -> np.ndarray:
    base = baseline_preds.set_index("run_id").loc[run_ids]
    return np.array([bool(base.loc[rid, f"did_alert__{z}"]) for rid, z in zip(run_ids, zones)])


def _baseline_true_zone_score(baseline_preds: pd.DataFrame, run_ids: pd.Series, zones: pd.Series) -> np.ndarray:
    base = baseline_preds.set_index("run_id").loc[run_ids]
    return np.array([float(base.loc[rid, f"max_abs_zscore__{z}"]) for rid, z in zip(run_ids, zones)])


def compute_accuracy_comparison(gnn_preds: pd.DataFrame, baseline_preds: pd.DataFrame) -> dict:
    test = gnn_preds[gnn_preds["split"] == "test"]
    y_true = test["compound_active"].astype(int).to_numpy()

    gnn_pred = (test["gnn_prob_true_zone"] > 0.5).astype(int).to_numpy()
    base_pred = _baseline_true_zone_alert(baseline_preds, test["run_id"], test["zone"]).astype(int)

    gnn_p, gnn_r, gnn_f1 = precision_recall_f1(y_true, gnn_pred)
    base_p, base_r, base_f1 = precision_recall_f1(y_true, base_pred)

    return {
        "n_test_runs": int(len(test)),
        "n_compound_positive": int(y_true.sum()),
        "note": "scored at each run's true hazard zone (run-level, not pooled across all 7 zones)",
        "gnn": {"precision": round(gnn_p, 3), "recall": round(gnn_r, 3), "f1": round(gnn_f1, 3)},
        "baseline_single_sensor": {"precision": round(base_p, 3), "recall": round(base_r, 3), "f1": round(base_f1, 3)},
    }


def compute_fnr_table(gnn_preds: pd.DataFrame, baseline_preds: pd.DataFrame) -> dict:
    test = gnn_preds[gnn_preds["split"] == "test"]
    y_true = test["compound_active"].astype(int).to_numpy()

    gnn_scores = test["gnn_prob_true_zone"].to_numpy()
    base_scores = _baseline_true_zone_score(baseline_preds, test["run_id"], test["zone"])

    target_fprs = [0.05, 0.10, 0.20]
    return {
        "gnn": fnr_at_matched_fpr(gnn_scores, y_true, target_fprs),
        "baseline_single_sensor": fnr_at_matched_fpr(base_scores, y_true, target_fprs),
    }


def compute_zone_localization(gnn_preds: pd.DataFrame, baseline_preds: pd.DataFrame) -> dict:
    """Real metric now: among test compound runs, does the model's TOP-scoring zone
    (out of all 7) match the zone the fault was actually injected into? Compared against
    the baseline's top_zone (whichever zone's sensors show the largest z-score deviation,
    with no permit/cross-zone context to disambiguate)."""
    test = gnn_preds[(gnn_preds["split"] == "test") & (gnn_preds["compound_active"])]
    base = baseline_preds.set_index("run_id").loc[test["run_id"]]

    gnn_correct = (test["gnn_top_zone"].to_numpy() == test["zone"].to_numpy())
    base_correct = (base["top_zone"].to_numpy() == test["zone"].to_numpy())

    return {
        "n_compound_test_runs": int(len(test)),
        "gnn_top_zone_accuracy": round(float(gnn_correct.mean()), 3),
        "baseline_top_zone_accuracy": round(float(base_correct.mean()), 3),
        "note": "fraction of compound test runs where the model's highest-scoring zone "
                 "(of all 7) matches the zone the dual-fault was actually injected into",
    }


def compute_lead_time(gnn_preds: pd.DataFrame, baseline_preds: pd.DataFrame, n_runs: int = 30) -> dict:
    """Streaming re-inference: re-run the GNN at increasing cutoffs for a sample of
    test-split compound runs, find the first cutoff where the TRUE ZONE's prob>0.5,
    compare detection latency (alert_sample - ground_truth_onset_sample) to the
    baseline's first-alert-sample at that same zone. Negative = anticipatory."""
    permits = pd.read_parquet(REPO_ROOT / "data" / "permits" / "permits.parquet").set_index("run_id")
    presences = pd.read_parquet(REPO_ROOT / "data" / "shiftlogs" / "shiftlogs.parquet").set_index("run_id")
    manifest = pd.read_csv(REPO_ROOT / "data" / "synthetic" / "manifest.csv").set_index("run_id")

    model = CompoundRiskGNN()
    model.load_state_dict(torch.load(REPO_ROOT / "models" / "gnn" / "checkpoint.pt"))
    model.eval()
    # MUST apply the same train-set normalization stats used during training -- raw
    # sensor scales (~0-4500) saturate the model otherwise (verified empirically: this
    # was a real bug, raw features gave logits of ~-220 vs the correct, normalized ~+4).
    norm_stats = torch.load(REPO_ROOT / "models" / "gnn" / "norm_stats.pt")

    def normalize(graph):
        for ntype, (mean, std) in norm_stats.items():
            graph[ntype].x = (graph[ntype].x - mean) / std
        return graph

    test_compound = gnn_preds[(gnn_preds["split"] == "test") & (gnn_preds["compound_active"])]
    sample_runs = test_compound["run_id"].head(n_runs).tolist()

    # step=1 (matching the baseline's per-sample resolution) starting just below the
    # earliest possible fault onset -- a coarser grid (e.g. step=5) under-resolves runs
    # whose onset falls between grid points and makes the GNN look artificially slower
    # than it is; verified this was inflating median latency before the fix.
    cutoffs = list(range(31, 120, 1))
    gnn_latencies, base_latencies = [], []

    for run_id in sample_runs:
        row = manifest.loc[run_id]
        onset = row["ground_truth_onset_sample"]
        zone_idx = ZONE_VOCAB.index(row["zone"])
        sensor_df = pd.read_parquet(REPO_ROOT / row["path"])
        permit, presence, zone = permits.loc[run_id], presences.loc[run_id], row["zone"]
        # The synthetic generator schedules the permit from onset-5 and worker presence
        # from onset (see simulator/permit_shiftlog_synth.py's compound-positive branch).
        # Streaming re-inference MUST gate on this -- passing the full-run permit/presence
        # record at every cutoff would let the model "see" a permit before it was actually
        # filed, inflating lead time with information that wasn't really available yet.
        # Verified this was happening: many runs were alerting at the very first cutoff
        # regardless of onset, which is the signature of a future-leak, not real precursor
        # signal.
        permit_from_sample = max(0, onset - 5)

        gnn_alert_sample = None
        for cutoff in cutoffs:
            gated_permit = permit.copy()
            gated_permit["has_permit"] = bool(permit["has_permit"]) and cutoff >= permit_from_sample
            gated_presence = presence.copy()
            gated_presence["has_presence"] = bool(presence["has_presence"]) and cutoff >= onset

            graph = normalize(build_graph(sensor_df.iloc[:cutoff], gated_permit, gated_presence, zone))
            with torch.no_grad():
                batch = next(iter(DataLoader([graph], batch_size=1)))
                probs = torch.sigmoid(model(batch))
            if probs[zone_idx].item() > 0.5:
                gnn_alert_sample = cutoff
                break
        if gnn_alert_sample is not None:
            gnn_latencies.append(gnn_alert_sample - onset)

        base_row = baseline_preds.set_index("run_id").loc[run_id]
        if base_row[f"did_alert__{zone}"]:
            base_latencies.append(base_row[f"alert_sample__{zone}"] - onset)

    def summarize(latencies):
        if not latencies:
            return {"n": 0, "median_samples": None, "p10_samples": None}
        arr = np.array(latencies)
        return {"n": len(arr), "median_samples": float(np.median(arr)), "p10_samples": float(np.percentile(arr, 10))}

    minutes_per_sample = 3
    gnn_summary = summarize(gnn_latencies)
    base_summary = summarize(base_latencies)
    for s in (gnn_summary, base_summary):
        if s["median_samples"] is not None:
            s["median_minutes"] = s["median_samples"] * minutes_per_sample
            s["p10_minutes"] = s["p10_samples"] * minutes_per_sample

    return {"n_runs_evaluated": len(sample_runs), "gnn": gnn_summary, "baseline_single_sensor": base_summary,
            "note": "negative = detected before/at ground-truth onset (anticipatory); "
                     "positive = detected N samples (3 min each) after onset; both measured "
                     "at the run's true hazard zone. NOT a headline metric: with permit/presence "
                     "visibility properly time-gated (no future-leak), the GNN does not detect "
                     "earlier than the baseline on this benchmark -- training data was not "
                     "time-gated, so the model learned 'permit visible => compound' and "
                     "correctly waits for the permit to actually appear before committing. "
                     "The model's real advantages are precision/zone-localization, not lead time "
                     "-- see accuracy_vs_baseline and zone_localization above."}


def main():
    gnn_preds = pd.read_parquet(REPO_ROOT / "data" / "gnn_predictions.parquet")
    baseline_preds = pd.read_parquet(REPO_ROOT / "data" / "baseline_predictions.parquet")

    results = {
        "accuracy_vs_baseline": compute_accuracy_comparison(gnn_preds, baseline_preds),
        "fnr_at_matched_fpr": compute_fnr_table(gnn_preds, baseline_preds),
        "zone_localization": compute_zone_localization(gnn_preds, baseline_preds),
        "lead_time": compute_lead_time(gnn_preds, baseline_preds),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
