"""
The rigorous test for "is the graph structurally necessary": leave-one-SCENARIO-out
generalization. s4_feed_system is the only scenario whose true_positive examples live in
feed_zone -- so if we remove ALL of s4 (positive AND negative-control runs) from
train/val entirely, a model has literally never seen a single feed_zone example, positive
or negative, during training. At test time it sees feed_zone runs for the first time ever.

CompoundRiskGNN can only pass this by using its SHARED per-edge-type weights (the same
"sensor_cluster --GATv2--> zone" and "permit --GATv2--> zone" update rule applies
identically regardless of which zone the message is going to) to transfer the
sensor+permit fusion pattern it learned on reactor_zone/condenser_zone onto feed_zone,
which it's structurally never conditioned a prediction on before.

The flat baseline here is a POOLED Random Forest (one shared model across all zones, not
one-per-zone like eval/tep_classical/run.py and classical_v2.py used) -- each zone's
many-channel sensor stats are aggregated into a small fixed-width, zone-width-agnostic
feature vector (mean/std/max-abs-zscore across that zone's channels) plus a zone one-hot
and the same permit/presence context features the GNN sees, so it's the fairest possible
non-graph competitor: same information, pooled across zones the same way the graph's
shared weights are, but no message passing.

Run: `python -m eval.permit_fusion_benchmark.held_out_zone`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from eval.permit_fusion_benchmark.classical_v2 import _permit_ctx_feat
from eval.permit_fusion_benchmark.graph_builder_v2 import DATA_DIR, N_SAMPLES, load_all_graphs
from models.gnn.graph_builder import CLUSTER_TO_ZONE, SENSOR_CLUSTERS, WINDOW, ZONE_VOCAB
from models.gnn.model import NODE_TYPES, CompoundRiskGNN

REPO_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_TO_ZONE = {
    "s1_reactor_heat_removal": "reactor_zone",
    "s2_condenser_pressure": "condenser_zone",
    "s3_deferred_maintenance": "reactor_zone",
    "s4_feed_system": "feed_zone",
    "s5_common_cause_utility": "reactor_zone",
}
HELD_OUT_SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "s4_feed_system"
HELD_OUT_ZONE = SCENARIO_TO_ZONE[HELD_OUT_SCENARIO]
OUT_DIR = Path(__file__).resolve().parent / f"checkpoints_heldout_{HELD_OUT_ZONE}"
RESULTS_PATH = Path(__file__).resolve().parent / f"results_heldout_{HELD_OUT_ZONE}.json"
SEED = 0
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 32
N_ZONES = len(ZONE_VOCAB)
ZONE_TO_CLUSTER = {z: c for c, z in CLUSTER_TO_ZONE.items()}
SCORED_ZONES = list(CLUSTER_TO_ZONE.values())


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


def zone_pooled_feat(window_df: pd.DataFrame, zone: str) -> np.ndarray:
    """Fixed-width (3-dim), channel-count-agnostic zone summary: comparable across zones
    with different numbers of instrumented channels, unlike the per-channel features used
    elsewhere -- necessary for a single POOLED classifier across zones of different width."""
    cols = SENSOR_CLUSTERS[ZONE_TO_CLUSTER[zone]]
    vals = window_df[cols].to_numpy(dtype=np.float32)
    means = vals.mean(axis=0)
    stds = vals.std(axis=0)
    return np.array([means.mean(), stds.mean(), np.abs(vals - means).max()], dtype=np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading graphs + pooled features for all runs...")
    graphs, labels, manifest = load_all_graphs()
    permits = pd.read_parquet(DATA_DIR / "permits.parquet").set_index("run_id")
    presences = pd.read_parquet(DATA_DIR / "presences.parquet").set_index("run_id")

    held_out_mask = (manifest["scenario_id"] == HELD_OUT_SCENARIO).to_numpy()
    in_dist = manifest[~held_out_mask].reset_index()
    strata = in_dist["scenario_id"] + "_" + in_dist["condition"]
    train_pos, val_pos = train_test_split(np.arange(len(in_dist)), test_size=0.15, random_state=SEED, stratify=strata)
    train_idx = in_dist.loc[train_pos, "index"].to_numpy()
    val_idx = in_dist.loc[val_pos, "index"].to_numpy()
    test_idx = np.where(held_out_mask)[0]  # ALL of s4 -- entirely unseen during training
    print(f"train={len(train_idx)} val={len(val_idx)} held-out-scenario test={len(test_idx)} "
          f"(scenario={HELD_OUT_SCENARIO}, zone={HELD_OUT_ZONE})")

    # ---------- GNN ----------
    def subset(indices):
        return [graphs[i] for i in indices], [labels[i] for i in indices]

    train_graphs, train_labels = subset(train_idx)
    val_graphs, val_labels = subset(val_idx)
    test_graphs, test_labels = subset(test_idx)

    stats = {}
    for ntype in NODE_TYPES:
        all_x = torch.cat([g[ntype].x for g in train_graphs])
        dims = (0, 1) if all_x.dim() == 3 else 0
        mean = all_x.mean(dim=dims)
        std = all_x.std(dim=dims)
        std[std < 1e-6] = 1.0
        stats[ntype] = (mean, std)

    for g in graphs:
        for ntype in NODE_TYPES:
            mean, std = stats[ntype]
            g[ntype].x = (g[ntype].x - mean) / std

    for g, y in zip(train_graphs, train_labels):
        g["zone"].y = torch.tensor(y, dtype=torch.float32)
    for g, y in zip(val_graphs, val_labels):
        g["zone"].y = torch.tensor(y, dtype=torch.float32)
    for g, y in zip(test_graphs, test_labels):
        g["zone"].y = torch.tensor(y, dtype=torch.float32)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CompoundRiskGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0], device=device))

    def run_eval(loader):
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                all_logits.append(logits.cpu())
                all_y.append(batch["zone"].y.cpu())
        logits, y = torch.cat(all_logits), torch.cat(all_y)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1, probs, y

    best_val_f1, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch["zone"].y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        val_p, val_r, val_f1, _, _ = run_eval(val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d} | train_loss {total_loss/len(train_graphs):.4f} | val_F1 {val_f1:.3f}")

    model.load_state_dict(best_state)
    gnn_test_p, gnn_test_r, gnn_test_f1, gnn_probs, gnn_y = run_eval(test_loader)
    gnn_probs_np, gnn_y_np = gnn_probs.numpy(), gnn_y.numpy().astype(int)
    gnn_auc = roc_auc(gnn_probs_np, gnn_y_np)
    print(f"\nGNN on held-out scenario/zone | precision {gnn_test_p:.3f} recall {gnn_test_r:.3f} "
          f"f1 {gnn_test_f1:.3f} AUC {gnn_auc:.4f}")

    # ---------- pooled flat RF ----------
    print("\nExtracting pooled zone-agnostic features for RF...")
    rows_x, rows_y, rows_scenario = [], [], []
    for i, row in manifest.iterrows():
        sensor_df = pd.read_parquet(REPO_ROOT / row["path"])
        window_df = sensor_df.iloc[-WINDOW:]
        permit, presence = permits.loc[row["run_id"]], presences.loc[row["run_id"]]
        ctx = _permit_ctx_feat(permit, presence)
        for zi, zone in enumerate(SCORED_ZONES):
            pooled = zone_pooled_feat(window_df, zone)
            zone_onehot = [1.0 if zone == z else 0.0 for z in SCORED_ZONES]
            rows_x.append(np.concatenate([pooled, zone_onehot, ctx]))
            rows_y.append(1.0 if (bool(row["true_positive"]) and zone == row["zone"]) else 0.0)
            rows_scenario.append(row["scenario_id"])
    X = np.stack(rows_x)
    Y = np.array(rows_y, dtype=np.float32)
    S = np.array(rows_scenario)

    run_scenario_per_row = np.repeat(manifest["scenario_id"].to_numpy(), len(SCORED_ZONES))
    rf_train_rows = run_scenario_per_row != HELD_OUT_SCENARIO
    rf_test_rows = run_scenario_per_row == HELD_OUT_SCENARIO

    rf = RandomForestClassifier(n_estimators=400, max_depth=10, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1).fit(X[rf_train_rows], Y[rf_train_rows])
    rf_probs = rf.predict_proba(X[rf_test_rows])[:, 1]
    rf_y = Y[rf_test_rows]
    rf_pred = (rf_probs > 0.5).astype(int)
    tp = ((rf_pred == 1) & (rf_y == 1)).sum()
    fp = ((rf_pred == 1) & (rf_y == 0)).sum()
    fn = ((rf_pred == 0) & (rf_y == 1)).sum()
    rf_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rf_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    rf_f1 = 2 * rf_p * rf_r / (rf_p + rf_r) if (rf_p + rf_r) > 0 else 0.0
    rf_auc = roc_auc(rf_probs, rf_y.astype(int))
    print(f"Pooled RF on held-out scenario/zone | precision {rf_p:.3f} recall {rf_r:.3f} f1 {rf_f1:.3f} "
          f"AUC {rf_auc:.4f} (n_positive={int(rf_y.sum())}/{len(rf_y)})")

    results = {
        "held_out_scenario": HELD_OUT_SCENARIO,
        "held_out_zone": HELD_OUT_ZONE,
        "note": f"model never saw ANY {HELD_OUT_ZONE} example (positive or negative) during training; "
                 "GNN predictions come from shared per-edge-type weights transferred from the other "
                 "zones; pooled RF is one shared model across zones with zone-width-agnostic features, "
                 "the fairest non-graph competitor. Precision/recall/F1 at the 0.5 threshold are known "
                 "to be uninformative here (the model's score scale for a never-seen zone doesn't match "
                 "the threshold calibrated on training zones) -- AUC is the meaningful number.",
        "n_test_runs": int(len(test_idx)),
        "gnn": {"precision": round(gnn_test_p, 3), "recall": round(gnn_test_r, 3),
                "f1": round(gnn_test_f1, 3), "auc": round(gnn_auc, 4)},
        "pooled_random_forest": {"precision": round(float(rf_p), 3), "recall": round(float(rf_r), 3),
                                   "f1": round(float(rf_f1), 3), "auc": round(rf_auc, 4)},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    torch.save(best_state, OUT_DIR / "checkpoint.pt")
    np.savez(OUT_DIR / "test_scores.npz", gnn_probs=gnn_probs_np, gnn_y=gnn_y_np, rf_probs=rf_probs, rf_y=rf_y)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
