"""
Trains CompoundRiskGNN on the multi-zone heterogeneous graphs (one graph per run, 7 zone
nodes each) and writes per-run, per-zone predicted probabilities to
data/gnn_predictions.parquet, so eval/metrics.py can compute the headline comparison
against the baseline -- including the now-real zone-localization metric.

Labels live on the "zone" node type (g['zone'].y, length 7) rather than on a top-level
g.y, so PyG's batching concatenates them in the same order as the model's per-zone
output automatically.

Split is by whole run (each run = one independent graph, no timestep leakage possible
by construction), stratified by (scenario_id, condition) to keep class balance.

Run directly: `python -m models.gnn.train`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from models.gnn.graph_builder import ZONE_VOCAB, load_all_graphs
from models.gnn.model import CompoundRiskGNN

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 0
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 32
N_ZONES = len(ZONE_VOCAB)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    graphs, labels, manifest = load_all_graphs()
    strata = manifest["scenario_id"] + "_" + manifest["condition"]

    idx = np.arange(len(graphs))
    train_idx, rest_idx = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=strata)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, random_state=SEED, stratify=strata.iloc[rest_idx]
    )

    def subset(indices):
        return [graphs[i] for i in indices], [labels[i] for i in indices]

    train_graphs, train_labels = subset(train_idx)
    val_graphs, val_labels = subset(val_idx)
    test_graphs, test_labels = subset(test_idx)

    # z-score normalize each node type's features using TRAIN-split statistics only,
    # then apply to all splits -- raw sensor scales span ~0 to ~4500, which otherwise
    # stalls GATv2Conv training (verified: loss plateaus at the prior-class entropy).
    node_types = ["sensor_cluster", "permit", "presence", "zone"]
    stats = {}
    for ntype in node_types:
        all_x = torch.cat([g[ntype].x for g in train_graphs])
        mean = all_x.mean(dim=0)
        std = all_x.std(dim=0)
        std[std < 1e-6] = 1.0
        stats[ntype] = (mean, std)

    for g in graphs:
        for ntype in node_types:
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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CompoundRiskGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # positive zone-slots are now rarer (1 positive out of 7 zones at most per run) --
    # weight the loss so the model doesn't just learn to predict all-zero.
    pos_weight = torch.tensor([6.0], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1, best_state = -1.0, None

    def evaluate(loader):
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                all_logits.append(logits.cpu())
                all_y.append(batch["zone"].y.cpu())
        logits = torch.cat(all_logits)
        y = torch.cat(all_y)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1, probs, y

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

        val_p, val_r, val_f1, _, _ = evaluate(val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d} | train_loss {total_loss/len(train_graphs):.4f} "
                  f"| val_P {val_p:.3f} val_R {val_r:.3f} val_F1 {val_f1:.3f}")

    model.load_state_dict(best_state)
    test_p, test_r, test_f1, test_probs, test_y = evaluate(test_loader)
    print(f"\nTEST (per-zone-slot) | precision {test_p:.3f} recall {test_r:.3f} f1 {test_f1:.3f}")

    torch.save(best_state, REPO_ROOT / "models" / "gnn" / "checkpoint.pt")
    # persist normalization stats too -- any later inference (e.g. eval/metrics.py's
    # streaming lead-time re-inference) MUST apply these same train-set stats, since the
    # model was trained on normalized features (raw sensor scales span ~0-4500).
    torch.save(stats, REPO_ROOT / "models" / "gnn" / "norm_stats.pt")

    # write per-run, per-zone predictions for ALL runs (train+val+test)
    model.eval()
    all_probs = []
    full_loader = DataLoader(graphs, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            all_probs.append(probs.reshape(-1, N_ZONES))
    all_probs = np.concatenate(all_probs, axis=0)  # [n_runs, 7]

    split_col = np.full(len(graphs), "train", dtype=object)
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"

    out = manifest.copy()
    for i, zone_name in enumerate(ZONE_VOCAB):
        out[f"gnn_prob__{zone_name}"] = all_probs[:, i]
    out["gnn_prob_true_zone"] = [all_probs[i, ZONE_VOCAB.index(z)] for i, z in enumerate(manifest["zone"])]
    out["gnn_top_zone"] = [ZONE_VOCAB[i] for i in all_probs.argmax(axis=1)]
    out["split"] = split_col
    out_path = REPO_ROOT / "data" / "gnn_predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote per-run, per-zone GNN predictions to {out_path}")


if __name__ == "__main__":
    main()
