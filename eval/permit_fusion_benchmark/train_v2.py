"""
Trains CompoundRiskGNN (reused unmodified from models/gnn/model.py) on the permit-fusion
benchmark. Same training discipline as models/gnn/train.py: run-level stratified 70/15/15
split, per-node-type normalization fit on train only, class-weighted BCE.

Run: `python -m eval.permit_fusion_benchmark.train_v2`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from eval.permit_fusion_benchmark.graph_builder_v2 import load_all_graphs
from models.gnn.graph_builder import ZONE_VOCAB
from models.gnn.model import NODE_TYPES, CompoundRiskGNN

OUT_DIR = Path(__file__).resolve().parent / "checkpoints"
SEED = 0
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 32
N_ZONES = len(ZONE_VOCAB)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    graphs, labels, manifest = load_all_graphs()
    strata = manifest["scenario_id"] + "_" + manifest["condition"]

    idx = np.arange(len(graphs))
    train_idx, rest_idx = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=strata)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=SEED, stratify=strata.iloc[rest_idx])

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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CompoundRiskGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
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

    torch.save(best_state, OUT_DIR / "checkpoint.pt")
    torch.save(stats, OUT_DIR / "norm_stats.pt")

    test_manifest = manifest.iloc[test_idx].reset_index(drop=True)
    torch.save({"probs": test_probs, "y": test_y, "test_idx": test_idx,
                "zone_vocab": ZONE_VOCAB}, OUT_DIR / "test_predictions.pt")
    test_manifest.to_csv(OUT_DIR / "test_manifest.csv", index=False)
    print(f"Wrote checkpoint + test predictions to {OUT_DIR}")


if __name__ == "__main__":
    main()
