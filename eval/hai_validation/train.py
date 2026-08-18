"""
Trains HaiRiskGNN on real HAI 22.04 windows. Mirrors models/gnn/train.py's discipline
(per-node-type z-score normalization fit on the TRAIN split only, class-weighted BCE
since attack windows are a minority, best-val-F1 checkpointing) adapted to HAI's
file-level split (see dataset.py) instead of TEP's run-level split.

Run: `python -m eval.hai_validation.train`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from eval.hai_validation.dataset import TEST_FILES, TRAIN_FILES, VAL_FILES, HaiWindowDataset
from eval.hai_validation.model import NODE_TYPES, HaiRiskGNN

OUT_DIR = Path(__file__).resolve().parent / "checkpoints"
SEED = 0
EPOCHS = 25
LR = 1e-3
BATCH_SIZE = 64
NUM_WORKERS = 4


def compute_norm_stats(loader, device):
    sums = {ntype: None for ntype in NODE_TYPES}
    sqsums = {ntype: None for ntype in NODE_TYPES}
    counts = {ntype: 0 for ntype in NODE_TYPES}
    for batch in loader:
        for ntype in NODE_TYPES:
            x = batch[ntype].x
            dims = (0, 1) if x.dim() == 3 else (0,)
            s = x.sum(dim=dims)
            sq = (x ** 2).sum(dim=dims)
            n = x.shape[0] * (x.shape[1] if x.dim() == 3 else 1)
            sums[ntype] = s if sums[ntype] is None else sums[ntype] + s
            sqsums[ntype] = sq if sqsums[ntype] is None else sqsums[ntype] + sq
            counts[ntype] += n
    stats = {}
    for ntype in NODE_TYPES:
        mean = sums[ntype] / counts[ntype]
        var = sqsums[ntype] / counts[ntype] - mean ** 2
        std = var.clamp(min=1e-12).sqrt()
        std[std < 1e-6] = 1.0
        stats[ntype] = (mean.to(device), std.to(device))
    return stats


def normalize(batch, stats):
    for ntype, (mean, std) in stats.items():
        batch[ntype].x = (batch[ntype].x - mean) / std
    return batch


def evaluate(model, loader, stats, device):
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            batch = normalize(batch.to(device), stats)
            logits = model(batch)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_y.append(batch["zone"].y.cpu())
    probs = torch.cat(all_probs)
    y = torch.cat(all_y)
    preds = (probs > 0.5).float()
    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, probs, y


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = HaiWindowDataset(TRAIN_FILES)
    val_ds = HaiWindowDataset(VAL_FILES)
    test_ds = HaiWindowDataset(TEST_FILES)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    stat_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    stats = compute_norm_stats(stat_loader, device)

    model = HaiRiskGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # per-zone positive fraction (flattened over all zone-slots), not per-window --
    # zones are individually rare positives (P4 never, P1/P2/P3 each a few % at most)
    zone_labels = train_ds.zone_label_matrix()
    pos_frac = float(zone_labels.mean())
    pos_weight = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"train per-zone positive fraction {pos_frac:.4f}, pos_weight {pos_weight.item():.2f}")

    best_val_f1, best_state = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            batch = normalize(batch.to(device), stats)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch["zone"].y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs

        val_p, val_r, val_f1, _, _ = evaluate(model, val_loader, stats, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d} | train_loss {total_loss/n:.4f} | "
              f"val_P {val_p:.3f} val_R {val_r:.3f} val_F1 {val_f1:.3f}")

    model.load_state_dict(best_state)
    test_p, test_r, test_f1, test_probs, test_y = evaluate(model, test_loader, stats, device)
    print(f"\nTEST | precision {test_p:.3f} recall {test_r:.3f} f1 {test_f1:.3f}")

    torch.save(best_state, OUT_DIR / "checkpoint.pt")
    torch.save(stats, OUT_DIR / "norm_stats.pt")
    torch.save({"probs": test_probs, "y": test_y}, OUT_DIR / "test_predictions.pt")
    print(f"Wrote checkpoint + test predictions to {OUT_DIR}")


if __name__ == "__main__":
    main()
