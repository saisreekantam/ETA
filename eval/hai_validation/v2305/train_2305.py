"""
Trains HaiRiskGNN (same architecture/class as the 22.04 run, reused unmodified) on HAI
23.05. Same discipline as eval/hai_validation/train.py: per-node-type z-score fit on
train split only, class-weighted BCE, best-val-F1 checkpointing.

Run: `python -m eval.hai_validation.v2305.train_2305`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from eval.hai_validation.model import HaiRiskGNN
from eval.hai_validation.train import BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, SEED, compute_norm_stats, normalize
from eval.hai_validation.train_no_graph import evaluate
from eval.hai_validation.v2305.common import TEST_FILES, TRAIN_FILES, VAL_FILES
from eval.hai_validation.v2305.dataset_2305 import HaiWindowDataset2305

OUT_DIR = Path(__file__).resolve().parent / "checkpoints_2305"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = HaiWindowDataset2305(TRAIN_FILES)
    val_ds = HaiWindowDataset2305(VAL_FILES)
    test_ds = HaiWindowDataset2305(TEST_FILES)
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
