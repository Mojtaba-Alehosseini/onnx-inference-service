"""Train ChurnMLP and save weights to models/churn_mlp.pt.

Usage: python -m src.serve.train
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parents[2]
MODEL_PATH = ROOT / "models" / "churn_mlp.pt"
METRICS_PATH = ROOT / "models" / "train_metrics.json"

EPOCHS = 40
BATCH_SIZE = 256
LR = 1e-3
SEED = 42


def train() -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from src.serve.data import get_splits
    from src.serve.model import ChurnMLP

    X_train, X_test, y_train, y_test = get_splits()

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ChurnMLP()
    # weighted BCE for class imbalance
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={total_loss/len(train_ds):.4f}")

    # evaluate
    model.eval()
    X_t = torch.from_numpy(X_test)
    with torch.no_grad():
        prob = torch.sigmoid(model(X_t)).numpy()
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == y_test).mean())

    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_test, prob))
    print(f"\nTest  accuracy={acc:.4f}  ROC-AUC={auc:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    metrics = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4), "epochs": EPOCHS}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    train()
