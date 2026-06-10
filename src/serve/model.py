"""PyTorch MLP for tabular binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """Two-hidden-layer MLP: 19 -> 64 -> 32 -> 1 (logit, not sigmoid)."""

    def __init__(self, input_dim: int = 19, h1: int = 64, h2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
