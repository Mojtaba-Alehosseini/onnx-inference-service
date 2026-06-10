"""Tests for ChurnMLP architecture."""

from __future__ import annotations

import torch

from src.serve.model import ChurnMLP


def test_output_shape_batch():
    model = ChurnMLP()
    model.eval()
    x = torch.zeros(8, 19)
    out = model(x)
    assert out.shape == (8,)


def test_output_shape_single():
    model = ChurnMLP()
    model.eval()
    x = torch.zeros(1, 19)
    out = model(x)
    assert out.shape == (1,)


def test_predict_proba_range():
    model = ChurnMLP()
    model.eval()
    x = torch.randn(50, 19)
    probs = model.predict_proba(x).numpy()
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
