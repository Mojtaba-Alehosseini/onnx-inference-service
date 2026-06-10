"""Tests for ONNX export and parity check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
PT_PATH = ROOT / "models" / "churn_mlp.pt"
ONNX_PATH = ROOT / "models" / "churn_mlp.onnx"


@pytest.fixture(scope="module")
def onnx_path():
    if not PT_PATH.exists():
        pytest.skip("model not trained — run python -m src.serve.train")
    from src.serve.export_onnx import export

    return export()


def test_onnx_file_exists(onnx_path):
    assert onnx_path.exists()
    assert onnx_path.suffix == ".onnx"


def test_onnx_file_non_empty(onnx_path):
    assert onnx_path.stat().st_size > 0


def test_parity_on_random_input(onnx_path):
    import onnxruntime as ort
    import torch

    from src.serve.model import ChurnMLP

    model = ChurnMLP()
    model.load_state_dict(torch.load(PT_PATH, map_location="cpu", weights_only=True))
    model.eval()

    rng = np.random.default_rng(123)
    x = rng.standard_normal((50, 19)).astype(np.float32)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(x)).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"features": x})[0]

    assert np.allclose(pt_out, onnx_out, atol=1e-4), (
        f"max diff = {np.abs(pt_out - onnx_out).max():.2e}"
    )


def test_dynamic_batch(onnx_path):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for batch in (1, 4, 16):
        x = np.zeros((batch, 19), dtype=np.float32)
        out = sess.run(None, {"features": x})[0]
        assert out.shape == (batch,)
