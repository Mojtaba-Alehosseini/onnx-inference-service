"""Tests for the unified runner interface — all backends produce consistent outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
ONNX_PATH = ROOT / "models" / "churn_mlp.onnx"
INT8_PATH = ROOT / "models" / "churn_mlp_int8.onnx"
PT_PATH = ROOT / "models" / "churn_mlp.pt"


@pytest.fixture(scope="module")
def sample_input():
    rng = np.random.default_rng(999)
    return rng.standard_normal((10, 19)).astype(np.float32)


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="ONNX model not exported yet")
def test_onnx_runner_shape(sample_input):
    from src.serve.runner import get_runner

    runner = get_runner("onnx")
    out = runner(sample_input)
    assert out.shape == (10,)
    assert out.dtype == np.float32


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="ONNX model not exported yet")
def test_onnx_runner_proba_range(sample_input):
    from src.serve.runner import get_runner

    runner = get_runner("onnx")
    out = runner(sample_input)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@pytest.mark.skipif(not INT8_PATH.exists(), reason="INT8 model not quantized yet")
def test_int8_runner_shape(sample_input):
    from src.serve.runner import get_runner

    runner = get_runner("onnx-int8")
    out = runner(sample_input)
    assert out.shape == (10,)


@pytest.mark.skipif(
    not (ONNX_PATH.exists() and INT8_PATH.exists()),
    reason="Both ONNX models required",
)
def test_fp32_int8_consistency(sample_input):
    """INT8 predictions agree with FP32 on > 95% of samples."""
    from src.serve.runner import get_runner

    fp32_runner = get_runner("onnx")
    int8_runner = get_runner("onnx-int8")

    fp32_labels = (fp32_runner(sample_input) >= 0.5).astype(int)
    int8_labels = (int8_runner(sample_input) >= 0.5).astype(int)
    agreement = float((fp32_labels == int8_labels).mean())
    assert agreement >= 0.95, f"FP32/INT8 agreement only {agreement:.2%}"


@pytest.mark.skipif(
    not (PT_PATH.exists() and ONNX_PATH.exists()),
    reason="Both PyTorch and ONNX models required",
)
def test_pytorch_onnx_consistency(sample_input):
    """PyTorch and ONNX predictions agree on > 98% of samples."""
    from src.serve.runner import get_runner

    pt_runner = get_runner("pytorch")
    onnx_runner = get_runner("onnx")

    pt_labels = (pt_runner(sample_input) >= 0.5).astype(int)
    onnx_labels = (onnx_runner(sample_input) >= 0.5).astype(int)
    agreement = float((pt_labels == onnx_labels).mean())
    assert agreement >= 0.98, f"PyTorch/ONNX agreement only {agreement:.2%}"


def test_unknown_backend_raises():
    from src.serve.runner import get_runner

    with pytest.raises(ValueError, match="Unknown backend"):
        get_runner("unknown-backend")
