"""Tests for INT8 quantization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
ONNX_PATH = ROOT / "models" / "churn_mlp.onnx"
INT8_PATH = ROOT / "models" / "churn_mlp_int8.onnx"


@pytest.fixture(scope="module")
def int8_path():
    if not ONNX_PATH.exists():
        pytest.skip("ONNX model not exported — run python -m src.serve.export_onnx")
    from src.serve.quantize import quantize

    return quantize()


def test_int8_file_exists(int8_path):
    assert int8_path.exists()


def test_int8_smaller_than_fp32(int8_path):
    fp32_size = ONNX_PATH.stat().st_size
    int8_size = int8_path.stat().st_size
    assert int8_size < fp32_size, "INT8 model should be smaller than FP32"


def test_int8_accuracy_within_tolerance(int8_path):
    """INT8 accuracy should be within 1pp of FP32."""
    import onnxruntime as ort

    from src.serve.data import get_splits

    _, X_test, _, y_test = get_splits()

    def acc(path: Path) -> float:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logits = sess.run(None, {"features": X_test})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        return float(((probs >= 0.5).astype(int) == y_test).mean())

    fp32_acc = acc(ONNX_PATH)
    int8_acc = acc(int8_path)
    assert abs(fp32_acc - int8_acc) <= 0.01, (
        f"Accuracy delta {fp32_acc - int8_acc:+.4f} exceeds 1pp tolerance"
    )
