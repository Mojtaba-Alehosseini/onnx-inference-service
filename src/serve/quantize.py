"""Dynamic INT8 quantization via ONNX Runtime.

Usage: python -m src.serve.quantize
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).parents[2]
ONNX_PATH = ROOT / "models" / "churn_mlp.onnx"
INT8_PATH = ROOT / "models" / "churn_mlp_int8.onnx"
ACC_TOL = 0.01  # allow up to 1pp accuracy drop


def quantize(onnx_path: Path = ONNX_PATH, int8_path: Path = INT8_PATH) -> Path:
    """Quantize onnx_path -> int8_path, report size and accuracy delta."""
    int8_path.parent.mkdir(exist_ok=True)
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)

    fp32_size = onnx_path.stat().st_size / 1024
    int8_size = int8_path.stat().st_size / 1024
    ratio = fp32_size / int8_size
    print(f"FP32: {fp32_size:.1f} KB  INT8: {int8_size:.1f} KB  ({ratio:.2f}x smaller)")

    # accuracy comparison on the test split
    from src.serve.data import get_splits

    _, X_test, _, y_test = get_splits()

    def _acc(model_path: Path) -> float:
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        logits = sess.run(None, {"features": X_test})[0]
        prob = 1.0 / (1.0 + np.exp(-logits))
        return float(((prob >= 0.5).astype(int) == y_test).mean())

    fp32_acc = _acc(onnx_path)
    int8_acc = _acc(int8_path)
    delta = fp32_acc - int8_acc
    print(f"Accuracy  FP32={fp32_acc:.4f}  INT8={int8_acc:.4f}  delta={delta:+.4f}")
    assert abs(delta) <= ACC_TOL, f"Accuracy delta {delta:+.4f} exceeds tolerance {ACC_TOL}"
    print(f"Accuracy delta within tolerance ({ACC_TOL}): OK")
    return int8_path


if __name__ == "__main__":
    quantize()
