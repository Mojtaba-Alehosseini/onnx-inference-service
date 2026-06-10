"""Export ChurnMLP to ONNX (FP32) and assert output parity.

Usage: python -m src.serve.export_onnx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).parents[2]
PT_PATH = ROOT / "models" / "churn_mlp.pt"
ONNX_PATH = ROOT / "models" / "churn_mlp.onnx"
FEATURE_DIM = 19
PARITY_TOL = 1e-4


def export(pt_path: Path = PT_PATH, onnx_path: Path = ONNX_PATH) -> Path:
    """Export pt_path -> onnx_path, check parity, return onnx_path."""
    from src.serve.model import ChurnMLP

    model = ChurnMLP()
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.zeros(1, FEATURE_DIM)
    onnx_path.parent.mkdir(exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported: {onnx_path}  ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # parity check: compare pytorch vs onnxruntime on 200 samples
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((200, FEATURE_DIM)).astype(np.float32)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(x_np)).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"features": x_np})[0]

    max_diff = float(np.abs(pt_out - onnx_out).max())
    print(f"Parity check: max |pytorch - onnx| = {max_diff:.2e}  (tol={PARITY_TOL:.0e})")
    assert max_diff < PARITY_TOL, f"Parity failed: max_diff={max_diff}"
    print("Parity: OK")
    return onnx_path


if __name__ == "__main__":
    export()
