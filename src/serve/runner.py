"""Unified inference runner for all three backends: pytorch | onnx | onnx-int8.

All runners share the same interface:
    runner(x: np.ndarray[float32]) -> np.ndarray[float32]   (sigmoid probabilities)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = Path(__file__).parents[2]
PT_PATH = ROOT / "models" / "churn_mlp.pt"
ONNX_PATH = ROOT / "models" / "churn_mlp.onnx"
INT8_PATH = ROOT / "models" / "churn_mlp_int8.onnx"


class InferenceRunner(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray: ...


class PyTorchRunner:
    def __init__(self, pt_path: Path = PT_PATH) -> None:
        import torch

        from src.serve.model import ChurnMLP

        self._model = ChurnMLP()
        self._model.load_state_dict(
            torch.load(pt_path, map_location="cpu", weights_only=True)
        )
        self._model.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self._model(torch.from_numpy(x.astype(np.float32)))
        return torch.sigmoid(logits).numpy().astype(np.float32)


class OnnxRunner:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        logits = self._sess.run(None, {"features": x.astype(np.float32)})[0]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def get_runner(backend: str | None = None) -> InferenceRunner:
    """Return the runner for ``backend`` (or env var BACKEND, default onnx)."""
    b = (backend or os.getenv("BACKEND") or "onnx").lower()
    if b == "pytorch":
        return PyTorchRunner()
    if b == "onnx":
        return OnnxRunner(ONNX_PATH)
    if b in ("onnx-int8", "int8"):
        return OnnxRunner(INT8_PATH)
    raise ValueError(f"Unknown backend {b!r} — choose pytorch|onnx|onnx-int8")
