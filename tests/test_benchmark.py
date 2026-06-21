"""Tests for the benchmark utility functions (pure/non-GPU)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.serve.benchmark import _measure, _model_size_kb, _rows_to_markdown


def _dummy_runner(x: np.ndarray) -> np.ndarray:
    """Instant dummy runner — just returns zeros."""
    return np.zeros(x.shape[0], dtype=np.float32)


def test_measure_returns_correct_count():
    x = np.zeros((1, 19), dtype=np.float32)
    times = _measure(_dummy_runner, x, n_warmup=5, n_reps=20)
    assert times.shape == (20,)


def test_measure_non_negative():
    x = np.zeros((1, 19), dtype=np.float32)
    times = _measure(_dummy_runner, x, n_warmup=2, n_reps=10)
    assert (times >= 0).all()


def test_model_size_kb_existing_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as f:
        f.write(b"x" * 1024)
        tmp = Path(f.name)
    size = _model_size_kb(tmp)
    tmp.unlink()
    assert size == 1.0


def test_model_size_kb_missing_file():
    import math
    assert math.isnan(_model_size_kb(Path("/nonexistent/model.onnx")))


def test_rows_to_markdown_has_headers():
    rows = [{"backend": "onnx", "p50_ms": 0.1, "p95_ms": 0.2, "p99_ms": 0.3,
             "throughput_rps": 9999.9, "size_kb": 50.0, "cost_per_1k_usd": 0.000001, "accuracy": 0.95}]
    md = _rows_to_markdown(rows)
    assert "backend" in md
    assert "p50 ms" in md
    assert "accuracy" in md


def test_rows_to_markdown_has_separator():
    rows = [{"backend": "onnx", "p50_ms": 0.1, "p95_ms": 0.2, "p99_ms": 0.3,
             "throughput_rps": 9999.9, "size_kb": 50.0, "cost_per_1k_usd": 0.000001, "accuracy": 0.95}]
    md = _rows_to_markdown(rows)
    assert "---" in md


def test_rows_to_markdown_contains_values():
    rows = [{"backend": "pytorch", "p50_ms": 1.23, "p95_ms": 2.0, "p99_ms": 3.0,
             "throughput_rps": 813.0, "size_kb": 100.0, "cost_per_1k_usd": 0.00001, "accuracy": 0.91}]
    md = _rows_to_markdown(rows)
    assert "pytorch" in md
    assert "1.23" in md
