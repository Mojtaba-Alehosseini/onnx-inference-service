"""Latency / throughput / size / cost benchmark across three backends.

Usage: python -m src.serve.benchmark
Writes: reports/benchmark.md
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
REPORTS_DIR = ROOT / "reports"

WARMUP = 200
REPS = 2000
BATCH_SIZE = 1
# Cost model: a 2-vCPU cloud VM ~ $0.05/hour
COST_PER_HOUR_USD = 0.05
FEATURE_DIM = 19


def _measure(runner, x: np.ndarray, n_warmup: int, n_reps: int) -> np.ndarray:
    """Return latency in ms for each of n_reps calls (after warmup)."""
    for _ in range(n_warmup):
        runner(x)
    times = np.empty(n_reps)
    for i in range(n_reps):
        t0 = time.perf_counter()
        runner(x)
        times[i] = (time.perf_counter() - t0) * 1000.0  # ms
    return times


def _model_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024 if path.exists() else float("nan")


def run_benchmark() -> dict:
    from src.serve.data import get_splits
    from src.serve.runner import INT8_PATH, ONNX_PATH, PT_PATH, get_runner

    _, X_test, _, y_test = get_splits()
    x_single = X_test[:BATCH_SIZE]

    rows = []
    for backend in ("pytorch", "onnx", "onnx-int8"):
        print(f"  [{backend}] warming up + measuring {REPS} calls...", end=" ", flush=True)
        runner = get_runner(backend)
        latencies = _measure(runner, x_single, WARMUP, REPS)

        # accuracy on full test set
        all_probs = runner(X_test)
        acc = float(((all_probs >= 0.5).astype(int) == y_test).mean())

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        throughput = 1000.0 / p50  # req/s
        cost_per_1k = (COST_PER_HOUR_USD / 3600_000.0) * p50 * 1000  # $

        path_map = {"pytorch": PT_PATH, "onnx": ONNX_PATH, "onnx-int8": INT8_PATH}
        size_kb = _model_size_kb(path_map[backend])

        rows.append(
            {
                "backend": backend,
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
                "throughput_rps": round(throughput, 1),
                "size_kb": round(size_kb, 1),
                "cost_per_1k_usd": round(cost_per_1k, 6),
                "accuracy": round(acc, 4),
            }
        )
        print(f"p50={p50:.2f}ms  p99={p99:.2f}ms  acc={acc:.4f}")

    return rows


def _rows_to_markdown(rows: list[dict]) -> str:
    headers = ["backend", "p50 ms", "p95 ms", "p99 ms", "req/s", "size KB", "$/1k", "accuracy"]
    keys = ["backend", "p50_ms", "p95_ms", "p99_ms", "throughput_rps", "size_kb", "cost_per_1k_usd", "accuracy"]
    sep = "|" + "|".join("---" for _ in headers) + "|"
    head = "|" + "|".join(headers) + "|"
    body_lines = []
    for row in rows:
        body_lines.append("|" + "|".join(str(row[k]) for k in keys) + "|")
    return "\n".join([head, sep] + body_lines)


def save_report(rows: list[dict]) -> Path:
    import json

    REPORTS_DIR.mkdir(exist_ok=True)
    md_path = REPORTS_DIR / "benchmark.md"
    json_path = REPORTS_DIR / "benchmark.json"

    # write committed backing file
    json_path.write_text(json.dumps(rows, indent=2))

    onnx = next(r for r in rows if r["backend"] == "onnx")
    int8 = next(r for r in rows if r["backend"] == "onnx-int8")
    pytorch = next(r for r in rows if r["backend"] == "pytorch")
    speedup = pytorch["p50_ms"] / onnx["p50_ms"]
    int8_speedup = pytorch["p50_ms"] / int8["p50_ms"]
    size_reduction = pytorch["size_kb"] / int8["size_kb"]

    md = f"""# Benchmark report

Measurement: {REPS} calls after {WARMUP} warm-up iterations, batch=1, CPU-only.
Cost model: ${COST_PER_HOUR_USD}/hr (2-vCPU cloud VM).

## Results

{_rows_to_markdown(rows)}

## Trade-off summary

- **ONNX FP32 vs PyTorch**: {speedup:.2f}x faster at p50, same accuracy.
- **ONNX INT8 vs PyTorch**: {int8_speedup:.2f}x faster at p50, {size_reduction:.2f}x smaller model.
- **Accuracy delta (FP32 -> INT8)**: {onnx["accuracy"] - int8["accuracy"]:+.4f} pp.

## Recommendation

| SLA | Recommended backend | Reason |
|-----|---------------------|--------|
| < 5 ms p99 | onnx-int8 | Lowest latency + smallest footprint |
| accuracy-critical | onnx | Full FP32 precision, still faster than pytorch |
| debugging / local dev | pytorch | Readable stack traces, no ONNX artefact needed |
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"\nReport: {md_path}")
    return md_path


if __name__ == "__main__":
    print(f"Benchmarking ({REPS} reps, batch={BATCH_SIZE})...")
    rows = run_benchmark()
    save_report(rows)
