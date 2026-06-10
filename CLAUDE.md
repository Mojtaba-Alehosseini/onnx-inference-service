# Project: ONNX inference service + benchmark

## What this is
Export a PyTorch MLP to ONNX, INT8-quantize it, serve via FastAPI, and benchmark
latency (p50/p95/p99), throughput, size, and cost-per-1k across PyTorch vs ONNX FP32 vs ONNX INT8.
Cost/efficiency framing = the "dollars-per-watt" mindset applied to model serving.

## Stack
Python 3.10+ / PyTorch 2.5 / ONNX 1.17 / ONNX Runtime 1.21 / FastAPI / pytest / ruff.

## Commands
- Train:     `python -m src.serve.train`
- Export:    `python -m src.serve.export_onnx`
- Quantize:  `python -m src.serve.quantize`
- API:       `uvicorn api.main:app --reload`
- Benchmark: `python -m src.serve.benchmark`
- Tests:     `pytest -q`
- Lint:      `ruff check .`

## Conventions
- After export AND quantization, assert output parity/accuracy within tolerance.
- runner.py serves all three backends behind a common interface.
- Report p50/p95/p99 (not mean). Keep it CPU-servable.
- All model artefacts except .pt (large) are committed (onnx models < 1 MB).

## Done per step
Verify command passes -> commit.
