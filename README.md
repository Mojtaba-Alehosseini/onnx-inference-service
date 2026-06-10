# ONNX Inference Service

Export a PyTorch model to ONNX, apply INT8 quantisation, serve it behind a FastAPI endpoint,
and benchmark latency (p50/p95/p99), throughput, model size, and cost-per-1 000 inferences
across three backends. Numbers from `reports/benchmark.json` (committed).

## Benchmark (reproduce: `python -m src.serve.benchmark`)

2 000 calls after 200 warm-up iterations, batch=1, CPU-only (2-vCPU VM @ $0.05/hr).

| backend | p50 ms | p95 ms | p99 ms | req/s | size KB | $/1 k | accuracy |
|---------|-------:|-------:|-------:|------:|--------:|------:|----------|
| pytorch | 0.112 | 0.160 | 0.292 | 8 921 | 18.2 | 0.0000020 | 0.8450 |
| **onnx** | **0.043** | **0.064** | **0.082** | **23 015** | 15.6 | 0.0000010 | 0.8450 |
| **onnx-int8** | **0.043** | 0.058 | 0.077 | 23 042 | **9.1** | 0.0000010 | 0.8455 |

Key findings:

- **ONNX FP32 is 2.6x faster than PyTorch** at p50 (0.043 ms vs 0.112 ms); same accuracy.
- **INT8 is 1.71x smaller** than FP32 (9.1 KB vs 15.6 KB) with negligible accuracy delta (+0.0005).
- Cost halves from PyTorch to ONNX at the same hardware tier.

### SLA recommendation

| SLA | Backend | Reason |
|-----|---------|--------|
| < 1 ms p99 | onnx-int8 | Smallest footprint + lowest latency (0.077 ms p99) |
| accuracy-critical | onnx | FP32 precision, 2.6x faster than pytorch |
| debugging / local dev | pytorch | Readable stack traces; no ONNX artefact required |

## Model

Two-layer MLP (19 → 64 → 32 → 1) trained on a deterministic synthetic binary-classification
dataset (10 000 rows, 19 features, churn-like boundary). Seed 42, reproducible.
Training metrics (from `models/train_metrics.json`): accuracy=0.845, ROC-AUC=0.931.

The point is **not** the model — it is the serving pipeline and the benchmark discipline.
The same export → quantise → serve → benchmark pattern applies to any PyTorch model.

## Problem

Most ML demos stop at "it returns a prediction." Real serving decisions require numbers:
*How much faster is ONNX Runtime? What does quantisation cost in accuracy? Which backend
meets a given SLA, and at what dollar cost?* This project answers those questions with
measured data — the same "dollars-per-watt" mindset I applied to a 100+ GPU HPC fleet.

## Approach

```
PyTorch MLP (train.py)
    |
    v
ONNX export (torch.onnx.export, opset 17) --> parity assert (max|pt-onnx| < 1e-4)
    |
    v
INT8 quantisation (onnxruntime.quantization.quantize_dynamic, QInt8)
    |           --> accuracy assert (delta <= 1pp)
    v
FastAPI /predict (backend selectable via BACKEND env var)
    |
    v
Benchmark harness: p50/p95/p99 latency, throughput, size, $/1k --> reports/benchmark.json
```

## Run it

```bash
# CPU-only torch (smaller download than the default CUDA build)
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e .

# Model artefacts are already committed — skip training if cloning fresh
python -m src.serve.train          # optional: retrain from scratch
python -m src.serve.export_onnx    # optional: re-export
python -m src.serve.quantize       # optional: re-quantise

# Start API (default backend: onnx)
uvicorn api.main:app --port 8000
# Switch backend: BACKEND=onnx-int8 uvicorn api.main:app --port 8000

# Benchmark
python -m src.serve.benchmark

# Tests + lint
pytest -q
ruff check .
```

### API usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"values": [0.1, -0.3, 0.8, 1.2, -0.5, 0.0, 0.7, -1.1, 0.4, 0.9,
                  0.2, -0.6, 1.0, -0.2, 0.5, 0.3, -0.9, 0.6, -0.4]}'
```

Response: `{"probability": 0.73, "label": 1, "backend": "onnx"}`

## Structure

```
onnx-inference-service/
├── src/serve/
│   ├── data.py          # deterministic synthetic dataset (seed=42)
│   ├── model.py         # PyTorch ChurnMLP (19->64->32->1)
│   ├── train.py         # train + save models/churn_mlp.pt
│   ├── export_onnx.py   # torch.onnx.export + parity check
│   ├── quantize.py      # onnxruntime dynamic INT8 + accuracy check
│   ├── runner.py        # unified inference interface (all 3 backends)
│   └── benchmark.py     # p50/p95/p99 latency + throughput + cost
├── api/main.py          # FastAPI: POST /predict, GET /health
├── models/              # committed: .pt (18 KB), .onnx (16 KB), .onnx int8 (9 KB)
├── reports/
│   ├── benchmark.json   # raw numbers (all README claims traced here)
│   └── benchmark.md     # human-readable report with recommendation
├── Dockerfile
└── tests/               # 24 hermetic tests
```

## Limitations

- Synthetic dataset; results will differ on real tabular data (more features, sparser signal).
- Dynamic INT8 quantises weights only; activations remain FP32. Static quantisation would
  yield larger size reductions but requires a calibration dataset.
- Benchmark is single-threaded batch=1; concurrent load (Locust / async) would show
  different throughput characteristics.

## Attributions

- [PyTorch](https://pytorch.org) — BSD-style
- [ONNX Runtime](https://onnxruntime.ai) — MIT
- [ONNX](https://onnx.ai) — Apache-2.0
- [FastAPI](https://fastapi.tiangolo.com) — MIT
