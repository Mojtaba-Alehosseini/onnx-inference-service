# Benchmark report

Measurement: 2000 calls after 200 warm-up iterations, batch=1, CPU-only.
Cost model: $0.05/hr (2-vCPU cloud VM).

## Results

|backend|p50 ms|p95 ms|p99 ms|req/s|size KB|$/1k|accuracy|
|---|---|---|---|---|---|---|---|
|pytorch|0.112|0.16|0.292|8920.6|18.2|2e-06|0.845|
|onnx|0.043|0.064|0.082|23014.9|15.6|1e-06|0.845|
|onnx-int8|0.043|0.058|0.077|23041.5|9.1|1e-06|0.8455|

## Trade-off summary

- **ONNX FP32 vs PyTorch**: 2.60x faster at p50, same accuracy.
- **ONNX INT8 vs PyTorch**: 2.60x faster at p50, 2.00x smaller model.
- **Accuracy delta (FP32 -> INT8)**: -0.0005 pp.

## Recommendation

| SLA | Recommended backend | Reason |
|-----|---------------------|--------|
| < 5 ms p99 | onnx-int8 | Lowest latency + smallest footprint |
| accuracy-critical | onnx | Full FP32 precision, still faster than pytorch |
| debugging / local dev | pytorch | Readable stack traces, no ONNX artefact needed |
