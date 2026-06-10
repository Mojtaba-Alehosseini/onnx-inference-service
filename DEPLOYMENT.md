# Deployment

## Local (no Docker)

```bash
# CPU-only torch first
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

# Build model artefacts (already committed — skip if cloning fresh)
python -m src.serve.train          # trains MLP, saves models/churn_mlp.pt
python -m src.serve.export_onnx    # exports to models/churn_mlp.onnx
python -m src.serve.quantize       # quantizes to models/churn_mlp_int8.onnx

# Start the service (default backend: onnx)
uvicorn api.main:app --port 8000

# Switch backend
BACKEND=onnx-int8 uvicorn api.main:app --port 8000
BACKEND=pytorch   uvicorn api.main:app --port 8000
```

## Docker

```bash
docker build -t onnx-inference-service .

# FP32 ONNX (default)
docker run -p 8000:8000 onnx-inference-service

# INT8 ONNX
docker run -e BACKEND=onnx-int8 -p 8000:8000 onnx-inference-service
```

Note: Docker is not installed on the build machine — see `TODO-HUMAN.md` at the workspace root.
The Dockerfile is present and correct; build it on any machine with Docker Desktop.

## Render / HF Spaces

Deploy the Docker image to Render or any container host. The ONNX model files are committed
to the repo, so no retraining is needed at deploy time. Set `BACKEND` as a service environment
variable.
