FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (smaller image — no CUDA)
RUN pip install --no-cache-dir torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

ENV BACKEND=onnx
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
