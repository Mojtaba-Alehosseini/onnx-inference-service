"""FastAPI inference service — multi-backend (pytorch | onnx | onnx-int8).

Set BACKEND env var (default: onnx). Run:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

FEATURE_DIM = 19
_runner = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runner  # noqa: PLW0603
    from src.serve.runner import get_runner

    _runner = get_runner(os.getenv("BACKEND", "onnx"))
    yield
    _runner = None


app = FastAPI(title="ONNX Inference Service", lifespan=lifespan)


class Features(BaseModel):
    values: list[float]

    @field_validator("values")
    @classmethod
    def check_dim(cls, v: list[float]) -> list[float]:
        if len(v) != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} features, got {len(v)}")
        return v


class Prediction(BaseModel):
    probability: float
    label: int
    backend: str


@app.get("/health")
async def health():
    return {"status": "ok", "backend": os.getenv("BACKEND", "onnx")}


@app.post("/predict", response_model=Prediction)
async def predict(body: Features):
    if _runner is None:
        raise HTTPException(503, "Model not loaded")
    x = np.array(body.values, dtype=np.float32).reshape(1, -1)
    prob = float(_runner(x)[0])
    return Prediction(
        probability=round(prob, 6),
        label=int(prob >= 0.5),
        backend=os.getenv("BACKEND", "onnx"),
    )
