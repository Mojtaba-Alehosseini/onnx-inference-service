"""FastAPI contract tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
ONNX_PATH = ROOT / "models" / "churn_mlp.onnx"

VALID_FEATURES = [0.1 * i for i in range(19)]


@pytest.fixture(scope="module")
def client():
    if not ONNX_PATH.exists():
        pytest.skip("ONNX model not exported — run python -m src.serve.export_onnx")
    os.environ["BACKEND"] = "onnx"
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid(client):
    resp = client.post("/predict", json={"values": VALID_FEATURES})
    assert resp.status_code == 200
    body = resp.json()
    assert "probability" in body
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in (0, 1)


def test_predict_wrong_dim(client):
    resp = client.post("/predict", json={"values": [0.0] * 10})
    assert resp.status_code == 422


def test_predict_missing_field(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
