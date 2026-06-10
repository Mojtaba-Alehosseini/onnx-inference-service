"""Tests for the synthetic data generator."""

from __future__ import annotations

import numpy as np

from src.serve.data import FEATURE_DIM, get_splits, make_dataset


def test_make_dataset_shape():
    X, y = make_dataset(n_samples=500)
    assert X.shape == (500, FEATURE_DIM)
    assert y.shape == (500,)
    assert X.dtype == np.float32


def test_make_dataset_deterministic():
    X1, y1 = make_dataset(seed=42)
    X2, y2 = make_dataset(seed=42)
    assert np.allclose(X1, X2)
    assert (y1 == y2).all()


def test_split_no_leakage():
    X_train, X_test, y_train, y_test = get_splits(n_samples=1000)
    # Verify total size
    assert len(X_train) + len(X_test) == 1000
    # Float32 dtype
    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32


def test_binary_labels():
    _, y = make_dataset(n_samples=1000)
    assert set(np.unique(y)).issubset({0, 1})
