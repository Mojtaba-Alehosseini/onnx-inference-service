"""Deterministic synthetic binary-classification dataset (churn-like, 19 features).

Run standalone: ``python -m src.serve.data`` prints a shape summary.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_DIM = 19
SEED = 42


def make_dataset(
    n_samples: int = 10_000,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X is float32 (n_samples, 19) and y is int (n_samples,)."""
    rng = np.random.default_rng(seed)

    # 19 features: first 10 informative, next 9 noise
    X_info = rng.standard_normal((n_samples, 10)).astype(np.float32)
    X_noise = rng.standard_normal((n_samples, 9)).astype(np.float32)

    # Non-linear boundary: logistic of a weighted mix
    w = rng.standard_normal(10).astype(np.float32)
    logit = X_info @ w + 0.5 * (X_info[:, 0] ** 2) - 0.3 * (X_info[:, 2] * X_info[:, 5])
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n_samples) < prob).astype(np.int64)

    X = np.concatenate([X_info, X_noise], axis=1)
    return X, y


def get_splits(
    n_samples: int = 10_000,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, X_test, y_train, y_test) — scaled, float32."""
    X, y = make_dataset(n_samples=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_splits()
    churn_rate = y_test.mean()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Churn rate (test): {churn_rate:.3f}")
