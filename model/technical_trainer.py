import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LogisticModel:
    weights: np.ndarray  # shape (d,)
    bias: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-z))

    def predict_label(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)


def standardize(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0).values
    sigma = X.std(axis=0).replace(0, 1e-8).values
    Xn = (X.values - mu) / sigma
    return Xn, mu, sigma


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    lr: float = 0.05,
    epochs: int = 300,
    l2: float = 1e-3,
    seed: int = 42,
) -> Tuple[LogisticModel, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xn, mu, sigma = standardize(X)
    n, d = Xn.shape
    w = rng.normal(0, 0.1, size=d)
    b = 0.0
    yv = y.values.astype(float)

    for _ in range(epochs):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        # gradients
        grad_w = (Xn.T @ (p - yv)) / n + l2 * w
        grad_b = float(np.mean(p - yv))
        w -= lr * grad_w
        b -= lr * grad_b
    return LogisticModel(weights=w, bias=b), mu, sigma


def prepare_direction_labels(close: pd.Series, horizon: int = 1) -> pd.Series:
    future = close.shift(-horizon)
    return (future > close).astype(int).iloc[:-horizon]


def transform_with_stats(X: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X.values - mu) / sigma


