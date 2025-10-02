import pandas as pd
import numpy as np

from logic.features import build_technical_features, compute_rsi, compute_sma
from model.technical_trainer import prepare_direction_labels, train_logistic_regression


def _make_close(n=200, seed=1):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1, size=n).cumsum()
    close = pd.Series(100 + steps).abs() + 1
    close.index = pd.date_range("2020-01-01", periods=n, freq="D")
    return close


def test_feature_shapes():
    close = _make_close()
    feats = build_technical_features(close)
    assert set(["sma_5_ratio", "sma_20_ratio", "rsi_14", "ret_1"]) <= set(feats.columns)
    assert len(feats) < len(close)


def test_rsi_bounds():
    close = _make_close()
    rsi = compute_rsi(close, 14)
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_train_logistic_regression_runs():
    close = _make_close()
    feats = build_technical_features(close)
    y = prepare_direction_labels(close.loc[feats.index])
    feats = feats.loc[y.index]
    model, mu, sigma = train_logistic_regression(feats, y, epochs=50)
    # model should produce probabilities in [0,1]
    p = model.predict_proba(((feats.values - mu) / sigma)[:5])
    assert (p >= 0).all() and (p <= 1).all()


