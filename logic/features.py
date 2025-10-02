import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be > 0")
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be > 0")
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)


def compute_log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_technical_features(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": close})
    df["sma_5"] = compute_sma(df["close"], 5)
    df["sma_20"] = compute_sma(df["close"], 20)
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["ret_1"] = compute_log_returns(df["close"]) 
    # Normalize simple ratios
    df["sma_5_ratio"] = df["sma_5"] / df["close"] - 1.0
    df["sma_20_ratio"] = df["sma_20"] / df["close"] - 1.0
    features = df[["sma_5_ratio", "sma_20_ratio", "rsi_14", "ret_1"]].dropna()
    return features


