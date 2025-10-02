import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from logic.features import build_technical_features
from model.sentiment_model import analyze_sentiment
from model.technical_trainer import (
    LogisticModel,
    prepare_direction_labels,
    train_logistic_regression,
    transform_with_stats,
)


@dataclass
class TechnicalPredictor:
    model: LogisticModel
    mu: object
    sigma: object

    def predict_direction_proba(self, features: pd.DataFrame) -> float:
        X = transform_with_stats(features.tail(1), self.mu, self.sigma)
        proba_up = float(self.model.predict_proba(X)[0])
        return proba_up


def fetch_history(symbol: str, period: str = "6mo") -> pd.Series:
    data = yf.download(symbol, period=period, progress=False)
    if data.empty:
        raise ValueError(f"No history for {symbol}")
    close = data["Close"]
    # yfinance may return a DataFrame if multiple tickers or a nested column
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            raise ValueError(f"No close prices for {symbol}")
        close = close.iloc[:, 0]
    return close.astype(float)


def train_technical_predictor(symbol: str, period: str = "6mo") -> TechnicalPredictor:
    close = fetch_history(symbol, period=period)
    features = build_technical_features(close)
    # Align labels
    y = prepare_direction_labels(close.loc[features.index], horizon=1)
    features = features.loc[y.index]
    model, mu, sigma = train_logistic_regression(features, y)
    return TechnicalPredictor(model=model, mu=mu, sigma=sigma)


def aggregate_decision(symbol: str, news_text: Optional[str]) -> Tuple[str, float, float]:
    """Return (decision, sentiment_score, tech_up_proba)

    Rule:
    - If sentiment highly positive and tech proba>0.55 -> BUY
    - If sentiment highly negative and tech proba<0.45 -> SELL
    - Else HOLD
    """
    # Sentiment
    if news_text and news_text.strip():
        label, s_score = analyze_sentiment(news_text)
        sent = label.lower()
        sent_score = float(s_score)
    else:
        sent = "neutral"
        sent_score = 0.5

    # Technical
    predictor = train_technical_predictor(symbol)
    close = fetch_history(symbol, period="3mo")
    features = build_technical_features(close)
    tech_proba = predictor.predict_direction_proba(features)

    decision = "HOLD"
    if sent == "positive" and sent_score >= 0.70 and tech_proba >= 0.55:
        decision = "BUY"
    elif sent == "negative" and sent_score >= 0.70 and tech_proba <= 0.45:
        decision = "SELL"
    return decision, sent_score, float(tech_proba)


