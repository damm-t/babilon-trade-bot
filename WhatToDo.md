# Babilon v3 — Architecture Diagram & Implementation Roadmap

**Purpose:** A complete, actionable blueprint to evolve *Babilon Trade Bot* into a hybrid AI-driven trading system (sentiment + market signals), with milestones, file layout, data flow, and implementation tasks.

---

## 1. Overview

Goal: turn the repo from a FinBERT + rule system into a production-ready hybrid signal engine that:

* Combines sentiment and market features
* Trains lightweight ML models and ensembles with rule logic
* Backtests with realistic market constraints
* Executes safe paper trades with risk controls
* Provides monitoring, retraining, and observability

---

## 2. High-level Architecture (Mermaid)

```mermaid
flowchart TD
  subgraph INPUTS
    A[News & Text APIs]\n(NewsAPI, RSS, Twitter, Reddit) -->|articles| B[Text Preprocessing]
    C[Market Data APIs]\n(Yahoo, Alpaca, Polygon) -->|price ticks| D[Price/TA Engine]
  end

  B --> E[Sentiment Models]\nE --> F[Feature Fusion]
  D --> F

  F --> G[Signal Engine]\n  subgraph MODEL
    G --> H[ML Model (LightGBM/XGBoost)]
    G --> I[Rule Logic / Heuristics]
    H --> J[Ensembler]
    I --> J
  end

  J --> K[Trading Decision]
  K --> L[Risk Management]\nL --> M[Broker API (Alpaca Paper)]
  M --> N[Execution & Order Status]

  N --> O[Metrics & Logging]
  O --> P[Dashboard / Monitoring]
  O --> Q[Backtesting Engine]
  Q --> R[Model Retraining]
  R --> H

```

---

## 3. Component Breakdown

* **Input Layer**

  * News API adapters (NewsAPI, custom RSS, Twitter/Reddit scrapers)
  * Market adapters (yfinance / Alpaca / Polygon)

* **Preprocessing**

  * Text deduplication, filtering, NER to confirm ticker relevance
  * Timestamp normalization, timezone handling

* **Sentiment Models**

  * FinBERT / FinancialBERT ensemble
  * Fallback lexicon (Loughran-McDonald) for robustness
  * Calibration (Platt or isotonic) and output: probability + confidence

* **Price / TA Engine**

  * Compute indicators: SMA, EMA, RSI, MACD, ATR, Volume z-score
  * Market regime detection (simple: 50/200 SMA crossover; advanced: HMM)

* **Feature Fusion**

  * Merge sentiment/time features and price/TA features into a single vector
  * Normalization and missing value handling

* **Signal Engine**

  * ML model (LightGBM) predicting probability of price_up > X% in next T
  * Rules: cooldown, confidence gating, require N confirmations
  * Ensembler: weighted average between ML_pred and rule_score

* **Trading Logic & Risk**

  * Position sizing (Kelly-lite or fixed fraction)
  * Stop-loss / take-profit orders, max exposure, per-ticker limits
  * Execution manager with retry, slippage estimate, and timeout

* **Backtester / Evaluation**

  * Backtrader-based, with slippage, latency, and fees simulation
  * Performance: P&L, Sharpe, Sortino, max drawdown, hit rate

* **Monitoring & Storage**

  * Logging (structured), MLflow for model versioning
  * Streamlit dashboard for signals & P&L
  * Prometheus metrics and Grafana dashboards (optional)

---

## 4. Repo File Layout (recommended)

```
/ (root)
├─ data/                # cached price & news, small samples
├─ notebooks/           # exploration, EDA, training notebooks
├─ src/
│  ├─ inputs/
│  │  ├─ news_adapter.py
│  │  └─ market_adapter.py
│  ├─ preprocess/
│  │  ├─ text_cleaner.py
│  │  └─ ner_filter.py
│  ├─ model/
│  │  ├─ sentiment/
│  │  │  └─ finbert_ensemble.py
│  │  └─ signal_model.py
│  ├─ features/
│  │  └─ ta_features.py
│  ├─ logic/
│  │  ├─ hybrid_signal.py
│  │  └─ decision.py
│  ├─ backtest/
│  │  └─ backtest_runner.py
│  ├─ trading/
│  │  └─ executor.py
│  ├─ dashboard/
│  │  └─ streamlit_app.py
│  └─ tests/
├─ scripts/
│  └─ run_live.sh
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```

---

## 5. Detailed Implementation Milestones

### Phase 1 — Repo hygiene & CI (Week 1)

* Move configs to `.env` using `python-dotenv`.
* Add `requirements.txt`, `Dockerfile`, `docker-compose.yml`.
* Add basic unit tests and set up GitHub Actions for lint & tests.
* Deliverable: Clean repo + CI passing.

### Phase 2 — Backtesting (Week 2-3)

* Integrate Backtrader; create `backtest_runner.py` that reads historical news + price and simulates current rule pipeline.
* Implement slippage and commission simulation.
* Deliverable: Backtest report (CSV + plots) for last 12 months.

### Phase 3 — Feature Expansion (Week 4-5)

* Build `ta_features.py` to compute TA features for all tickers.
* Add NER-based relevance filter so news is mapped to correct ticker.
* Deliverable: Feature store (parquet) for model training.

### Phase 4 — Train ML Signal Model (Week 6-7)

* Create dataset builder: label = 1 if price rises > X% within T minutes/hours.
* Train LightGBM (or XGBoost). Save model with `mlflow`.
* Deliverable: Trained model + performance metrics (ROC, Precision, P&L simulation).

### Phase 5 — Ensemble & Hybrid Logic (Week 8)

* Implement `hybrid_signal.py` that weights ML + rule.
* Tune weights by backtest grid search.
* Deliverable: Hybrid rule with better Sharpe than baseline.

### Phase 6 — Risk Controls & Executor (Week 9)

* Position sizing; stop loss & take profit; safety limits.
* Implement broker wrapper (`executor.py`) with retries and order status monitor.
* Deliverable: Paper-trading run on Alpaca with safety constraints.

### Phase 7 — Monitoring & Retraining (Week 10-11)

* Streamlit dashboard with signal overlay, P&L and confusion analysis.
* Auto retrain scheduler (cron or Airflow) + model drift alerting.
* Deliverable: Live dashboard + scheduled retrain job.

### Phase 8 — Harden & Scale (Week 12+)

* Optimize for multi-ticker streaming (async, batching).
* Add Prometheus metrics + Grafana dashboards.
* Containerize and add helm/k8s manifests if needed.

---

## 6. Example Implementations (Short Snippets)

**Hybrid scorer (src/logic/hybrid_signal.py)**

```python
import numpy as np

def hybrid_score(ml_pred: float, sentiment: float, momentum: float, weights=None):
    weights = weights or {'ml': 0.6, 'sent': 0.3, 'mom': 0.1}
    rule_boost = 0.0
    if sentiment > 0.65 and momentum > 0.02:
        rule_boost = 0.15
    score = weights['ml'] * ml_pred + weights['sent'] * sentiment + weights['mom'] * momentum + rule_boost
    return score
```

**Decision & risk (src/logic/decision.py)**

```python
def decide(score, thresholds, position):
    if score > thresholds['buy'] and position == 0:
        return 'BUY'
    if score < thresholds['sell'] and position > 0:
        return 'SELL'
    return 'HOLD'
```

---

## 7. Evaluation Metrics & Backtest Settings

* Use walk-forward split; no lookahead. Label with forward horizon T (e.g. 1h, 4h, 1d).
* Include slippage = 0.05% per trade, commission (if simulating non-free broker).
* Primary KPIs: Sharpe (annualized), Max Drawdown, CAGR, Win rate, Avg profit per trade.
* Model metrics: ROC-AUC, Precision@K, Calibration (Brier score).

---

## 8. Open-source Resources & Reading

* FinBERT on HuggingFace
* Backtrader docs + examples
* LightGBM / XGBoost tutorials for time-series classification
* Loughran-McDonald Financial Sentiment Lexicon
* Papers: "News and the Stock Market" and recent works on NLP for finance

---

## 9. Next Steps (practical)

1. I'll generate the initial skeleton files for `src/` (boilerplate) if you want.
2. Choose 5 tickers and 3 months of historical news + price to run Phase 2 backtests.
3. Pick initial hyperparameters: X% = 0.8% move within T = 4 hours (changeable).

---

*End of document.*
