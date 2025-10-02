# streamlit_app.py

import datetime
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from broker.alpaca_client import get_alpaca_client, place_order
from config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD, STOCK_SYMBOLS
from logic.trade_decision import generate_trade_signal
from model.sentiment_model import analyze_sentiment

ALPACA_KEY_ID = os.environ.get("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")

# Sidebar controls
st.sidebar.header("⚙️ Settings")

enable_alpaca_trading = st.sidebar.checkbox("Enable Alpaca Trading", value=False)

if enable_alpaca_trading:
    ALPACA_KEY_ID = st.sidebar.text_input("Alpaca API Key ID", value=ALPACA_KEY_ID or "", type="password")
    ALPACA_SECRET_KEY = st.sidebar.text_input("Alpaca Secret Key", value=ALPACA_SECRET_KEY or "", type="password")
else:
    ALPACA_KEY_ID = None
    ALPACA_SECRET_KEY = None

force_test_signal = st.sidebar.checkbox(
    "Force Test Signal", value=os.environ.get("FORCE_TEST_SIGNAL", "False") == "True"
)
pos_threshold = st.sidebar.slider(
    "Positive Threshold",
    0.5,
    0.9,
    float(os.environ.get("POSITIVE_THRESHOLD", 0.65)),
    0.01,
)
neg_threshold = st.sidebar.slider(
    "Negative Threshold",
    0.5,
    0.9,
    float(os.environ.get("NEGATIVE_THRESHOLD", 0.7)),
    0.01,
)
trade_mode = st.sidebar.radio("Trading Mode:", ["Demo", "Live"])

if not enable_alpaca_trading:
    trade_mode = "Demo"
elif trade_mode == "Live":
    if ALPACA_KEY_ID and ALPACA_SECRET_KEY:
        pass  # keep Live mode
    else:
        st.sidebar.warning("⚠️ Alpaca keys missing. Switching to Demo mode.")
        trade_mode = "Demo"

with st.sidebar.expander("ℹ️ How to Use"):
    st.write("""
    - **Demo Mode** → safe, no trades are sent.
    - **Live Mode** → requires valid Alpaca API keys.
    - Paste a news headline → AI will analyze sentiment.
    - Select stocks → system generates BUY/SELL/HOLD signals.
    - Candlestick charts → show recent data + trade signals.
    - Logs → last 10 actions are listed below.
    """)
    
client = None
if trade_mode == "Live":
    if ALPACA_KEY_ID and ALPACA_SECRET_KEY:
        client = get_alpaca_client()
    else:
        st.sidebar.warning("⚠️ Alpaca keys missing. Switching to Demo mode.")
        trade_mode = "Demo"

st.title("📈 Babilon AI Stock Sentiment Trader")

news_input = st.text_area("Paste financial news or earnings headline:", "")

selected_stocks = st.sidebar.multiselect(
    "Select stocks to analyze and trade:",
    options=STOCK_SYMBOLS,
    default=STOCK_SYMBOLS[:1],
)

# Optional: fetching controls for historical data saving
st.sidebar.subheader("📥 Historical Data Fetch")
fetch_period = st.sidebar.selectbox(
    "Period",
    options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=3,
)
fetch_interval = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1h", "30m", "15m"],
    index=0,
)
auto_fetch_on_load = st.sidebar.checkbox("Auto-fetch OHLCV on load", value=False)
fetch_button = st.sidebar.button("Fetch latest OHLCV to data/")

# Date range filter for visualization
today = datetime.date.today()
default_start = today - datetime.timedelta(days=30)
date_range = st.sidebar.date_input(
    "Chart date range",
    value=(default_start, today),
)

LOGS_DIR = "data"
LOGS_FILE = os.path.join(LOGS_DIR, "logs.csv")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


def log_decision(timestamp, input_text, sentiment, score, decision, stock_symbol):
    log_entry = {
        "timestamp": timestamp,
        "input_text": input_text,
        "sentiment": sentiment,
        "score": score,
        "decision": decision,
        "stock_symbol": stock_symbol,
    }
    if os.path.exists(LOGS_FILE):
        df = pd.read_csv(LOGS_FILE)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv(LOGS_FILE, index=False)


def _fetch_and_save_history(symbol: str, period: str, interval: str) -> str:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist is None or hist.empty:
        raise ValueError(f"No data returned for {symbol}")
    hist = hist.reset_index()
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol.upper()}.csv")
    hist.to_csv(out_path, index=False)
    return out_path


# Load candles from CSV if available
def _load_csv_candles(symbol: str) -> list:
    csv_path = os.path.join("data", f"\{symbol.upper()}\.csv").replace("\\", "").replace(" ", " ")
    csv_path = os.path.join("data", f"{symbol.upper()}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    # Try common yfinance columns
    time_col = None
    for c in ["Date", "Datetime", "date", "datetime", "Time", "time"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("No time column found in CSV")
    required = ["Open", "High", "Low", "Close"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in CSV")
    candles = []
    for _, row in df.iterrows():
        ts = pd.to_datetime(row[time_col]).isoformat()
        candles.append({
            "timestamp": ts,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        })
    return candles


# Perform auto-fetch once per session when enabled
if auto_fetch_on_load and selected_stocks:
    if "__auto_fetch_done__" not in st.session_state:
        st.session_state["__auto_fetch_done__"] = True
        for s in selected_stocks:
            try:
                path = _fetch_and_save_history(s, fetch_period, fetch_interval)
                st.sidebar.success(f"Saved {s}: {path}")
            except Exception as e:
                st.sidebar.warning(f"Fetch failed for {s}: {e}")

# Manual fetch button
if fetch_button:
    if not selected_stocks:
        st.sidebar.warning("Please select at least one stock to fetch.")
    else:
        for s in selected_stocks:
            try:
                path = _fetch_and_save_history(s, fetch_period, fetch_interval)
                st.sidebar.success(f"Saved {s}: {path}")
            except Exception as e:
                st.sidebar.warning(f"Fetch failed for {s}: {e}")


if st.button("Analyze & Trade"):
    if not news_input.strip():
        st.warning("Please enter a news headline.")
    elif not selected_stocks:
        st.warning("Please select at least one stock.")
    else:
        if trade_mode == "Live" and enable_alpaca_trading and ALPACA_KEY_ID and ALPACA_SECRET_KEY:
            client = get_alpaca_client()
        else:
            client = None
            if trade_mode == "Live":
                st.sidebar.warning("⚠️ Alpaca keys missing or Alpaca Trading disabled. Running in Demo mode.")
                trade_mode = "Demo"

        for stock in selected_stocks:
            sentiment, score = analyze_sentiment(news_input)
            decision = generate_trade_signal(
                sentiment.lower(),
                score,
                positive_threshold=pos_threshold,
                negative_threshold=neg_threshold,
            )

            timestamp = datetime.datetime.now().isoformat()
            log_decision(timestamp, news_input, sentiment, score, decision, stock)

            st.markdown(f"### Stock: {stock}")
            st.markdown(f"**Sentiment:** {sentiment}")
            st.markdown(f"**Confidence Score:** {score:.2f}")
            st.markdown(f"**Decision:** `{decision}`")

            if decision in ["BUY", "SELL"]:
                if trade_mode == "Live" and client:
                    place_order(client, stock, decision)
                    st.success(f"{decision} order placed for {stock} (Live)")
                else:
                    st.info(f"Simulated {decision} for {stock} (Demo mode)")
            else:
                st.info("No trade executed.")

            # --- Visualization Section ---
            chart_json_path = f"visualization/output/{stock}_chart.json"
            try:
                # Prefer CSV from data/ if available
                try:
                    candles = _load_csv_candles(stock)
                    signals = []
                except Exception:
                    with open(chart_json_path, "r") as f:
                        chart_data = json.load(f)
                    candles = chart_data.get("candles", [])
                    signals = chart_data.get("signals", [])
                # If force_test_signal is True, append a fake BUY marker
                if force_test_signal:
                    if candles:
                        fake_index = len(candles) // 2
                        signals.append({"type": "BUY", "index": fake_index})
                if not candles:
                    st.warning("No candle data available for visualization.")
                else:
                    # Filter candles by selected date range if provided
                    try:
                        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                            start_date, end_date = date_range
                            def _in_range(ts):
                                d = pd.to_datetime(ts).date()
                                return (d >= start_date) and (d <= end_date)
                            candles = [c for c in candles if _in_range(c.get("time") or c.get("timestamp"))]
                            # reattach only signals that fall inside the range when timestamp is present
                            if signals and isinstance(signals, list) and "timestamp" in signals[0]:
                                signals = [s for s in signals if _in_range(s.get("timestamp"))]
                    except Exception:
                        pass

                    # Parse OHLCV
                    times = [c.get("time") or c.get("timestamp") for c in candles]
                    opens = [c["open"] for c in candles]
                    highs = [c["high"] for c in candles]
                    lows = [c["low"] for c in candles]
                    closes = [c["close"] for c in candles]
                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=times,
                                open=opens,
                                high=highs,
                                low=lows,
                                close=closes,
                                name="Candles",
                            )
                        ]
                    )
                    # Overlay BUY/SELL markers
                    for sig in signals:
                        typ = sig.get("type")
                        if typ not in ("BUY", "SELL"):
                            continue
                        # Prefer timestamped signals; fallback to index-based
                        ts = sig.get("timestamp")
                        if ts:
                            # Find closest candle by time
                            try:
                                closest_idx = min(
                                    range(len(times)),
                                    key=lambda i: abs(pd.to_datetime(times[i]) - pd.to_datetime(ts)),
                                )
                            except Exception:
                                closest_idx = None
                        else:
                            closest_idx = sig.get("index")
                            if not isinstance(closest_idx, int):
                                closest_idx = None

                        if closest_idx is None or closest_idx < 0 or closest_idx >= len(times):
                            continue

                        color = "green" if typ == "BUY" else "red"
                        symbol = "arrow-up" if typ == "BUY" else "arrow-down"
                        price = closes[closest_idx]
                        fig.add_trace(
                            go.Scatter(
                                x=[times[closest_idx]],
                                y=[price],
                                mode="markers+text",
                                marker=dict(symbol=symbol, color=color, size=16),
                                text=[typ],
                                textposition="top center" if typ == "BUY" else "bottom center",
                                name=typ,
                            )
                        )
                    fig.update_layout(
                        title=f"{stock} Candlestick Chart with Signals",
                        xaxis_title="Time",
                        yaxis_title="Price",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except FileNotFoundError:
                st.warning(f"No chart data found for {stock}. Showing demo data instead.")
                demo_times = pd.date_range(end=datetime.datetime.now(), periods=10).tolist()
                demo_prices = [100 + i + (i % 3) * 2 for i in range(10)]
                fig = go.Figure(
                    data=[go.Candlestick(
                        x=demo_times,
                        open=demo_prices,
                        high=[p+3 for p in demo_prices],
                        low=[p-3 for p in demo_prices],
                        close=demo_prices,
                    )]
                )
                fig.add_trace(go.Scatter(
                    x=[demo_times[5]],
                    y=[demo_prices[5]],
                    mode="markers+text",
                    marker=dict(symbol="arrow-up", color="green", size=16),
                    text=["BUY"],
                    textposition="top center"
                ))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render chart: {e}")
            st.sidebar.subheader("📜 Recent Logs")
if os.path.exists(LOGS_FILE):
    df_logs = pd.read_csv(LOGS_FILE)
    st.sidebar.dataframe(df_logs.tail(10))
else:
    st.sidebar.info("No logs yet.")
