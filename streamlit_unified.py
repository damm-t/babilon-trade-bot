# streamlit_unified.py - Unified Streamlit App combining both streamlit_app.py and streamlit_enhanced.py

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
from logic.portfolio import load_portfolio, upsert_holding, remove_holding, compute_portfolio_metrics
from logic.backtest import simulate_from_logs
from model.sentiment_model import analyze_sentiment

# Try to import requests, fallback gracefully if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    st.warning("⚠️ requests module not available. API features will be disabled.")

# API Configuration (for enhanced features)
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="Babilon AI Stock Trader",
    page_icon="📈",
    layout="wide"
)

# Environment Variables
ALPACA_KEY_ID = os.environ.get("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")

# Utility Functions
def log_decision(timestamp, input_text, sentiment, score, decision, stock_symbol):
    """Log trading decisions to CSV file"""
    log_entry = {
        "timestamp": timestamp,
        "input_text": input_text,
        "sentiment": sentiment,
        "score": score,
        "decision": decision,
        "stock_symbol": stock_symbol,
    }
    LOGS_DIR = "data"
    LOGS_FILE = os.path.join(LOGS_DIR, "logs.csv")
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    
    if os.path.exists(LOGS_FILE):
        df = pd.read_csv(LOGS_FILE)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv(LOGS_FILE, index=False)

def _fetch_and_save_history(symbol: str, period: str, interval: str) -> str:
    """Fetch and save historical data"""
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

def _load_csv_candles(symbol: str) -> list:
    """Load candle data from CSV file"""
    csv_path = os.path.join("data", f"{symbol.upper()}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    
    # Find time column
    time_col = None
    for c in ["Date", "Datetime", "date", "datetime", "Time", "time"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("No time column found in CSV")
    
    # Validate required columns
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

def create_candlestick_chart(chart_data, symbol, signals=None, date_range=None):
    """Create candlestick chart with trading signals"""
    if not chart_data or "candles" not in chart_data:
        return None
    
    candles = chart_data["candles"]
    if signals is None:
        signals = chart_data.get("signals", [])
    
    if not candles:
        return None
    
    # Filter candles by date range if provided
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        def _in_range(ts):
            try:
                if isinstance(ts, (int, float)):
                    d = pd.to_datetime(ts, unit='ms').date()
                else:
                    d = pd.to_datetime(ts).date()
                return (d >= start_date) and (d <= end_date)
            except:
                return True  # Keep candle if timestamp parsing fails
        candles = [c for c in candles if _in_range(c.get("time") or c.get("timestamp"))]
        # Filter signals that fall inside the range
        if signals and isinstance(signals, list) and signals and "timestamp" in signals[0]:
            signals = [s for s in signals if _in_range(s.get("timestamp"))]
    
    # Extract OHLCV data and convert timestamps to proper format
    times = []
    for candle in candles:
        timestamp = candle.get("time") or candle.get("timestamp")
        if timestamp:
            # Handle both Unix timestamp (milliseconds) and ISO format
            if isinstance(timestamp, (int, float)):
                # Convert Unix timestamp (milliseconds) to datetime
                times.append(pd.to_datetime(timestamp, unit='ms'))
            else:
                # Assume it's already a string format
                times.append(pd.to_datetime(timestamp))
    
    opens = [candle["open"] for candle in candles]
    highs = [candle["high"] for candle in candles]
    lows = [candle["low"] for candle in candles]
    closes = [candle["close"] for candle in candles]
    
    # Create candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=times,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Price"
        )
    ])
    
    # Add BUY/SELL signals
    for signal in signals:
        signal_type = signal.get("type")
        if signal_type not in ("BUY", "SELL"):
            continue
            
        # Handle both timestamped and index-based signals
        signal_time = signal.get("timestamp")
        if signal_time:
            # Find closest candle by time
            try:
                # Convert signal timestamp to datetime for comparison
                if isinstance(signal_time, (int, float)):
                    signal_dt = pd.to_datetime(signal_time, unit='ms')
                else:
                    signal_dt = pd.to_datetime(signal_time)
                
                closest_idx = min(
                    range(len(times)),
                    key=lambda i: abs(times[i] - signal_dt),
                )
            except Exception:
                continue
        else:
            closest_idx = signal.get("index")
            if not isinstance(closest_idx, int) or closest_idx < 0 or closest_idx >= len(times):
                continue
        
        if closest_idx is None or closest_idx < 0 or closest_idx >= len(times):
            continue
        
        color = "green" if signal_type == "BUY" else "red"
        symbol_marker = "arrow-up" if signal_type == "BUY" else "arrow-down"
        price = signal.get("price", closes[closest_idx])
        
        fig.add_trace(go.Scatter(
            x=[times[closest_idx]],
            y=[price],
            mode="markers+text",
            marker=dict(symbol=symbol_marker, color=color, size=16),
            text=[signal_type],
            textposition="top center" if signal_type == "BUY" else "bottom center",
            name=signal_type,
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{symbol} Stock Chart with Trading Signals",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=500
    )
    
    return fig

# API Functions (for enhanced features)
def check_api_connection():
    """Check if API is available"""
    if not REQUESTS_AVAILABLE:
        return False
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def search_stocks_api(query):
    """Search for stocks using the API"""
    if not REQUESTS_AVAILABLE:
        st.error("API functionality not available - requests module missing")
        return []
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/search", params={"query": query}, timeout=5)
        if response.status_code == 200:
            return response.json()["stocks"]
        return []
    except Exception as e:
        st.error(f"Error searching stocks: {e}")
        return []

def get_stock_data_api(symbol):
    """Get stock data from API"""
    if not REQUESTS_AVAILABLE:
        st.error("API functionality not available - requests module missing")
        return None
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/{symbol}/data", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def analyze_stock_api(symbol, positive_threshold, negative_threshold, enable_trading):
    """Analyze stock using API"""
    if not REQUESTS_AVAILABLE:
        st.error("API functionality not available - requests module missing")
        return None
    try:
        payload = {
            "symbol": symbol,
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
            "enable_trading": enable_trading
        }
        response = requests.post(f"{API_BASE_URL}/stocks/analyze", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error analyzing stock: {e}")
        return None

def analyze_custom_news_api(symbol, news_text, positive_threshold, negative_threshold, enable_trading):
    """Analyze custom news using API"""
    if not REQUESTS_AVAILABLE:
        st.error("API functionality not available - requests module missing")
        return None
    try:
        payload = {
            "symbol": symbol,
            "news_text": news_text,
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
            "enable_trading": enable_trading
        }
        response = requests.post(f"{API_BASE_URL}/news/analyze", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"News analysis failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error analyzing news: {e}")
        return None

# Main App
st.title("📈 Babilon AI Stock Sentiment Trader")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# API Connection Check
api_available = check_api_connection()
if api_available:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.warning("⚠️ API Not Available - Using Direct Mode")

# Trading Settings
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

# Data Fetch Controls
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

with st.sidebar.expander("ℹ️ How to Use"):
    st.write("""
    - **Demo Mode** → safe, no trades are sent.
    - **Live Mode** → requires valid Alpaca API keys.
    - **API Mode** → uses REST API for enhanced features.
    - **Direct Mode** → uses local models directly.
    - Paste a news headline → AI will analyze sentiment.
    - Select stocks → system generates BUY/SELL/HOLD signals.
    - Candlestick charts → show recent data + trade signals.
    - Logs → last 10 actions are listed below.
    """)

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Stock Analysis", "📰 News Analysis", "📊 Portfolio View", "📈 Direct Trading"])

# Initialize client
client = None
if trade_mode == "Live":
    if ALPACA_KEY_ID and ALPACA_SECRET_KEY:
        client = get_alpaca_client()
    else:
        st.sidebar.warning("⚠️ Alpaca keys missing. Switching to Demo mode.")
        trade_mode = "Demo"

# Tab 1: Stock Analysis (Enhanced API-based)
with tab1:
    st.header("Stock Analysis & Trading")
    
    if api_available:
        # API-based stock analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input("Search for stocks:", placeholder="e.g., AAPL, Tesla, Microsoft")
        
        with col2:
            if st.button("Search", type="primary"):
                if search_query:
                    with st.spinner("Searching stocks..."):
                        stocks = search_stocks_api(search_query)
                        if stocks:
                            st.session_state.search_results = stocks
                        else:
                            st.warning("No stocks found. Try a different search term.")
        
        # Display search results
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            selected_stock = st.selectbox("Select a stock:", st.session_state.search_results)
            
            if selected_stock:
                # Get stock data
                with st.spinner(f"Fetching data for {selected_stock}..."):
                    stock_data = get_stock_data_api(selected_stock)
                
                if stock_data:
                    # Display stock information
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${stock_data['current_price']:.2f}")
                    with col2:
                        st.metric("Reference Price", f"${stock_data['reference_price']:.2f}")
                    with col3:
                        change = stock_data['price_change']
                        st.metric("Price Change", f"${change:.2f}", f"{stock_data['price_change_percent']:.2f}%")
                    with col4:
                        st.metric("Volume", f"{stock_data['volume']:,}")
                    
                    # Analyze button
                    if st.button("🤖 Analyze Stock", type="primary"):
                        with st.spinner("Analyzing stock sentiment and generating trading signal..."):
                            analysis = analyze_stock_api(
                                selected_stock, 
                                pos_threshold, 
                                neg_threshold, 
                                trade_mode == "Live"
                            )
                        
                        if analysis:
                            # Display analysis results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                sentiment_color = "green" if analysis['sentiment'] == "positive" else "red" if analysis['sentiment'] == "negative" else "orange"
                                st.markdown(f"**Sentiment:** :{sentiment_color}[{analysis['sentiment'].upper()}]")
                            
                            with col2:
                                st.markdown(f"**Confidence:** {analysis['score']:.2f}")
                            
                            with col3:
                                decision_color = "green" if analysis['decision'] == "BUY" else "red" if analysis['decision'] == "SELL" else "gray"
                                st.markdown(f"**Decision:** :{decision_color}[{analysis['decision']}]")
                            
                            # Display news articles
                            if analysis.get('news_articles'):
                                st.subheader("📰 Recent News")
                                for article in analysis['news_articles'][:5]:  # Show top 5
                                    with st.expander(f"{article['title']} - {article['source']}"):
                                        st.write(article['description'])
                                        st.markdown(f"[Read more]({article['url']})")
                            
                            # Display chart
                            if analysis.get('chart_data'):
                                st.subheader("📊 Price Chart with Signals")
                                fig = create_candlestick_chart(analysis['chart_data'], selected_stock, date_range=date_range)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Unable to create chart")
    else:
        st.warning("API not available. Please use the Direct Trading tab or start the API server: `python api.py`")

# Tab 2: News Analysis (Enhanced API-based)
with tab2:
    st.header("Custom News Analysis")
    
    if api_available:
        # Stock selection for news analysis
        news_stock = st.text_input("Enter stock symbol:", placeholder="e.g., AAPL", key="news_stock")
        
        # News input
        news_text = st.text_area(
            "Enter news headline or article:",
            placeholder="e.g., Apple reports record quarterly earnings...",
            height=100
        )
        
        if st.button("📰 Analyze News", type="primary"):
            if news_stock and news_text:
                with st.spinner("Analyzing news sentiment..."):
                    analysis = analyze_custom_news_api(
                        news_stock.upper(),
                        news_text,
                        pos_threshold,
                        neg_threshold,
                        trade_mode == "Live"
                    )
                
                if analysis:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sentiment_color = "green" if analysis['sentiment'] == "positive" else "red" if analysis['sentiment'] == "negative" else "orange"
                        st.markdown(f"**Sentiment:** :{sentiment_color}[{analysis['sentiment'].upper()}]")
                    
                    with col2:
                        st.markdown(f"**Confidence:** {analysis['score']:.2f}")
                    
                    with col3:
                        decision_color = "green" if analysis['decision'] == "BUY" else "red" if analysis['decision'] == "SELL" else "gray"
                        st.markdown(f"**Decision:** :{decision_color}[{analysis['decision']}]")
                    
                    # Display stock info
                    if analysis.get('current_price'):
                        st.subheader("📊 Stock Information")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Price", f"${analysis['current_price']:.2f}")
                        with col2:
                            st.metric("Reference Price", f"${analysis['reference_price']:.2f}")
                    
                    # Display chart
                    if analysis.get('chart_data'):
                        st.subheader("📈 Price Chart with Signals")
                        fig = create_candlestick_chart(analysis['chart_data'], news_stock.upper(), date_range=date_range)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter both stock symbol and news text.")
    else:
        st.warning("API not available. Please use the Direct Trading tab or start the API server: `python api.py`")

# Tab 3: Portfolio View (Combined from both apps)
with tab3:
    st.header("Portfolio & Trading History")
    
    # Portfolio Section
    st.subheader("Holdings")
    with st.form("add_holding_form", clear_on_submit=True):
        col_a, col_b, col_c, col_d = st.columns([1,1,1,2])
        with col_a:
            sym_in = st.text_input("Symbol", placeholder="AAPL")
        with col_b:
            qty_in = st.number_input("Quantity", min_value=0.0, step=1.0)
        with col_c:
            price_in = st.number_input("Avg Price", min_value=0.0, step=0.01, format="%.2f")
        with col_d:
            notes_in = st.text_input("Notes", placeholder="Optional")
        submitted = st.form_submit_button("Add / Update Holding")
    
    if submitted and sym_in and qty_in >= 0 and price_in >= 0:
        try:
            upsert_holding(sym_in.upper(), float(qty_in), float(price_in), notes_in)
            st.success("Holding saved.")
        except Exception as e:
            st.error(f"Failed to save holding: {e}")

    try:
        holdings = load_portfolio()
    except Exception:
        holdings = []
    
    if holdings:
        # Fetch current prices for metrics table
        prices_map = {}
        for h in holdings:
            try:
                if api_available:
                    data = get_stock_data_api(h.symbol)
                    if data and "current_price" in data:
                        prices_map[h.symbol] = data["current_price"]
                else:
                    # Fallback to yfinance
                    ticker = yf.Ticker(h.symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        prices_map[h.symbol] = hist['Close'].iloc[-1]
            except Exception:
                continue
        
        df_metrics = compute_portfolio_metrics(prices_map)
        if not df_metrics.empty:
            st.dataframe(df_metrics, use_container_width=True)
            st.caption("Unrealized P&L based on current market price.")
        
        # Remove holding UI
        with st.expander("Remove a holding"):
            to_remove = st.selectbox("Select symbol to remove", [h.symbol for h in holdings])
            if st.button("Remove"):
                remove_holding(to_remove)
                st.success(f"Removed {to_remove}")
    else:
        st.info("No holdings yet. Add one above.")

    st.markdown("---")
    
    # Backtest Section
    st.subheader("Backtest from Logs")
    coll, colr = st.columns([1,3])
    with coll:
        days = st.number_input("Lookback days", min_value=7, max_value=365, value=30)
        run_bt = st.button("Run Backtest")
    
    if run_bt:
        with st.spinner("Simulating trades from logs..."):
            trades_df, summary = simulate_from_logs(int(days))
        st.markdown("**Summary**")
        st.metric("Win Rate", f"{summary.get('win_rate',0):.1f}%")
        st.metric("Avg Return / Trade", f"{summary.get('avg_return_pct',0):.2f}%")
        st.metric("Total Return (realized)", f"{summary.get('total_return_pct',0):.2f}%")
        st.caption(f"Trades counted: {summary.get('num_trades',0)}")
        if not trades_df.empty:
            st.subheader("Executed Trades")
            st.dataframe(trades_df, use_container_width=True)
    
    st.markdown("---")
    
    # Logs Section
    logs_file = "data/logs.csv"
    st.subheader("Recent Trading Activity")
    if os.path.exists(logs_file):
        try:
            df_logs = pd.read_csv(logs_file)
            st.dataframe(df_logs.tail(10), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load trading logs: {e}")
    else:
        st.info("No trading logs available yet.")

# Tab 4: Direct Trading (Original streamlit_app.py functionality)
with tab4:
    st.header("Direct Trading (Local Models)")
    
    news_input = st.text_area("Paste financial news or earnings headline:", "")
    
    selected_stocks = st.sidebar.multiselect(
        "Select stocks to analyze and trade:",
        options=STOCK_SYMBOLS,
        default=STOCK_SYMBOLS[:1],
    )
    
    # Auto-fetch functionality
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

                # Visualization Section
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
                        chart_data = {"candles": candles, "signals": signals}
                        fig = create_candlestick_chart(chart_data, stock, signals=signals, date_range=date_range)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not create chart")
                            
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

# Sidebar Recent Logs
st.sidebar.subheader("📜 Recent Logs")
LOGS_FILE = os.path.join("data", "logs.csv")
if os.path.exists(LOGS_FILE):
    try:
        df_logs = pd.read_csv(LOGS_FILE)
        if not df_logs.empty:
            st.sidebar.dataframe(df_logs.tail(10))
        else:
            st.sidebar.info("No logs yet.")
    except Exception:
        st.sidebar.info("No logs yet.")
else:
    st.sidebar.info("No logs yet.")

# Footer
st.markdown("---")
st.markdown("**Babilon AI Stock Trader** - Unified app combining direct model access and API-based features")
