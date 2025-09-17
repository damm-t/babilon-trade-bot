import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from logic.portfolio import load_portfolio, upsert_holding, remove_holding, compute_portfolio_metrics
from logic.backtest import simulate_from_logs

# API Configuration
API_BASE_URL = "http://localhost:8000"

def search_stocks(query):
    """Search for stocks using the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/search", params={"query": query})
        if response.status_code == 200:
            return response.json()["stocks"]
        return []
    except Exception as e:
        st.error(f"Error searching stocks: {e}")
        return []

def get_stock_data(symbol):
    """Get stock data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stocks/{symbol}/data")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def analyze_stock(symbol, positive_threshold, negative_threshold, enable_trading):
    """Analyze stock using API"""
    try:
        payload = {
            "symbol": symbol,
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
            "enable_trading": enable_trading
        }
        response = requests.post(f"{API_BASE_URL}/stocks/analyze", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error analyzing stock: {e}")
        return None

def analyze_custom_news(symbol, news_text, positive_threshold, negative_threshold, enable_trading):
    """Analyze custom news using API"""
    try:
        payload = {
            "symbol": symbol,
            "news_text": news_text,
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
            "enable_trading": enable_trading
        }
        response = requests.post(f"{API_BASE_URL}/news/analyze", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"News analysis failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error analyzing news: {e}")
        return None

def create_candlestick_chart(chart_data, symbol):
    """Create candlestick chart from chart data"""
    if not chart_data or "candles" not in chart_data:
        return None
    
    candles = chart_data["candles"]
    signals = chart_data.get("signals", [])
    
    if not candles:
        return None
    
    # Extract OHLCV data
    times = [candle["timestamp"] for candle in candles]
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
        signal_time = signal.get("timestamp")
        signal_type = signal.get("type")
        signal_price = signal.get("price")
        
        if signal_time and signal_type and signal_price:
            # Find closest candle time
            closest_idx = min(range(len(times)), key=lambda i: abs(pd.to_datetime(times[i]) - pd.to_datetime(signal_time)))
            
            color = "green" if signal_type == "BUY" else "red"
            symbol_marker = "arrow-up" if signal_type == "BUY" else "arrow-down"
            
            fig.add_trace(go.Scatter(
                x=[times[closest_idx]],
                y=[signal_price],
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

# Main Streamlit App
st.set_page_config(
    page_title="Babilon AI Stock Trader",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Babilon AI Stock Sentiment Trader")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# API Connection Check
try:
    response = requests.get(f"{API_BASE_URL}/")
    if response.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Connection Failed")
except:
    st.sidebar.error("❌ API Not Available")
    st.error("Please start the API server: `python api.py`")
    st.stop()

# Trading Settings
enable_trading = st.sidebar.checkbox("Enable Live Trading", value=False)
positive_threshold = st.sidebar.slider("Positive Sentiment Threshold", 0.5, 0.9, 0.65, 0.01)
negative_threshold = st.sidebar.slider("Negative Sentiment Threshold", 0.5, 0.9, 0.7, 0.01)

# Main Content
tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📰 News Analysis", "📊 Portfolio View"])

with tab1:
    st.header("Stock Analysis & Trading")
    
    # Stock Search and Selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input("Search for stocks:", placeholder="e.g., AAPL, Tesla, Microsoft")
    
    with col2:
        if st.button("Search", type="primary"):
            if search_query:
                with st.spinner("Searching stocks..."):
                    stocks = search_stocks(search_query)
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
                stock_data = get_stock_data(selected_stock)
            
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
                        analysis = analyze_stock(
                            selected_stock, 
                            positive_threshold, 
                            negative_threshold, 
                            enable_trading
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
                            fig = create_candlestick_chart(analysis['chart_data'], selected_stock)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Unable to create chart")

with tab2:
    st.header("Custom News Analysis")
    
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
                analysis = analyze_custom_news(
                    news_stock.upper(),
                    news_text,
                    positive_threshold,
                    negative_threshold,
                    enable_trading
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
                    fig = create_candlestick_chart(analysis['chart_data'], news_stock.upper())
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter both stock symbol and news text.")

with tab3:
    st.header("Portfolio & Trading History")
    
    # --- Portfolio Section ---
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

    holdings = load_portfolio()
    if holdings:
        # Fetch current prices via API for metrics table
        prices_map = {}
        for h in holdings:
            try:
                data = get_stock_data(h.symbol)
                if data and "current_price" in data:
                    prices_map[h.symbol] = data["current_price"]
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
    
    # --- Backtest Section ---
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
    
    # --- Logs Section ---
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

# Footer
st.markdown("---")
st.markdown("**Babilon AI Stock Trader** - Powered by FinBERT sentiment analysis and Alpaca trading API")
