# streamlit_app.py

import os
import datetime
import pandas as pd
import streamlit as st
from broker.alpaca_client import get_alpaca_client, place_order
from config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD, STOCK_SYMBOLS
from logic.trade_decision import generate_trade_signal
from model.sentiment_model import analyze_sentiment

st.title("📈 Babilon AI Stock Sentiment Trader")

news_input = st.text_area("Paste financial news or earnings headline:", "")

selected_stocks = st.multiselect(
    "Select stocks to analyze and trade:",
    options=STOCK_SYMBOLS,
    default=STOCK_SYMBOLS[:1]
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
        df = df.append(log_entry, ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv(LOGS_FILE, index=False)

if st.button("Analyze & Trade"):
    if not news_input.strip():
        st.warning("Please enter a news headline.")
    elif not selected_stocks:
        st.warning("Please select at least one stock.")
    else:
        client = get_alpaca_client()
        for stock in selected_stocks:
            sentiment, score = analyze_sentiment(news_input)
            decision = generate_trade_signal(
                sentiment.lower(),
                score,
                POSITIVE_THRESHOLD=POSITIVE_THRESHOLD,
                NEGATIVE_THRESHOLD=NEGATIVE_THRESHOLD,
            )

            timestamp = datetime.datetime.now().isoformat()
            log_decision(timestamp, news_input, sentiment, score, decision, stock)

            st.markdown(f"### Stock: {stock}")
            st.markdown(f"**Sentiment:** {sentiment}")
            st.markdown(f"**Confidence Score:** {score:.2f}")
            st.markdown(f"**Decision:** `{decision}`")

            if decision in ["BUY", "SELL"]:
                place_order(client, stock, decision)
                st.success(f"{decision} order placed for {stock}")
            else:
                st.info("No trade executed.")