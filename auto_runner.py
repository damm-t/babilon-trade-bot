import datetime
import os
import time

import yfinance as yf
from broker.alpaca_client import get_alpaca_client, place_order
from config import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD
from logic.trade_decision import generate_trade_signal
from model.sentiment_model import analyze_sentiment
from visualization.chart_data import format_chart_data, to_json
from visualization.export_plot import export_candlestick

# Example news headline to simulate input (replace with real source if available)
sample_news = {
    "AAPL": "Apple beats revenue expectations amid strong iPhone sales",
    "TSLA": "Tesla shares dip after missing delivery estimates",
    "NVDA": "Nvidia announces major AI chip upgrade"
}

STOCK_SYMBOLS = ["AAPL", "TSLA", "NVDA"]

OUTPUT_DIR = os.path.join("visualization", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_auto_trader():
    client = get_alpaca_client()
    for stock in STOCK_SYMBOLS:
        news = sample_news.get(stock, "No headline available.")
        sentiment, score = analyze_sentiment(news)

        ticker = yf.Ticker(stock)
        hist = ticker.history(period="5d")
        if hist.empty:
            print(f"[{stock}] No historical data available, skipping.")
            continue
        current_price = hist["Close"].iloc[-1]
        reference_price = hist["Close"].mean()

        print(f"DEBUG -> sentiment={sentiment}, score={score}, "f"current_price={current_price}, ref_price={reference_price}")

        decision = generate_trade_signal(
            sentiment.lower(),
            score,
            positive_threshold=POSITIVE_THRESHOLD,
            negative_threshold=NEGATIVE_THRESHOLD,
            current_price=current_price,
            reference_price=reference_price
        )

        print(f"[{stock}] Sentiment: {sentiment}, Score: {score:.2f}")
        print(f"[{stock}] Price: ${current_price:.2f} | Avg: ${reference_price:.2f}")
        print(f"[{stock}] Decision: {decision}")

        # Convert price history into JSON format
        prices = []
        recent_hist = hist.tail(20)  # take last 20 rows
        for idx, row in recent_hist.iterrows():
            prices.append({
                "timestamp": idx.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            })

        signals = []
        if decision in ["BUY", "SELL"]:
            place_order(client, stock, decision)
            print(f"[{stock}] Order placed: {decision}")
            signals.append({
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "price": float(current_price),
                "type": decision
            })
        else:
            print(f"[{stock}] No trade executed.")

        # Save JSON chart data
        chart_json_path = os.path.join(OUTPUT_DIR, f"{stock}_chart.json")
        with open(chart_json_path, "w") as f:
            f.write(to_json(prices, signals))
        print(f"[{stock}] Chart data saved to {chart_json_path}")

        # Export static PNG chart
        chart_png_path = os.path.join(OUTPUT_DIR, f"{stock}_chart.png")
        export_candlestick(prices, signals, filename=chart_png_path)
        print(f"[{stock}] Chart image saved to {chart_png_path}")

        print("-" * 40)

if __name__ == "__main__":
    run_auto_trader()