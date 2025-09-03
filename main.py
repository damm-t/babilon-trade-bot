from broker.alpaca_client import get_alpaca_client, place_order
from config import STOCK_SYMBOL, TRADE_THRESHOLD
from logic.trade_decision import generate_trade_signal
from model.sentiment_model import analyze_sentiment


def run_bot(news_text):
    sentiment, score = analyze_sentiment(news_text)
    decision = generate_trade_signal(sentiment.lower(), score, threshold=TRADE_THRESHOLD)
    print(f"[ANALYSIS] {sentiment} @ {score:.2f} → {decision}")

    if decision in ["BUY", "SELL"]:
        client = get_alpaca_client()
        place_order(client, STOCK_SYMBOL, decision)
        print(f"[TRADE] {decision} order placed for {STOCK_SYMBOL}.")
    else:
        print("[TRADE] No trade executed.")

if __name__ == "__main__":
    example_news = "Nvidia shares soar after smashing Wall Street estimates."
    run_bot(example_news)