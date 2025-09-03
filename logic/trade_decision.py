def generate_trade_signal(
    sentiment,
    score,
    positive_threshold=0.65,
    negative_threshold=0.7,
    current_price=None,
    reference_price=None
):
    if sentiment == "positive" and score > positive_threshold:
        if reference_price and current_price and current_price < reference_price:
            return "BUY"
    elif sentiment == "negative" and score > negative_threshold:
        if reference_price and current_price and current_price > reference_price:
            return "SELL"
    return "HOLD"