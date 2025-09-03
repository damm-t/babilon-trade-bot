import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.trade_decision import generate_trade_signal


def test_buy_signal():
    sentiment = "positive"
    score = 0.9
    decision = generate_trade_signal(
        sentiment,
        score,
        positive_threshold=0.65,
        current_price=90,
        reference_price=100
    )
    assert decision == "BUY"

def test_sell_signal():
    sentiment = "negative"
    score = 0.9
    decision = generate_trade_signal(
        sentiment,
        score,
        negative_threshold=0.7,
        current_price=110,
        reference_price=100
    )
    assert decision == "SELL"

def test_hold_signal_low_score():
    sentiment = "positive"
    score = 0.5
    decision = generate_trade_signal(
        sentiment,
        score,
        positive_threshold=0.65,
        current_price=90,
        reference_price=100
    )
    assert decision == "HOLD"

def test_hold_signal_price_not_favorable():
    sentiment = "positive"
    score = 0.9
    decision = generate_trade_signal(
        sentiment,
        score,
        positive_threshold=0.65,
        current_price=110,
        reference_price=100
    )
    assert decision == "HOLD"