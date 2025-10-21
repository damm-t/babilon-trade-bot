"""Tests for trade decision logic."""

import pytest
from logic.trade_decision import generate_trade_signal


class TestTradeDecision:
    """Test trade decision logic."""

    def test_buy_signal_positive_sentiment(self):
        """Test BUY signal for positive sentiment above threshold."""
        signal = generate_trade_signal("positive", 0.8, 0.65, 0.7)
        assert signal == "BUY"

    def test_sell_signal_negative_sentiment(self):
        """Test SELL signal for negative sentiment above threshold."""
        signal = generate_trade_signal("negative", 0.8, 0.65, 0.7)
        assert signal == "SELL"

    def test_hold_signal_below_threshold(self):
        """Test HOLD signal when confidence below threshold."""
        signal = generate_trade_signal("positive", 0.5, 0.65, 0.7)
        assert signal == "HOLD"

    def test_hold_signal_neutral_sentiment(self):
        """Test HOLD signal for neutral sentiment."""
        signal = generate_trade_signal("neutral", 0.8, 0.65, 0.7)
        assert signal == "HOLD"

    def test_edge_case_thresholds(self):
        """Test edge cases at threshold boundaries."""
        # Exactly at positive threshold
        signal = generate_trade_signal("positive", 0.65, 0.65, 0.7)
        assert signal == "BUY"
        
        # Just below positive threshold
        signal = generate_trade_signal("positive", 0.649, 0.65, 0.7)
        assert signal == "HOLD"
        
        # Exactly at negative threshold
        signal = generate_trade_signal("negative", 0.7, 0.65, 0.7)
        assert signal == "SELL"
        
        # Just below negative threshold
        signal = generate_trade_signal("negative", 0.699, 0.65, 0.7)
        assert signal == "HOLD"

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Invalid sentiment
        signal = generate_trade_signal("invalid", 0.8, 0.65, 0.7)
        assert signal == "HOLD"
        
        # Invalid confidence score
        signal = generate_trade_signal("positive", -0.1, 0.65, 0.7)
        assert signal == "HOLD"
        
        signal = generate_trade_signal("positive", 1.1, 0.65, 0.7)
        assert signal == "HOLD"

    def test_case_insensitive_sentiment(self):
        """Test that sentiment is case insensitive."""
        signal1 = generate_trade_signal("POSITIVE", 0.8, 0.65, 0.7)
        signal2 = generate_trade_signal("positive", 0.8, 0.65, 0.7)
        assert signal1 == signal2 == "BUY"
        
        signal1 = generate_trade_signal("NEGATIVE", 0.8, 0.65, 0.7)
        signal2 = generate_trade_signal("negative", 0.8, 0.65, 0.7)
        assert signal1 == signal2 == "SELL"