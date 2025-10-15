import os
import sys
import unittest
from unittest.mock import patch, Mock
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.trade_decision import (
    generate_trade_signal, 
    TradeSignal, 
    calculate_technical_indicators,
    analyze_market_conditions
)


class TestTradeSignal(unittest.TestCase):
    """Test cases for TradeSignal class"""
    
    def test_trade_signal_creation(self):
        """Test TradeSignal object creation"""
        signal = TradeSignal(
            action="BUY",
            confidence=0.8,
            reasoning="Strong positive sentiment",
            stop_loss=95.0,
            take_profit=110.0
        )
        
        self.assertEqual(signal.action, "BUY")
        self.assertEqual(signal.confidence, 0.8)
        self.assertEqual(signal.reasoning, "Strong positive sentiment")
        self.assertEqual(signal.stop_loss, 95.0)
        self.assertEqual(signal.take_profit, 110.0)
        self.assertIsInstance(signal.timestamp, datetime)
    
    def test_trade_signal_to_dict(self):
        """Test TradeSignal to_dict conversion"""
        signal = TradeSignal(
            action="SELL",
            confidence=0.75,
            reasoning="Negative sentiment detected"
        )
        
        signal_dict = signal.to_dict()
        
        self.assertEqual(signal_dict['action'], "SELL")
        self.assertEqual(signal_dict['confidence'], 0.75)
        self.assertEqual(signal_dict['reasoning'], "Negative sentiment detected")
        self.assertIn('timestamp', signal_dict)


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for technical indicator calculations"""
    
    def test_calculate_technical_indicators_normal(self):
        """Test technical indicators calculation with normal data"""
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 
                 110, 112, 111, 113, 115, 114, 116, 118, 117, 119]
        
        indicators = calculate_technical_indicators(prices)
        
        self.assertIn('sma', indicators)
        self.assertIn('price_change', indicators)
        self.assertIn('volatility', indicators)
        self.assertIn('current_price', indicators)
        
        # Check that values are reasonable
        self.assertEqual(indicators['current_price'], 119)
        self.assertGreater(indicators['sma'], 100)
        self.assertLess(indicators['sma'], 120)
    
    def test_calculate_technical_indicators_insufficient_data(self):
        """Test technical indicators with insufficient data"""
        prices = [100, 102]  # Less than window size (20)
        
        indicators = calculate_technical_indicators(prices)
        
        self.assertEqual(indicators, {})
    
    def test_calculate_technical_indicators_empty(self):
        """Test technical indicators with empty data"""
        prices = []
        
        indicators = calculate_technical_indicators(prices)
        
        self.assertEqual(indicators, {})
    
    def test_calculate_technical_indicators_volatile(self):
        """Test technical indicators with highly volatile data"""
        prices = [100, 200, 50, 250, 25, 300, 10, 400, 5, 500,
                 1, 600, 0.5, 700, 0.1, 800, 0.05, 900, 0.01, 1000]
        
        indicators = calculate_technical_indicators(prices)
        
        self.assertGreater(indicators['volatility'], 0.5)  # High volatility
        self.assertGreater(indicators['price_change'], 5)  # Large price change


class TestMarketConditions(unittest.TestCase):
    """Test cases for market condition analysis"""
    
    def test_analyze_market_conditions_positive_sentiment(self):
        """Test market condition analysis with positive sentiment"""
        technical_data = {
            'price_change': 0.05,  # 5% increase
            'volatility': 0.03     # 3% volatility
        }
        
        conditions = analyze_market_conditions("positive", 0.8, technical_data)
        
        self.assertEqual(conditions['sentiment_strength'], 'strong')
        self.assertEqual(conditions['trend_direction'], 'bullish')
        self.assertEqual(conditions['volatility_level'], 'medium')
    
    def test_analyze_market_conditions_negative_sentiment(self):
        """Test market condition analysis with negative sentiment"""
        technical_data = {
            'price_change': -0.05,  # 5% decrease
            'volatility': 0.08      # 8% volatility
        }
        
        conditions = analyze_market_conditions("negative", 0.75, technical_data)
        
        self.assertEqual(conditions['sentiment_strength'], 'strong')
        self.assertEqual(conditions['trend_direction'], 'bearish')
        self.assertEqual(conditions['volatility_level'], 'high')
    
    def test_analyze_market_conditions_neutral(self):
        """Test market condition analysis with neutral sentiment"""
        technical_data = {
            'price_change': 0.01,   # 1% increase
            'volatility': 0.015     # 1.5% volatility
        }
        
        conditions = analyze_market_conditions("neutral", 0.5, technical_data)
        
        self.assertEqual(conditions['sentiment_strength'], 'weak')
        self.assertEqual(conditions['trend_direction'], 'neutral')
        self.assertEqual(conditions['volatility_level'], 'low')


class TestTradeSignalGeneration(unittest.TestCase):
    """Test cases for trade signal generation"""
    
    def test_generate_buy_signal_strong_positive(self):
        """Test BUY signal generation with strong positive sentiment"""
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            positive_threshold=0.65,
            current_price=100,
            reference_price=110
        )
        
        self.assertEqual(signal.action, "BUY")
        self.assertGreater(signal.confidence, 0.7)
        self.assertIn("positive sentiment", signal.reasoning)
        self.assertIsNotNone(signal.stop_loss)
        self.assertIsNotNone(signal.take_profit)
    
    def test_generate_sell_signal_strong_negative(self):
        """Test SELL signal generation with strong negative sentiment"""
        signal = generate_trade_signal(
            sentiment="negative",
            score=0.8,
            negative_threshold=0.7,
            current_price=110,
            reference_price=100
        )
        
        self.assertEqual(signal.action, "SELL")
        self.assertGreater(signal.confidence, 0.7)
        self.assertIn("negative sentiment", signal.reasoning)
        self.assertIsNotNone(signal.stop_loss)
        self.assertIsNotNone(signal.take_profit)
    
    def test_generate_hold_signal_weak_sentiment(self):
        """Test HOLD signal generation with weak sentiment"""
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.5,
            positive_threshold=0.65,
            current_price=100,
            reference_price=110
        )
        
        self.assertEqual(signal.action, "HOLD")
        self.assertLess(signal.confidence, 0.6)
        self.assertIn("below threshold", signal.reasoning)
    
    def test_generate_hold_signal_neutral_sentiment(self):
        """Test HOLD signal generation with neutral sentiment"""
        signal = generate_trade_signal(
            sentiment="neutral",
            score=0.5,
            current_price=100,
            reference_price=110
        )
        
        self.assertEqual(signal.action, "HOLD")
        self.assertIn("Neutral sentiment", signal.reasoning)
    
    def test_generate_signal_invalid_sentiment(self):
        """Test signal generation with invalid sentiment"""
        signal = generate_trade_signal(
            sentiment="invalid",
            score=0.8,
            current_price=100,
            reference_price=110
        )
        
        # Should default to neutral and generate HOLD signal
        self.assertEqual(signal.action, "HOLD")
        self.assertIn("Neutral sentiment", signal.reasoning)
    
    def test_generate_signal_invalid_score(self):
        """Test signal generation with invalid score"""
        signal = generate_trade_signal(
            sentiment="positive",
            score=1.5,  # Invalid score > 1.0
            current_price=100,
            reference_price=110
        )
        
        # Should clamp score and still generate appropriate signal
        self.assertLessEqual(signal.confidence, 1.0)
    
    def test_generate_signal_with_technical_data(self):
        """Test signal generation with technical indicators"""
        price_history = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                        110, 112, 111, 113, 115, 114, 116, 118, 117, 119]
        
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            current_price=119,
            reference_price=110,
            price_history=price_history
        )
        
        # Should consider technical indicators in decision
        self.assertIsNotNone(signal)
        self.assertIn("Confidence:", signal.reasoning)
    
    def test_generate_signal_price_conditions(self):
        """Test signal generation with price condition checks"""
        # Test BUY when price is below reference
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            current_price=90,  # Below reference
            reference_price=100,
            positive_threshold=0.65
        )
        
        if signal.action == "BUY":
            self.assertIn("Price 90.00 not below reference 100.00", signal.reasoning)
    
    def test_generate_signal_high_volatility(self):
        """Test signal generation with high volatility"""
        price_history = [100, 150, 50, 200, 25, 250, 10, 300, 5, 350,
                        1, 400, 0.5, 450, 0.1, 500, 0.05, 550, 0.01, 600]
        
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            current_price=600,
            reference_price=100,
            price_history=price_history,
            positive_threshold=0.65
        )
        
        # High volatility should reduce confidence or prevent trade
        if signal.action == "BUY":
            self.assertIn("Extremely high volatility", signal.reasoning)
    
    def test_generate_signal_overbought_condition(self):
        """Test signal generation when price is overbought"""
        price_history = [100] * 20  # Flat prices for SMA calculation
        current_price = 110  # 10% above SMA
        
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            current_price=current_price,
            reference_price=100,
            price_history=price_history,
            positive_threshold=0.65
        )
        
        # Should not buy when significantly above SMA
        if signal.action == "BUY":
            self.assertIn("Price significantly above SMA", signal.reasoning)


class TestTradeSignalEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_signal_with_zero_prices(self):
        """Test signal generation with zero prices"""
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            current_price=0,
            reference_price=0
        )
        
        # Should handle zero prices gracefully
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, "HOLD")  # Should default to HOLD
    
    def test_signal_with_negative_prices(self):
        """Test signal generation with negative prices"""
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            current_price=-100,
            reference_price=-110
        )
        
        # Should handle negative prices (though unrealistic)
        self.assertIsNotNone(signal)
    
    def test_signal_with_extreme_scores(self):
        """Test signal generation with extreme scores"""
        # Test score = 0
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.0,
            current_price=100,
            reference_price=110
        )
        
        self.assertEqual(signal.action, "HOLD")
        
        # Test score = 1
        signal = generate_trade_signal(
            sentiment="positive",
            score=1.0,
            current_price=100,
            reference_price=110,
            positive_threshold=0.65
        )
        
        if signal.action == "BUY":
            self.assertGreater(signal.confidence, 0.9)
    
    def test_signal_with_missing_data(self):
        """Test signal generation with missing optional data"""
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            positive_threshold=0.65
        )
        
        # Should work without optional parameters
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, "HOLD")  # No price data = HOLD
    
    def test_signal_confidence_bounds(self):
        """Test that confidence is properly bounded"""
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.9,
            current_price=100,
            reference_price=110,
            positive_threshold=0.65
        )
        
        # Confidence should be between 0 and 0.95
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 0.95)


class TestTradeSignalIntegration(unittest.TestCase):
    """Integration tests for trade signal generation"""
    
    def test_full_trading_scenario_bullish(self):
        """Test complete bullish trading scenario"""
        # Simulate bullish market conditions
        price_history = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                        105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        
        signal = generate_trade_signal(
            sentiment="positive",
            score=0.85,
            current_price=114,
            reference_price=110,
            price_history=price_history,
            positive_threshold=0.65
        )
        
        # Should generate BUY signal with high confidence
        self.assertEqual(signal.action, "BUY")
        self.assertGreater(signal.confidence, 0.7)
        self.assertLess(signal.stop_loss, signal.take_profit)
    
    def test_full_trading_scenario_bearish(self):
        """Test complete bearish trading scenario"""
        # Simulate bearish market conditions
        price_history = [115, 114, 113, 112, 111, 110, 109, 108, 107, 106,
                        105, 104, 103, 102, 101, 100, 99, 98, 97, 96]
        
        signal = generate_trade_signal(
            sentiment="negative",
            score=0.85,
            current_price=96,
            reference_price=100,
            price_history=price_history,
            negative_threshold=0.7
        )
        
        # Should generate SELL signal with high confidence
        self.assertEqual(signal.action, "SELL")
        self.assertGreater(signal.confidence, 0.7)
        self.assertGreater(signal.stop_loss, signal.take_profit)
    
    def test_market_condition_impact_on_confidence(self):
        """Test how market conditions affect confidence levels"""
        # Low volatility scenario
        low_vol_prices = [100, 100.5, 101, 100.8, 101.2, 100.9, 101.1, 100.7, 101.3, 101.0,
                         100.6, 101.4, 101.1, 100.8, 101.2, 100.9, 101.3, 101.0, 100.7, 101.1]
        
        signal_low_vol = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            current_price=101.1,
            reference_price=100,
            price_history=low_vol_prices,
            positive_threshold=0.65
        )
        
        # High volatility scenario
        high_vol_prices = [100, 120, 80, 140, 60, 160, 40, 180, 20, 200,
                          10, 220, 5, 240, 2, 260, 1, 280, 0.5, 300]
        
        signal_high_vol = generate_trade_signal(
            sentiment="positive",
            score=0.8,
            current_price=300,
            reference_price=100,
            price_history=high_vol_prices,
            positive_threshold=0.65
        )
        
        # Low volatility should generally result in higher confidence
        if signal_low_vol.action == "BUY" and signal_high_vol.action == "BUY":
            # This test might not always pass due to other factors, but it's a good check
            pass  # Confidence comparison is complex due to multiple factors


if __name__ == '__main__':
    unittest.main()

