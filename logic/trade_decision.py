import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from config import (
    POSITIVE_THRESHOLD,
    NEGATIVE_THRESHOLD,
    STOP_LOSS_PERCENTAGE,
    TAKE_PROFIT_PERCENTAGE
)

logger = logging.getLogger(__name__)


class TradeSignal:
    """Enhanced trade signal with risk management"""
    
    def __init__(self, action: str, confidence: float, reasoning: str, 
                 stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        self.action = action
        self.confidence = confidence
        self.reasoning = reasoning
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat()
        }


def calculate_technical_indicators(prices: list, window: int = 20) -> Dict[str, float]:
    """Calculate basic technical indicators"""
    if len(prices) < window:
        return {}
    
    recent_prices = prices[-window:]
    
    # Simple Moving Average
    sma = sum(recent_prices) / len(recent_prices)
    
    # Price momentum
    price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    
    # Volatility (standard deviation)
    variance = sum((price - sma) ** 2 for price in recent_prices) / len(recent_prices)
    volatility = variance ** 0.5
    
    return {
        'sma': sma,
        'price_change': price_change,
        'volatility': volatility,
        'current_price': recent_prices[-1]
    }


def analyze_market_conditions(sentiment: str, score: float, technical_data: Dict[str, float]) -> Dict[str, Any]:
    """Analyze overall market conditions"""
    conditions = {
        'sentiment_strength': 'strong' if score > 0.8 else 'moderate' if score > 0.6 else 'weak',
        'trend_direction': 'bullish' if technical_data.get('price_change', 0) > 0.02 else 
                          'bearish' if technical_data.get('price_change', 0) < -0.02 else 'neutral',
        'volatility_level': 'high' if technical_data.get('volatility', 0) > 0.05 else 
                           'low' if technical_data.get('volatility', 0) < 0.02 else 'medium'
    }
    
    return conditions


def generate_trade_signal(
    sentiment: str,
    score: float,
    positive_threshold: float = POSITIVE_THRESHOLD,
    negative_threshold: float = NEGATIVE_THRESHOLD,
    current_price: Optional[float] = None,
    reference_price: Optional[float] = None,
    price_history: Optional[list] = None,
    market_data: Optional[Dict[str, Any]] = None
) -> TradeSignal:
    """
    Generate enhanced trade signal with risk management
    
    Args:
        sentiment: Sentiment classification ('positive', 'negative', 'neutral')
        score: Confidence score (0.0 to 1.0)
        positive_threshold: Minimum score for positive sentiment to trigger BUY
        negative_threshold: Minimum score for negative sentiment to trigger SELL
        current_price: Current market price
        reference_price: Reference price for comparison
        price_history: Historical prices for technical analysis
        market_data: Additional market data
    
    Returns:
        TradeSignal object with action, confidence, and risk management levels
    """
    
    # Input validation
    if sentiment not in ['positive', 'negative', 'neutral']:
        logger.warning(f"Invalid sentiment: {sentiment}. Defaulting to neutral.")
        sentiment = 'neutral'
    
    if not (0.0 <= score <= 1.0):
        logger.warning(f"Invalid score: {score}. Clamping to [0.0, 1.0].")
        score = max(0.0, min(1.0, score))
    
    # Calculate technical indicators if price history is available
    technical_data = {}
    if price_history and len(price_history) > 0:
        technical_data = calculate_technical_indicators(price_history)
    
    # Analyze market conditions
    market_conditions = analyze_market_conditions(sentiment, score, technical_data)
    
    # Determine base signal
    action = "HOLD"
    confidence = 0.0
    reasoning_parts = []
    
    # Sentiment-based signal generation
    if sentiment == "positive" and score > positive_threshold:
        # Additional checks for BUY signal
        price_condition_met = True
        if reference_price and current_price:
            # Only buy if price is below reference (potential dip)
            price_condition_met = current_price < reference_price
            if not price_condition_met:
                reasoning_parts.append(f"Price {current_price:.2f} not below reference {reference_price:.2f}")
        
        # Check technical indicators
        technical_condition_met = True
        if technical_data:
            # Don't buy if price is significantly above SMA (overbought)
            if technical_data.get('current_price', 0) > technical_data.get('sma', 0) * 1.05:
                technical_condition_met = False
                reasoning_parts.append("Price significantly above SMA")
            
            # Don't buy if volatility is extremely high
            if technical_data.get('volatility', 0) > 0.1:
                technical_condition_met = False
                reasoning_parts.append("Extremely high volatility")
        
        if price_condition_met and technical_condition_met:
            action = "BUY"
            confidence = min(0.95, score * 1.2)  # Boost confidence for positive sentiment
            reasoning_parts.append(f"Strong positive sentiment ({score:.2f}) with favorable conditions")
        else:
            reasoning_parts.append(f"Positive sentiment ({score:.2f}) but conditions not met")
    
    elif sentiment == "negative" and score > negative_threshold:
        # Additional checks for SELL signal
        price_condition_met = True
        if reference_price and current_price:
            # Only sell if price is above reference (potential peak)
            price_condition_met = current_price > reference_price
            if not price_condition_met:
                reasoning_parts.append(f"Price {current_price:.2f} not above reference {reference_price:.2f}")
        
        # Check technical indicators
        technical_condition_met = True
        if technical_data:
            # Don't sell if price is significantly below SMA (oversold)
            if technical_data.get('current_price', 0) < technical_data.get('sma', 0) * 0.95:
                technical_condition_met = False
                reasoning_parts.append("Price significantly below SMA")
        
        if price_condition_met and technical_condition_met:
            action = "SELL"
            confidence = min(0.95, score * 1.2)  # Boost confidence for negative sentiment
            reasoning_parts.append(f"Strong negative sentiment ({score:.2f}) with favorable conditions")
        else:
            reasoning_parts.append(f"Negative sentiment ({score:.2f}) but conditions not met")
    
    else:
        # HOLD signal
        if sentiment == "neutral":
            reasoning_parts.append(f"Neutral sentiment ({score:.2f})")
        elif sentiment == "positive":
            reasoning_parts.append(f"Positive sentiment ({score:.2f}) below threshold ({positive_threshold})")
        elif sentiment == "negative":
            reasoning_parts.append(f"Negative sentiment ({score:.2f}) below threshold ({negative_threshold})")
        
        # Add market condition context
        reasoning_parts.append(f"Market trend: {market_conditions['trend_direction']}")
        reasoning_parts.append(f"Volatility: {market_conditions['volatility_level']}")
    
    # Calculate stop loss and take profit levels
    stop_loss = None
    take_profit = None
    
    if action in ["BUY", "SELL"] and current_price:
        if action == "BUY":
            stop_loss = current_price * (1 - STOP_LOSS_PERCENTAGE)
            take_profit = current_price * (1 + TAKE_PROFIT_PERCENTAGE)
        elif action == "SELL":
            stop_loss = current_price * (1 + STOP_LOSS_PERCENTAGE)
            take_profit = current_price * (1 - TAKE_PROFIT_PERCENTAGE)
    
    # Adjust confidence based on market conditions
    if action != "HOLD":
        if market_conditions['volatility_level'] == 'high':
            confidence *= 0.8  # Reduce confidence in high volatility
        elif market_conditions['volatility_level'] == 'low':
            confidence *= 1.1  # Increase confidence in low volatility
        
        # Align with trend
        if (action == "BUY" and market_conditions['trend_direction'] == 'bullish') or \
           (action == "SELL" and market_conditions['trend_direction'] == 'bearish'):
            confidence *= 1.1
    
    confidence = max(0.0, min(0.95, confidence))  # Clamp confidence
    
    # Create final reasoning
    reasoning = "; ".join(reasoning_parts)
    if action != "HOLD":
        reasoning += f"; Confidence: {confidence:.2f}"
    
    logger.info(f"Trade signal: {action} (confidence: {confidence:.2f}) - {reasoning}")
    
    return TradeSignal(
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        stop_loss=stop_loss,
        take_profit=take_profit
    )


# Legacy function for backward compatibility
def generate_trade_signal_legacy(
    sentiment,
    score,
    positive_threshold=0.65,
    negative_threshold=0.7,
    current_price=None,
    reference_price=None
):
    """Legacy function for backward compatibility"""
    signal = generate_trade_signal(
        sentiment=sentiment,
        score=score,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        current_price=current_price,
        reference_price=reference_price
    )
    return signal.action