"""
Hybrid Signal Engine for Phase 5
Combines ML predictions with rule-based logic for enhanced trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class HybridSignal:
    """Hybrid signal result"""
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    strength: SignalStrength
    ml_score: float
    rule_score: float
    combined_score: float
    reasoning: List[str]
    timestamp: datetime
    features: Dict[str, float]
    risk_metrics: Dict[str, float]


class HybridSignalEngine:
    """Hybrid signal engine combining ML and rule-based logic"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.ml_model = None
        self.feature_weights = self._get_feature_weights()
        self.performance_history = []
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'ml_weight': 0.6,
            'rule_weight': 0.4,
            'sentiment_weight': 0.3,
            'momentum_weight': 0.2,
            'volatility_weight': 0.1,
            'volume_weight': 0.1,
            'trend_weight': 0.3,
            'buy_threshold': 0.65,
            'sell_threshold': 0.35,
            'confidence_boost': 0.15,
            'cooldown_period': 30,  # minutes
            'max_positions': 3,
            'position_size_limit': 0.1  # 10% of portfolio
        }
    
    def _get_feature_weights(self) -> Dict[str, float]:
        """Get feature importance weights"""
        return {
            # Price momentum features (high importance)
            'price_change': 0.15,
            'log_return': 0.15,
            'price_sma20_ratio': 0.12,
            'price_sma50_ratio': 0.10,
            
            # Technical indicators (medium-high importance)
            'rsi_14': 0.08,
            'macd': 0.08,
            'bb_position': 0.07,
            'stoch_k': 0.06,
            
            # Volume features (medium importance)
            'volume_ratio': 0.05,
            'volume_price_trend': 0.05,
            
            # Trend and regime (medium importance)
            'trend_20_50': 0.06,
            'regime': 0.04,
            'regime_strength': 0.04,
            
            # Volatility (lower importance)
            'historical_volatility': 0.03,
            'atr_ratio': 0.02,
            'volatility_ratio': 0.01
        }
    
    def set_ml_model(self, model):
        """Set the ML model for predictions"""
        self.ml_model = model
        logger.info("ML model set for hybrid signal engine")
    
    def calculate_rule_score(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """Calculate rule-based score from features"""
        reasoning = []
        rule_score = 0.0
        
        # Sentiment-based rules
        sentiment_score = self._calculate_sentiment_score(features, reasoning)
        
        # Momentum-based rules
        momentum_score = self._calculate_momentum_score(features, reasoning)
        
        # Volatility-based rules
        volatility_score = self._calculate_volatility_score(features, reasoning)
        
        # Volume-based rules
        volume_score = self._calculate_volume_score(features, reasoning)
        
        # Trend-based rules
        trend_score = self._calculate_trend_score(features, reasoning)
        
        # Combine rule scores
        rule_score = (
            self.config['sentiment_weight'] * sentiment_score +
            self.config['momentum_weight'] * momentum_score +
            self.config['volatility_weight'] * volatility_score +
            self.config['volume_weight'] * volume_score +
            self.config['trend_weight'] * trend_score
        )
        
        # Apply rule-based adjustments
        rule_score = self._apply_rule_adjustments(rule_score, features, reasoning)
        
        return rule_score, reasoning
    
    def _calculate_sentiment_score(self, features: Dict[str, float], reasoning: List[str]) -> float:
        """Calculate sentiment-based score"""
        sentiment_score = 0.5  # neutral baseline
        
        # Check for sentiment features
        if 'score_mean' in features and not pd.isna(features['score_mean']):
            sentiment_score = features['score_mean']
            reasoning.append(f"Sentiment score: {sentiment_score:.3f}")
        
        # Check for weighted sentiment
        if 'weighted_sentiment' in features and not pd.isna(features['weighted_sentiment']):
            sentiment_score = features['weighted_sentiment']
            reasoning.append(f"Weighted sentiment: {sentiment_score:.3f}")
        
        # Sentiment momentum
        if 'sentiment_momentum' in features and not pd.isna(features['sentiment_momentum']):
            momentum = features['sentiment_momentum']
            if momentum > 0.1:
                sentiment_score += 0.1
                reasoning.append(f"Positive sentiment momentum: {momentum:.3f}")
            elif momentum < -0.1:
                sentiment_score -= 0.1
                reasoning.append(f"Negative sentiment momentum: {momentum:.3f}")
        
        return max(0.0, min(1.0, sentiment_score))
    
    def _calculate_momentum_score(self, features: Dict[str, float], reasoning: List[str]) -> float:
        """Calculate momentum-based score"""
        momentum_score = 0.5  # neutral baseline
        
        # Price momentum
        if 'price_change' in features and not pd.isna(features['price_change']):
            price_change = features['price_change']
            momentum_score += price_change * 2  # Scale price change
            reasoning.append(f"Price change: {price_change:.3f}")
        
        # RSI momentum
        if 'rsi_14' in features and not pd.isna(features['rsi_14']):
            rsi = features['rsi_14']
            if rsi > 70:
                momentum_score -= 0.2  # Overbought
                reasoning.append(f"RSI overbought: {rsi:.1f}")
            elif rsi < 30:
                momentum_score += 0.2  # Oversold
                reasoning.append(f"RSI oversold: {rsi:.1f}")
            else:
                # Normalize RSI to 0-1 scale
                rsi_normalized = (rsi - 30) / 40  # 30-70 range to 0-1
                momentum_score += (rsi_normalized - 0.5) * 0.3
                reasoning.append(f"RSI normalized: {rsi_normalized:.3f}")
        
        # MACD momentum
        if 'macd' in features and not pd.isna(features['macd']):
            macd = features['macd']
            if macd > 0:
                momentum_score += 0.1
                reasoning.append(f"MACD bullish: {macd:.3f}")
            else:
                momentum_score -= 0.1
                reasoning.append(f"MACD bearish: {macd:.3f}")
        
        return max(0.0, min(1.0, momentum_score))
    
    def _calculate_volatility_score(self, features: Dict[str, float], reasoning: List[str]) -> float:
        """Calculate volatility-based score"""
        volatility_score = 0.5  # neutral baseline
        
        # Historical volatility
        if 'historical_volatility' in features and not pd.isna(features['historical_volatility']):
            vol = features['historical_volatility']
            if vol > 0.3:  # High volatility
                volatility_score -= 0.2
                reasoning.append(f"High volatility: {vol:.3f}")
            elif vol < 0.1:  # Low volatility
                volatility_score += 0.1
                reasoning.append(f"Low volatility: {vol:.3f}")
        
        # ATR ratio
        if 'atr_ratio' in features and not pd.isna(features['atr_ratio']):
            atr_ratio = features['atr_ratio']
            if atr_ratio > 0.05:  # High ATR
                volatility_score -= 0.1
                reasoning.append(f"High ATR: {atr_ratio:.3f}")
        
        return max(0.0, min(1.0, volatility_score))
    
    def _calculate_volume_score(self, features: Dict[str, float], reasoning: List[str]) -> float:
        """Calculate volume-based score"""
        volume_score = 0.5  # neutral baseline
        
        # Volume ratio
        if 'volume_ratio' in features and not pd.isna(features['volume_ratio']):
            vol_ratio = features['volume_ratio']
            if vol_ratio > 1.5:  # High volume
                volume_score += 0.2
                reasoning.append(f"High volume: {vol_ratio:.2f}x")
            elif vol_ratio < 0.5:  # Low volume
                volume_score -= 0.1
                reasoning.append(f"Low volume: {vol_ratio:.2f}x")
        
        # Volume-price trend
        if 'volume_price_trend' in features and not pd.isna(features['volume_price_trend']):
            vpt = features['volume_price_trend']
            if vpt > 0:
                volume_score += 0.1
                reasoning.append(f"Positive volume-price trend: {vpt:.3f}")
            else:
                volume_score -= 0.1
                reasoning.append(f"Negative volume-price trend: {vpt:.3f}")
        
        return max(0.0, min(1.0, volume_score))
    
    def _calculate_trend_score(self, features: Dict[str, float], reasoning: List[str]) -> float:
        """Calculate trend-based score"""
        trend_score = 0.5  # neutral baseline
        
        # SMA ratios
        if 'price_sma20_ratio' in features and not pd.isna(features['price_sma20_ratio']):
            sma20_ratio = features['price_sma20_ratio']
            trend_score += sma20_ratio * 0.3
            reasoning.append(f"SMA20 ratio: {sma20_ratio:.3f}")
        
        if 'price_sma50_ratio' in features and not pd.isna(features['price_sma50_ratio']):
            sma50_ratio = features['price_sma50_ratio']
            trend_score += sma50_ratio * 0.2
            reasoning.append(f"SMA50 ratio: {sma50_ratio:.3f}")
        
        # Trend strength
        if 'trend_20_50' in features and not pd.isna(features['trend_20_50']):
            trend_strength = features['trend_20_50']
            trend_score += trend_strength * 0.2
            reasoning.append(f"Trend strength: {trend_strength:.3f}")
        
        # Market regime
        if 'regime' in features and not pd.isna(features['regime']):
            regime = features['regime']
            if regime > 0:  # Uptrend
                trend_score += 0.1
                reasoning.append(f"Uptrend regime: {regime}")
            elif regime < 0:  # Downtrend
                trend_score -= 0.1
                reasoning.append(f"Downtrend regime: {regime}")
        
        return max(0.0, min(1.0, trend_score))
    
    def _apply_rule_adjustments(self, rule_score: float, features: Dict[str, float], reasoning: List[str]) -> float:
        """Apply rule-based adjustments to the score"""
        adjusted_score = rule_score
        
        # Bollinger Bands position
        if 'bb_position' in features and not pd.isna(features['bb_position']):
            bb_pos = features['bb_position']
            if bb_pos > 0.8:  # Near upper band
                adjusted_score -= 0.1
                reasoning.append(f"Near upper Bollinger Band: {bb_pos:.3f}")
            elif bb_pos < 0.2:  # Near lower band
                adjusted_score += 0.1
                reasoning.append(f"Near lower Bollinger Band: {bb_pos:.3f}")
        
        # Stochastic position
        if 'stoch_k' in features and not pd.isna(features['stoch_k']):
            stoch = features['stoch_k']
            if stoch > 80:  # Overbought
                adjusted_score -= 0.1
                reasoning.append(f"Stochastic overbought: {stoch:.1f}")
            elif stoch < 20:  # Oversold
                adjusted_score += 0.1
                reasoning.append(f"Stochastic oversold: {stoch:.1f}")
        
        # MACD signal
        if 'macd_signal' in features and not pd.isna(features['macd_signal']):
            macd_signal = features['macd_signal']
            if 'macd' in features and not pd.isna(features['macd']):
                macd = features['macd']
                if macd > macd_signal:  # MACD above signal
                    adjusted_score += 0.05
                    reasoning.append("MACD above signal line")
                else:  # MACD below signal
                    adjusted_score -= 0.05
                    reasoning.append("MACD below signal line")
        
        return max(0.0, min(1.0, adjusted_score))
    
    def calculate_ml_score(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """Calculate ML-based score"""
        reasoning = []
        
        if self.ml_model is None:
            # Fallback to weighted feature score
            ml_score = self._calculate_weighted_feature_score(features, reasoning)
            reasoning.append("ML model not available, using weighted features")
        else:
            try:
                # Prepare features for ML model
                feature_vector = self._prepare_feature_vector(features)
                ml_score = self.ml_model.predict_proba(feature_vector.reshape(1, -1))[0][1]
                reasoning.append(f"ML model prediction: {ml_score:.3f}")
            except Exception as e:
                logger.warning(f"ML model prediction failed: {e}")
                ml_score = self._calculate_weighted_feature_score(features, reasoning)
                reasoning.append(f"ML model failed, using weighted features: {e}")
        
        return ml_score, reasoning
    
    def _calculate_weighted_feature_score(self, features: Dict[str, float], reasoning: List[str]) -> float:
        """Calculate weighted feature score as ML fallback"""
        weighted_score = 0.0
        total_weight = 0.0
        
        for feature_name, weight in self.feature_weights.items():
            if feature_name in features and not pd.isna(features[feature_name]):
                feature_value = features[feature_name]
                # Normalize feature value to 0-1 range
                normalized_value = self._normalize_feature(feature_name, feature_value)
                weighted_score += weight * normalized_value
                total_weight += weight
                reasoning.append(f"{feature_name}: {normalized_value:.3f} (weight: {weight:.3f})")
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.5  # neutral score
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize feature value to 0-1 range"""
        # Price ratios: -0.1 to 0.1 -> 0 to 1
        if 'ratio' in feature_name:
            return max(0.0, min(1.0, (value + 0.1) / 0.2))
        
        # RSI: 0-100 -> 0-1
        elif 'rsi' in feature_name:
            return value / 100.0
        
        # MACD: normalize based on typical range
        elif 'macd' in feature_name:
            return max(0.0, min(1.0, (value + 2.0) / 4.0))
        
        # Volume ratio: 0-5 -> 0-1
        elif 'volume' in feature_name:
            return max(0.0, min(1.0, value / 5.0))
        
        # Default normalization
        else:
            return max(0.0, min(1.0, (value + 1.0) / 2.0))
    
    def _prepare_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for ML model"""
        feature_vector = []
        feature_names = list(self.feature_weights.keys())
        
        for feature_name in feature_names:
            if feature_name in features and not pd.isna(features[feature_name]):
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value for missing features
        
        return np.array(feature_vector)
    
    def generate_hybrid_signal(self, features: Dict[str, float], current_position: float = 0.0) -> HybridSignal:
        """Generate hybrid signal combining ML and rule-based logic"""
        # Calculate ML score
        ml_score, ml_reasoning = self.calculate_ml_score(features)
        
        # Calculate rule score
        rule_score, rule_reasoning = self.calculate_rule_score(features)
        
        # Combine scores
        combined_score = (
            self.config['ml_weight'] * ml_score +
            self.config['rule_weight'] * rule_score
        )
        
        # Apply confidence boost for strong signals
        if combined_score > 0.8 or combined_score < 0.2:
            combined_score = min(1.0, max(0.0, combined_score + self.config['confidence_boost']))
        
        # Determine action
        if combined_score > self.config['buy_threshold'] and current_position == 0:
            action = "BUY"
        elif combined_score < self.config['sell_threshold'] and current_position > 0:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Determine signal strength
        if combined_score > 0.8 or combined_score < 0.2:
            strength = SignalStrength.VERY_STRONG
        elif combined_score > 0.7 or combined_score < 0.3:
            strength = SignalStrength.STRONG
        elif combined_score > 0.6 or combined_score < 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Combine reasoning
        reasoning = ml_reasoning + rule_reasoning
        reasoning.append(f"Combined score: {combined_score:.3f} (ML: {ml_score:.3f}, Rule: {rule_score:.3f})")
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(features, combined_score)
        
        return HybridSignal(
            action=action,
            confidence=combined_score,
            strength=strength,
            ml_score=ml_score,
            rule_score=rule_score,
            combined_score=combined_score,
            reasoning=reasoning,
            timestamp=datetime.now(),
            features=features,
            risk_metrics=risk_metrics
        )
    
    def _calculate_risk_metrics(self, features: Dict[str, float], score: float) -> Dict[str, float]:
        """Calculate risk metrics for the signal"""
        risk_metrics = {}
        
        # Volatility risk
        if 'historical_volatility' in features and not pd.isna(features['historical_volatility']):
            risk_metrics['volatility_risk'] = features['historical_volatility']
        else:
            risk_metrics['volatility_risk'] = 0.2  # Default moderate risk
        
        # Position size risk
        risk_metrics['position_size_risk'] = min(0.1, score * 0.15)  # Max 10% position
        
        # Drawdown risk
        if 'atr_ratio' in features and not pd.isna(features['atr_ratio']):
            risk_metrics['drawdown_risk'] = features['atr_ratio'] * 2
        else:
            risk_metrics['drawdown_risk'] = 0.05  # Default 5% drawdown risk
        
        # Correlation risk (simplified)
        risk_metrics['correlation_risk'] = 0.3  # Default moderate correlation
        
        return risk_metrics
    
    def update_performance(self, signal: HybridSignal, actual_return: float):
        """Update performance tracking"""
        performance = {
            'timestamp': signal.timestamp,
            'action': signal.action,
            'confidence': signal.confidence,
            'actual_return': actual_return,
            'ml_score': signal.ml_score,
            'rule_score': signal.rule_score,
            'combined_score': signal.combined_score
        }
        self.performance_history.append(performance)
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.performance_history:
            return {}
        
        df = pd.DataFrame(self.performance_history)
        
        # Calculate performance metrics
        total_signals = len(df)
        correct_signals = len(df[df['actual_return'] > 0])
        accuracy = correct_signals / total_signals if total_signals > 0 else 0
        
        avg_return = df['actual_return'].mean()
        avg_confidence = df['confidence'].mean()
        
        # ML vs Rule performance
        ml_correct = len(df[(df['ml_score'] > 0.5) & (df['actual_return'] > 0)])
        rule_correct = len(df[(df['rule_score'] > 0.5) & (df['actual_return'] > 0)])
        
        return {
            'total_signals': total_signals,
            'accuracy': accuracy,
            'avg_return': avg_return,
            'avg_confidence': avg_confidence,
            'ml_accuracy': ml_correct / total_signals if total_signals > 0 else 0,
            'rule_accuracy': rule_correct / total_signals if total_signals > 0 else 0
        }


def create_hybrid_signal_engine(config: Optional[Dict] = None) -> HybridSignalEngine:
    """Create a hybrid signal engine with optional configuration"""
    return HybridSignalEngine(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create hybrid signal engine
    engine = create_hybrid_signal_engine()
    
    # Sample features
    sample_features = {
        'price_change': 0.02,
        'rsi_14': 65.0,
        'macd': 0.5,
        'volume_ratio': 1.2,
        'price_sma20_ratio': 0.01,
        'bb_position': 0.6,
        'stoch_k': 70.0,
        'historical_volatility': 0.15
    }
    
    # Generate hybrid signal
    signal = engine.generate_hybrid_signal(sample_features)
    
    print(f"Action: {signal.action}")
    print(f"Confidence: {signal.confidence:.3f}")
    print(f"Strength: {signal.strength.value}")
    print(f"ML Score: {signal.ml_score:.3f}")
    print(f"Rule Score: {signal.rule_score:.3f}")
    print(f"Reasoning: {'; '.join(signal.reasoning)}")
