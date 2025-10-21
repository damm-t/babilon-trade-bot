"""
Risk Management Module for Phase 6
Implements comprehensive risk controls and position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    expected_shortfall: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    beta: float
    correlation: float
    position_size: float
    leverage: float
    risk_level: RiskLevel


@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    recommended_size: float
    max_size: float
    risk_adjusted_size: float
    kelly_fraction: float
    confidence_multiplier: float
    risk_reasoning: List[str]


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.positions = {}
        self.risk_history = []
        self.alert_thresholds = self._get_alert_thresholds()
        
    def _get_default_config(self) -> Dict:
        """Get default risk management configuration"""
        return {
            'max_portfolio_risk': 0.02,  # 2% max portfolio risk
            'max_position_risk': 0.01,   # 1% max position risk
            'max_correlation': 0.7,      # Max correlation between positions
            'max_leverage': 2.0,         # Max leverage
            'stop_loss_pct': 0.05,       # 5% stop loss
            'take_profit_pct': 0.10,     # 10% take profit
            'max_drawdown': 0.15,        # 15% max drawdown
            'var_confidence': 0.95,      # VaR confidence level
            'rebalance_frequency': 24,   # hours
            'risk_free_rate': 0.02       # 2% risk-free rate
        }
    
    def _get_alert_thresholds(self) -> Dict[str, float]:
        """Get risk alert thresholds"""
        return {
            'high_volatility': 0.3,      # 30% volatility
            'high_correlation': 0.8,    # 80% correlation
            'high_leverage': 1.5,        # 1.5x leverage
            'high_drawdown': 0.10,      # 10% drawdown
            'high_var': 0.05,           # 5% VaR
            'low_liquidity': 0.1        # 10% of average volume
        }
    
    def calculate_position_size(self, 
                              signal_confidence: float,
                              volatility: float,
                              portfolio_value: float,
                              current_positions: Dict[str, float],
                              ticker: str) -> PositionSizing:
        """Calculate optimal position size using multiple methods"""
        
        risk_reasoning = []
        
        # 1. Kelly Criterion
        kelly_fraction = self._calculate_kelly_fraction(signal_confidence, volatility)
        risk_reasoning.append(f"Kelly fraction: {kelly_fraction:.3f}")
        
        # 2. Risk Parity
        risk_parity_size = self._calculate_risk_parity_size(volatility, portfolio_value)
        risk_reasoning.append(f"Risk parity size: {risk_parity_size:.2f}")
        
        # 3. Volatility targeting
        vol_target_size = self._calculate_volatility_target_size(volatility, portfolio_value)
        risk_reasoning.append(f"Volatility target size: {vol_target_size:.2f}")
        
        # 4. Confidence-based sizing
        confidence_multiplier = 0.5 + (signal_confidence * 0.5)  # 0.5x to 1.0x
        risk_reasoning.append(f"Confidence multiplier: {confidence_multiplier:.3f}")
        
        # 5. Correlation adjustment
        correlation_penalty = self._calculate_correlation_penalty(ticker, current_positions)
        risk_reasoning.append(f"Correlation penalty: {correlation_penalty:.3f}")
        
        # Combine methods
        base_size = min(kelly_fraction, risk_parity_size, vol_target_size)
        adjusted_size = base_size * confidence_multiplier * (1 - correlation_penalty)
        
        # Apply risk limits
        max_size = self._calculate_max_position_size(portfolio_value, volatility)
        recommended_size = min(adjusted_size, max_size)
        
        # Risk-adjusted size (conservative)
        risk_adjusted_size = recommended_size * 0.8  # 20% safety margin
        
        return PositionSizing(
            recommended_size=recommended_size,
            max_size=max_size,
            risk_adjusted_size=risk_adjusted_size,
            kelly_fraction=kelly_fraction,
            confidence_multiplier=confidence_multiplier,
            risk_reasoning=risk_reasoning
        )
    
    def _calculate_kelly_fraction(self, win_prob: float, avg_win: float, avg_loss: float = None) -> float:
        """Calculate Kelly fraction for position sizing"""
        if avg_loss is None:
            avg_loss = 0.02  # Assume 2% average loss
        
        if avg_loss <= 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_prob, q = 1 - win_prob
        b = avg_win / avg_loss
        q = 1 - win_prob
        kelly = (b * win_prob - q) / b
        
        # Cap Kelly at 25% to prevent over-leveraging
        return max(0.0, min(0.25, kelly))
    
    def _calculate_risk_parity_size(self, volatility: float, portfolio_value: float) -> float:
        """Calculate risk parity position size"""
        target_risk = self.config['max_position_risk']
        position_value = (target_risk * portfolio_value) / volatility
        return position_value / portfolio_value  # Return as fraction
    
    def _calculate_volatility_target_size(self, volatility: float, portfolio_value: float) -> float:
        """Calculate volatility target position size"""
        target_volatility = 0.15  # 15% target volatility
        if volatility <= 0:
            return 0.0
        
        vol_ratio = target_volatility / volatility
        return min(vol_ratio, 1.0)  # Cap at 100% of portfolio
    
    def _calculate_correlation_penalty(self, ticker: str, current_positions: Dict[str, float]) -> float:
        """Calculate correlation penalty for position sizing"""
        if not current_positions:
            return 0.0
        
        # Simplified correlation penalty based on number of positions
        num_positions = len(current_positions)
        if num_positions == 0:
            return 0.0
        elif num_positions == 1:
            return 0.1  # 10% penalty for second position
        elif num_positions == 2:
            return 0.2  # 20% penalty for third position
        else:
            return 0.3  # 30% penalty for additional positions
    
    def _calculate_max_position_size(self, portfolio_value: float, volatility: float) -> float:
        """Calculate maximum position size based on risk limits"""
        max_risk = self.config['max_position_risk']
        max_value = max_risk * portfolio_value
        
        if volatility <= 0:
            return 0.0
        
        max_size = max_value / (volatility * portfolio_value)
        return min(max_size, 0.2)  # Cap at 20% of portfolio
    
    def calculate_portfolio_risk(self, positions: Dict[str, Dict], price_data: pd.DataFrame) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not positions:
            return RiskMetrics(
                var_95=0.0, var_99=0.0, expected_shortfall=0.0,
                max_drawdown=0.0, volatility=0.0, sharpe_ratio=0.0,
                beta=0.0, correlation=0.0, position_size=0.0,
                leverage=0.0, risk_level=RiskLevel.LOW
            )
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions, price_data)
        
        # Value at Risk
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Volatility
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Sharpe ratio
        excess_return = np.mean(portfolio_returns) * 252 - self.config['risk_free_rate']
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        
        # Beta (simplified)
        beta = 1.0  # Assume market beta of 1.0
        
        # Correlation (average correlation between positions)
        correlation = self._calculate_avg_correlation(positions, price_data)
        
        # Leverage
        total_position_value = sum(pos.get('value', 0) for pos in positions.values())
        portfolio_value = sum(pos.get('portfolio_value', 10000) for pos in positions.values())
        leverage = total_position_value / portfolio_value if portfolio_value > 0 else 0.0
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        # Risk level
        risk_level = self._assess_risk_level(volatility, leverage, max_drawdown, var_95)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            beta=beta,
            correlation=correlation,
            position_size=len(positions),
            leverage=leverage,
            risk_level=risk_level
        )
    
    def _calculate_portfolio_returns(self, positions: Dict[str, Dict], price_data: pd.DataFrame) -> np.ndarray:
        """Calculate portfolio returns"""
        # Simplified calculation - in practice, this would be more complex
        returns = []
        
        for ticker, position in positions.items():
            if ticker in price_data.columns:
                ticker_returns = price_data[ticker].pct_change().dropna()
                position_weight = position.get('weight', 0.1)
                weighted_returns = ticker_returns * position_weight
                returns.append(weighted_returns)
        
        if returns:
            # Combine returns (simplified)
            portfolio_returns = np.sum(returns, axis=0)
            return portfolio_returns
        else:
            return np.array([0.0])
    
    def _calculate_avg_correlation(self, positions: Dict[str, Dict], price_data: pd.DataFrame) -> float:
        """Calculate average correlation between positions"""
        if len(positions) < 2:
            return 0.0
        
        tickers = [ticker for ticker in positions.keys() if ticker in price_data.columns]
        if len(tickers) < 2:
            return 0.0
        
        # Calculate correlation matrix
        returns_data = price_data[tickers].pct_change().dropna()
        correlation_matrix = returns_data.corr()
        
        # Get average correlation (excluding diagonal)
        correlations = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                correlations.append(correlation_matrix.iloc[i, j])
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _assess_risk_level(self, volatility: float, leverage: float, 
                          max_drawdown: float, var_95: float) -> RiskLevel:
        """Assess overall risk level"""
        risk_score = 0
        
        # Volatility risk
        if volatility > 0.3:
            risk_score += 3
        elif volatility > 0.2:
            risk_score += 2
        elif volatility > 0.1:
            risk_score += 1
        
        # Leverage risk
        if leverage > 2.0:
            risk_score += 3
        elif leverage > 1.5:
            risk_score += 2
        elif leverage > 1.0:
            risk_score += 1
        
        # Drawdown risk
        if max_drawdown < -0.15:
            risk_score += 3
        elif max_drawdown < -0.10:
            risk_score += 2
        elif max_drawdown < -0.05:
            risk_score += 1
        
        # VaR risk
        if var_95 < -0.05:
            risk_score += 3
        elif var_95 < -0.03:
            risk_score += 2
        elif var_95 < -0.01:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 8:
            return RiskLevel.CRITICAL
        elif risk_score >= 6:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def check_risk_limits(self, positions: Dict[str, Dict], 
                         new_position: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if new position violates risk limits"""
        violations = []
        
        # Check maximum positions
        if len(positions) >= 5:  # Max 5 positions
            violations.append("Maximum number of positions exceeded")
        
        # Check position size
        position_size = new_position.get('size', 0)
        if position_size > 0.2:  # Max 20% per position
            violations.append("Position size exceeds 20% limit")
        
        # Check portfolio risk
        total_risk = sum(pos.get('risk', 0) for pos in positions.values())
        new_risk = new_position.get('risk', 0)
        if total_risk + new_risk > self.config['max_portfolio_risk']:
            violations.append("Portfolio risk limit exceeded")
        
        # Check leverage
        total_leverage = sum(pos.get('leverage', 0) for pos in positions.values())
        new_leverage = new_position.get('leverage', 0)
        if total_leverage + new_leverage > self.config['max_leverage']:
            violations.append("Leverage limit exceeded")
        
        return len(violations) == 0, violations
    
    def generate_risk_alerts(self, risk_metrics: RiskMetrics) -> List[str]:
        """Generate risk alerts based on current metrics"""
        alerts = []
        
        if risk_metrics.volatility > self.alert_thresholds['high_volatility']:
            alerts.append(f"High volatility: {risk_metrics.volatility:.2%}")
        
        if risk_metrics.leverage > self.alert_thresholds['high_leverage']:
            alerts.append(f"High leverage: {risk_metrics.leverage:.2f}x")
        
        if risk_metrics.max_drawdown < -self.alert_thresholds['high_drawdown']:
            alerts.append(f"High drawdown: {risk_metrics.max_drawdown:.2%}")
        
        if risk_metrics.var_95 < -self.alert_thresholds['high_var']:
            alerts.append(f"High VaR: {risk_metrics.var_95:.2%}")
        
        if risk_metrics.correlation > self.alert_thresholds['high_correlation']:
            alerts.append(f"High correlation: {risk_metrics.correlation:.2%}")
        
        return alerts
    
    def recommend_risk_actions(self, risk_metrics: RiskMetrics) -> List[str]:
        """Recommend risk management actions"""
        actions = []
        
        if risk_metrics.risk_level == RiskLevel.CRITICAL:
            actions.append("REDUCE POSITIONS IMMEDIATELY")
            actions.append("Consider closing all positions")
            actions.append("Review risk management parameters")
        
        elif risk_metrics.risk_level == RiskLevel.HIGH:
            actions.append("Reduce position sizes")
            actions.append("Increase stop-loss levels")
            actions.append("Monitor positions closely")
        
        elif risk_metrics.risk_level == RiskLevel.MEDIUM:
            actions.append("Monitor risk metrics")
            actions.append("Consider reducing leverage")
            actions.append("Review correlation between positions")
        
        else:  # LOW risk
            actions.append("Risk levels are acceptable")
            actions.append("Continue monitoring")
        
        return actions


def create_risk_manager(config: Optional[Dict] = None) -> RiskManager:
    """Create a risk manager with optional configuration"""
    return RiskManager(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create risk manager
    risk_manager = create_risk_manager()
    
    print("Risk manager ready for position sizing and risk control")
