"""
Backtest Engine for Phase 5
Implements comprehensive backtesting with realistic market constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

from .hybrid_signal import HybridSignalEngine, HybridSignal

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest result data"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    trades: List[Dict]
    equity_curve: pd.Series
    drawdown_curve: pd.Series


class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        
    def _get_default_config(self) -> Dict:
        """Get default backtest configuration"""
        return {
            'initial_capital': 10000,
            'commission': 0.001,  # 0.1% commission
            'slippage': 0.0005,   # 0.05% slippage
            'max_position_size': 0.1,  # 10% max position
            'stop_loss': 0.05,    # 5% stop loss
            'take_profit': 0.10,  # 10% take profit
            'min_trade_size': 100,  # Minimum trade size
            'max_trades_per_day': 5,
            'cooldown_period': 30,  # minutes
            'risk_free_rate': 0.02  # 2% risk-free rate
        }
    
    def run_backtest(self, 
                    feature_data: pd.DataFrame,
                    price_data: pd.DataFrame,
                    signal_engine: HybridSignalEngine,
                    initial_capital: Optional[float] = None,
                    commission: Optional[float] = None) -> Dict[str, float]:
        """Run comprehensive backtest"""
        
        logger.info("Starting backtest simulation")
        
        # Override config if provided
        if initial_capital is not None:
            self.config['initial_capital'] = initial_capital
        if commission is not None:
            self.config['commission'] = commission
        
        # Initialize backtest state
        capital = self.config['initial_capital']
        position = 0.0
        position_value = 0.0
        last_trade_time = None
        daily_trades = 0
        current_date = None
        
        # Reset tracking
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Align feature and price data
        common_index = feature_data.index.intersection(price_data.index)
        if len(common_index) == 0:
            raise ValueError("No common timestamps between feature and price data")
        
        feature_data = feature_data.loc[common_index]
        price_data = price_data.loc[common_index]
        
        logger.info(f"Running backtest on {len(common_index)} data points")
        
        # Process each timestamp
        for i, timestamp in enumerate(common_index):
            current_price = price_data.loc[timestamp, 'Close']
            current_date = timestamp.date()
            
            # Reset daily trade counter
            if i == 0 or timestamp.date() != common_index[i-1].date():
                daily_trades = 0
            
            # Skip if we've hit daily trade limit
            if daily_trades >= self.config['max_trades_per_day']:
                continue

            # Check cooldown period
            if last_trade_time and (timestamp - last_trade_time).total_seconds() < self.config['cooldown_period'] * 60:
                continue

            # Get features for this timestamp
            features = feature_data.loc[timestamp].to_dict()
            
            # Generate signal
            try:
                signal = signal_engine.generate_hybrid_signal(features, position)
            except Exception as e:
                logger.warning(f"Error generating signal at {timestamp}: {e}")
                continue
            
            # Execute trade based on signal
            if signal.action == "BUY" and position == 0:
                # Calculate position size
                position_size = self._calculate_position_size(capital, current_price, signal.confidence)
                
                if position_size > 0:
                    # Execute buy order
                    executed_price = self._apply_slippage(current_price, "BUY")
                    commission_cost = position_size * executed_price * self.config['commission']
                    
                    if position_size * executed_price + commission_cost <= capital:
                        position = position_size
                        position_value = position * executed_price
                        capital -= position_value + commission_cost
                        
                        # Record trade
                        trade = {
                            'timestamp': timestamp,
                            'action': 'BUY',
                            'price': executed_price,
                            'quantity': position,
                            'value': position_value,
                            'commission': commission_cost,
                            'signal_confidence': signal.confidence,
                            'reasoning': '; '.join(signal.reasoning)
                        }
                        self.trades.append(trade)
                        last_trade_time = timestamp
                        daily_trades += 1
                        
                        logger.debug(f"BUY: {position:.2f} shares at ${executed_price:.2f}")
            
            elif signal.action == "SELL" and position > 0:
                # Execute sell order
                executed_price = self._apply_slippage(current_price, "SELL")
                position_value = position * executed_price
                commission_cost = position_value * self.config['commission']
                
                capital += position_value - commission_cost
                
                # Record trade
                trade = {
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': executed_price,
                    'quantity': position,
                    'value': position_value,
                    'commission': commission_cost,
                    'signal_confidence': signal.confidence,
                    'reasoning': '; '.join(signal.reasoning)
                }
                self.trades.append(trade)
                last_trade_time = timestamp
                daily_trades += 1
                
                logger.debug(f"SELL: {position:.2f} shares at ${executed_price:.2f}")
                position = 0.0
                position_value = 0.0
            
            # Check stop loss and take profit
            if position > 0:
                # Calculate current position value
                current_position_value = position * current_price
                position_return = (current_position_value - position_value) / position_value if position_value > 0 else 0
                
                # Stop loss
                if position_return <= -self.config['stop_loss']:
                    executed_price = self._apply_slippage(current_price, "SELL")
                    sell_value = position * executed_price
                    commission_cost = sell_value * self.config['commission']
                    
                    capital += sell_value - commission_cost
                    
                    trade = {
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'price': executed_price,
                        'quantity': position,
                        'value': sell_value,
                        'commission': commission_cost,
                        'signal_confidence': 1.0,
                        'reasoning': f'Stop loss triggered at {position_return:.2%}'
                    }
                    self.trades.append(trade)
                    last_trade_time = timestamp
                    daily_trades += 1
                    
                    logger.debug(f"STOP LOSS: {position:.2f} shares at ${executed_price:.2f}")
                    position = 0.0
                    position_value = 0.0
                
                # Take profit
                elif position_return >= self.config['take_profit']:
                    executed_price = self._apply_slippage(current_price, "SELL")
                    sell_value = position * executed_price
                    commission_cost = sell_value * self.config['commission']
                    
                    capital += sell_value - commission_cost
                    
                    trade = {
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'price': executed_price,
                        'quantity': position,
                        'value': sell_value,
                        'commission': commission_cost,
                        'signal_confidence': 1.0,
                        'reasoning': f'Take profit triggered at {position_return:.2%}'
                    }
                    self.trades.append(trade)
                    last_trade_time = timestamp
                    daily_trades += 1
                    
                    logger.debug(f"TAKE PROFIT: {position:.2f} shares at ${executed_price:.2f}")
                    position = 0.0
                    position_value = 0.0
            
            # Update equity curve
            current_equity = capital + (position * current_price if position > 0 else 0)
            self.equity_curve.append(current_equity)
            
            # Update drawdown curve
            if len(self.equity_curve) > 1:
                peak_equity = max(self.equity_curve)
                current_drawdown = (peak_equity - current_equity) / peak_equity
                self.drawdown_curve.append(current_drawdown)
            else:
                self.drawdown_curve.append(0.0)
        
        # Close any remaining position
        if position > 0:
            final_price = price_data.iloc[-1]['Close']
            executed_price = self._apply_slippage(final_price, "SELL")
            sell_value = position * executed_price
            commission_cost = sell_value * self.config['commission']
            capital += sell_value - commission_cost
            
            trade = {
                'timestamp': common_index[-1],
                'action': 'SELL',
                'price': executed_price,
                'quantity': position,
                'value': sell_value,
                'commission': commission_cost,
                'signal_confidence': 1.0,
                'reasoning': 'End of backtest - closing position'
            }
            self.trades.append(trade)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(capital)
        
        logger.info(f"Backtest completed: {len(self.trades)} trades, {results['total_return']:.2%} return")
        
        return results
    
    def _calculate_position_size(self, capital: float, price: float, confidence: float) -> float:
        """Calculate position size based on Kelly criterion and risk management"""
        
        # Base position size (Kelly-lite)
        base_size = capital * self.config['max_position_size'] * confidence
        
        # Adjust for confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x based on confidence
        adjusted_size = base_size * confidence_multiplier
        
        # Ensure minimum trade size
        if adjusted_size < self.config['min_trade_size']:
            return 0.0
        
        # Convert to shares
        shares = adjusted_size / price
        
        return shares
    
    def _apply_slippage(self, price: float, action: str) -> float:
        """Apply slippage to price"""
        slippage = self.config['slippage']
        
        if action == "BUY":
            return price * (1 + slippage)
        else:  # SELL
            return price * (1 - slippage)
    
    def _calculate_performance_metrics(self, final_capital: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if not self.trades:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'avg_trade_return': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0
            }
        
        # Basic metrics
        initial_capital = self.config['initial_capital']
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Calculate returns for each trade
        trade_returns = []
        for i in range(0, len(self.trades), 2):  # Pair buy/sell trades
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                
                if buy_trade['action'] == 'BUY' and sell_trade['action'] == 'SELL':
                    trade_return = (sell_trade['value'] - buy_trade['value']) / buy_trade['value']
                    trade_returns.append(trade_return)
        
        # Win rate
        winning_trades = len([r for r in trade_returns if r > 0])
        win_rate = winning_trades / len(trade_returns) if trade_returns else 0.0
        
        # Average trade return
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0.0
        
        # Profit factor
        gross_profit = sum([r for r in trade_returns if r > 0])
        gross_loss = abs(sum([r for r in trade_returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Volatility (annualized)
        if len(self.equity_curve) > 1:
            equity_series = pd.Series(self.equity_curve)
            returns = equity_series.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
        else:
            volatility = 0.0
        
        # Sharpe ratio
        if len(self.equity_curve) > 1:
            equity_series = pd.Series(self.equity_curve)
            returns = equity_series.pct_change().dropna()
            excess_return = returns.mean() * 252 - self.config['risk_free_rate']
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio
        if len(self.equity_curve) > 1:
            equity_series = pd.Series(self.equity_curve)
            returns = equity_series.pct_change().dropna()
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (returns.mean() * 252 - self.config['risk_free_rate']) / downside_volatility if downside_volatility > 0 else 0.0
        else:
            sortino_ratio = 0.0
        
        # Max drawdown
        max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0.0
        
        # Calmar ratio
        annualized_return = (1 + total_return) ** (252 / len(self.equity_curve)) - 1 if len(self.equity_curve) > 0 else 0.0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trade_returns),
            'avg_trade_return': avg_trade_return,
            'volatility': volatility,
            'calmar_ratio': calmar_ratio
        }
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get detailed trade summary"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve"""
        return pd.Series(self.equity_curve)
    
    def get_drawdown_curve(self) -> pd.Series:
        """Get drawdown curve"""
        return pd.Series(self.drawdown_curve)


def create_backtest_engine(config: Optional[Dict] = None) -> BacktestEngine:
    """Create a backtest engine with optional configuration"""
    return BacktestEngine(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create backtest engine
    engine = create_backtest_engine()
    
    print("Backtest engine ready for simulation")