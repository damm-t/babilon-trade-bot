"""
Enhanced Technical Analysis Features Module
Implements comprehensive TA indicators for Phase 3 of Babilon Trade Bot
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TAFeatures:
    """Enhanced Technical Analysis Features Calculator"""
    
    def __init__(self):
        self.feature_cache = {}
    
    def compute_sma(self, series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        if window <= 0:
            raise ValueError("window must be > 0")
        return series.rolling(window=window, min_periods=window).mean()
    
    def compute_ema(self, series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        if window <= 0:
            raise ValueError("window must be > 0")
        return series.ewm(span=window, adjust=False).mean()
    
    def compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if period <= 0:
            raise ValueError("period must be > 0")
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.bfill().fillna(50.0)
    
    def compute_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.compute_ema(series, fast)
        ema_slow = self.compute_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.compute_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def compute_bollinger_bands(self, series: pd.Series, window: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = self.compute_sma(series, window)
        rolling_std = series.rolling(window=window, min_periods=window).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period, min_periods=period).mean()
        return atr.bfill()
    
    def compute_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                          k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def compute_volume_indicators(self, close: pd.Series, volume: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Volume-based indicators"""
        # Volume SMA
        volume_sma = self.compute_sma(volume, window)
        
        # Volume ratio (current volume / average volume)
        volume_ratio = volume / volume_sma
        
        # On-Balance Volume (OBV)
        price_change = close.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0)).cumsum()
        obv_series = pd.Series(obv, index=close.index)
        
        # Volume Price Trend (VPT)
        vpt = (volume * (close / close.shift() - 1)).cumsum()
        
        return {
            'volume_sma': volume_sma,
            'volume_ratio': volume_ratio,
            'obv': obv_series,
            'vpt': vpt
        }
    
    def compute_momentum_indicators(self, close: pd.Series, window: int = 10) -> Dict[str, pd.Series]:
        """Momentum indicators"""
        # Rate of Change
        roc = (close / close.shift(window) - 1) * 100
        
        # Momentum
        momentum = close - close.shift(window)
        
        # Williams %R
        high = close.rolling(window=window, min_periods=window).max()
        low = close.rolling(window=window, min_periods=window).min()
        williams_r = -100 * ((high - close) / (high - low))
        
        return {
            'roc': roc,
            'momentum': momentum,
            'williams_r': williams_r
        }
    
    def compute_trend_indicators(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """Trend indicators"""
        # ADX (Average Directional Index) - simplified version
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm_series = pd.Series(plus_dm, index=close.index)
        minus_dm_series = pd.Series(minus_dm, index=close.index)
        
        # Price position within recent range
        recent_high = high.rolling(window=20, min_periods=20).max()
        recent_low = low.rolling(window=20, min_periods=20).min()
        price_position = (close - recent_low) / (recent_high - recent_low)
        
        return {
            'plus_dm': plus_dm_series,
            'minus_dm': minus_dm_series,
            'price_position': price_position
        }
    
    def compute_volatility_indicators(self, close: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Volatility indicators"""
        # Historical Volatility
        returns = np.log(close / close.shift(1))
        historical_vol = returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)
        
        # Average True Range (simplified)
        high = close.rolling(window=window, min_periods=window).max()
        low = close.rolling(window=window, min_periods=window).min()
        atr_simple = (high - low) / close
        
        return {
            'historical_volatility': historical_vol,
            'atr_simple': atr_simple
        }
    
    def compute_market_regime_features(self, close: pd.Series) -> Dict[str, pd.Series]:
        """Market regime detection features"""
        # Short vs Long term trend
        sma_20 = self.compute_sma(close, 20)
        sma_50 = self.compute_sma(close, 50)
        sma_200 = self.compute_sma(close, 200)
        
        # Trend strength
        trend_20_50 = (sma_20 / sma_50 - 1) * 100
        trend_50_200 = (sma_50 / sma_200 - 1) * 100
        
        # Market regime classification
        regime = np.where(trend_20_50 > 0, 1, -1)  # 1 for uptrend, -1 for downtrend
        regime_strength = np.abs(trend_20_50)
        
        return {
            'trend_20_50': trend_20_50,
            'trend_50_200': trend_50_200,
            'regime': pd.Series(regime, index=close.index),
            'regime_strength': pd.Series(regime_strength, index=close.index)
        }
    
    def build_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive technical analysis features from OHLCV data
        
        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        
        Returns:
            DataFrame with all TA features
        """
        try:
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            # Initialize result DataFrame
            features_df = pd.DataFrame(index=df.index)
            
            # Basic price features
            features_df['price_change'] = close.pct_change()
            features_df['log_return'] = np.log(close / close.shift(1))
            features_df['high_low_ratio'] = high / low
            features_df['close_open_ratio'] = close / df['Open']
            
            # Moving averages
            features_df['sma_5'] = self.compute_sma(close, 5)
            features_df['sma_10'] = self.compute_sma(close, 10)
            features_df['sma_20'] = self.compute_sma(close, 20)
            features_df['sma_50'] = self.compute_sma(close, 50)
            features_df['ema_12'] = self.compute_ema(close, 12)
            features_df['ema_26'] = self.compute_ema(close, 26)
            
            # Price ratios
            features_df['price_sma5_ratio'] = close / features_df['sma_5'] - 1
            features_df['price_sma20_ratio'] = close / features_df['sma_20'] - 1
            features_df['price_sma50_ratio'] = close / features_df['sma_50'] - 1
            features_df['ema12_ema26_ratio'] = features_df['ema_12'] / features_df['ema_26'] - 1
            
            # RSI
            features_df['rsi_14'] = self.compute_rsi(close, 14)
            features_df['rsi_21'] = self.compute_rsi(close, 21)
            
            # MACD
            macd_data = self.compute_macd(close)
            features_df['macd'] = macd_data['macd']
            features_df['macd_signal'] = macd_data['signal']
            features_df['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self.compute_bollinger_bands(close)
            features_df['bb_upper'] = bb_data['upper']
            features_df['bb_middle'] = bb_data['middle']
            features_df['bb_lower'] = bb_data['lower']
            features_df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
            features_df['bb_position'] = (close - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
            
            # ATR
            features_df['atr_14'] = self.compute_atr(high, low, close, 14)
            features_df['atr_ratio'] = features_df['atr_14'] / close
            
            # Stochastic
            stoch_data = self.compute_stochastic(high, low, close)
            features_df['stoch_k'] = stoch_data['k_percent']
            features_df['stoch_d'] = stoch_data['d_percent']
            
            # Volume indicators
            volume_data = self.compute_volume_indicators(close, volume)
            features_df['volume_sma_20'] = volume_data['volume_sma']
            features_df['volume_ratio'] = volume_data['volume_ratio']
            features_df['obv'] = volume_data['obv']
            features_df['vpt'] = volume_data['vpt']
            
            # Momentum indicators
            momentum_data = self.compute_momentum_indicators(close)
            features_df['roc_10'] = momentum_data['roc']
            features_df['momentum_10'] = momentum_data['momentum']
            features_df['williams_r'] = momentum_data['williams_r']
            
            # Trend indicators
            trend_data = self.compute_trend_indicators(close, high, low)
            features_df['plus_dm'] = trend_data['plus_dm']
            features_df['minus_dm'] = trend_data['minus_dm']
            features_df['price_position'] = trend_data['price_position']
            
            # Volatility indicators
            vol_data = self.compute_volatility_indicators(close)
            features_df['historical_volatility'] = vol_data['historical_volatility']
            features_df['atr_simple'] = vol_data['atr_simple']
            
            # Market regime features
            regime_data = self.compute_market_regime_features(close)
            features_df['trend_20_50'] = regime_data['trend_20_50']
            features_df['trend_50_200'] = regime_data['trend_50_200']
            features_df['regime'] = regime_data['regime']
            features_df['regime_strength'] = regime_data['regime_strength']
            
            # Additional derived features
            features_df['volatility_ratio'] = features_df['historical_volatility'] / features_df['historical_volatility'].rolling(20).mean()
            features_df['volume_price_trend'] = features_df['volume_ratio'] * features_df['price_change']
            
            # Clean up infinite and NaN values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.bfill().ffill()
            
            logger.info(f"Generated {len(features_df.columns)} TA features for {len(features_df)} data points")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error building TA features: {e}")
            raise
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Return feature importance weights for ensemble models"""
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


def create_feature_store(data_dir: str, output_dir: str = "data/features") -> None:
    """
    Create feature store by processing all ticker data and computing TA features
    
    Args:
        data_dir: Directory containing ticker CSV files
        output_dir: Directory to save feature store
    """
    import os
    import glob
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    ta_calculator = TAFeatures()
    
    # Process all CSV files in data directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    for csv_file in csv_files:
        try:
            # Extract ticker name from filename
            ticker = os.path.basename(csv_file).replace('.csv', '')
            
            # Skip non-ticker files
            if ticker in ['logs', 'portfolio']:
                continue
            
            logger.info(f"Processing {ticker}...")
            
            # Load data
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Compute TA features
            features_df = ta_calculator.build_comprehensive_features(df)
            
            # Add ticker column
            features_df['ticker'] = ticker
            
            # Save to parquet format
            output_file = os.path.join(output_dir, f"{ticker}_features.parquet")
            features_df.to_parquet(output_file)
            
            logger.info(f"Saved {len(features_df)} feature rows for {ticker} to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            continue
    
    logger.info(f"Feature store creation completed. Files saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create feature store from existing data
    create_feature_store("data", "data/features")
