"""Tests for configuration module."""

import os
import pytest
from unittest.mock import patch
import config


class TestConfig:
    """Test configuration loading and validation."""

    def test_env_variables_loaded(self):
        """Test that environment variables are properly loaded."""
        # Test that config module can be imported without errors
        assert hasattr(config, 'ALPACA_API_KEY')
        assert hasattr(config, 'ALPACA_SECRET_KEY')
        assert hasattr(config, 'POSITIVE_THRESHOLD')
        assert hasattr(config, 'NEGATIVE_THRESHOLD')

    def test_default_values(self):
        """Test that default values are set correctly."""
        assert isinstance(config.POSITIVE_THRESHOLD, float)
        assert isinstance(config.NEGATIVE_THRESHOLD, float)
        assert 0.0 <= config.POSITIVE_THRESHOLD <= 1.0
        assert 0.0 <= config.NEGATIVE_THRESHOLD <= 1.0

    def test_stock_symbols_default(self):
        """Test default stock symbols."""
        assert isinstance(config.STOCK_SYMBOLS, list)
        assert len(config.STOCK_SYMBOLS) > 0
        assert all(isinstance(symbol, str) for symbol in config.STOCK_SYMBOLS)

    @patch.dict(os.environ, {'STOCK_SYMBOLS': 'AAPL,TSLA,NVDA'})
    def test_stock_symbols_from_env(self):
        """Test stock symbols loaded from environment."""
        # Reload config to pick up new env vars
        import importlib
        importlib.reload(config)
        
        assert config.STOCK_SYMBOLS == ['AAPL', 'TSLA', 'NVDA']

    def test_risk_management_config(self):
        """Test risk management configuration."""
        assert isinstance(config.MAX_POSITION_SIZE, float)
        assert config.MAX_POSITION_SIZE > 0
        assert isinstance(config.MAX_PORTFOLIO_RISK, float)
        assert 0.0 <= config.MAX_PORTFOLIO_RISK <= 1.0
        assert isinstance(config.STOP_LOSS_PERCENTAGE, float)
        assert config.STOP_LOSS_PERCENTAGE > 0

    def test_api_config(self):
        """Test API configuration."""
        assert isinstance(config.MAX_RETRIES, int)
        assert config.MAX_RETRIES > 0
        assert isinstance(config.RETRY_DELAY, float)
        assert config.RETRY_DELAY > 0
        assert isinstance(config.REQUEST_TIMEOUT, float)
        assert config.REQUEST_TIMEOUT > 0
