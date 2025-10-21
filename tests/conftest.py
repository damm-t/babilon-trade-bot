"""Pytest configuration and fixtures."""

import pytest
import os
import tempfile
from unittest.mock import patch


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    env_content = """
ALPACA_API_KEY=test_key
ALPACA_SECRET_KEY=test_secret
POSITIVE_THRESHOLD=0.65
NEGATIVE_THRESHOLD=0.70
STOCK_SYMBOL=AAPL
MAX_POSITION_SIZE=1000.0
MAX_PORTFOLIO_RISK=0.02
LOG_LEVEL=INFO
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(env_content)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    try:
        os.unlink(temp_file)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca client for testing."""
    with patch('broker.alpaca_client.Alpaca') as mock:
        mock_instance = mock.return_value
        mock_instance.get_account.return_value = {
            'cash': '10000.00',
            'portfolio_value': '10000.00',
            'buying_power': '10000.00'
        }
        mock_instance.list_positions.return_value = []
        mock_instance.submit_order.return_value = {'id': 'test_order_id'}
        yield mock_instance


@pytest.fixture
def sample_news_data():
    """Sample news data for testing."""
    return {
        'positive': [
            "Apple reports record quarterly revenue exceeding expectations",
            "Tesla stock surges on strong delivery numbers",
            "Microsoft announces breakthrough in AI technology"
        ],
        'negative': [
            "Company faces major financial crisis and layoffs",
            "Stock plummets after disappointing earnings report",
            "Regulatory investigation threatens business operations"
        ],
        'neutral': [
            "Company releases quarterly financial report",
            "Board of directors meets to discuss strategy",
            "Regular trading session concludes normally"
        ]
    }


@pytest.fixture
def sample_stock_data():
    """Sample stock data for testing."""
    return {
        'AAPL': {
            'symbol': 'AAPL',
            'price': 150.00,
            'change': 2.50,
            'change_percent': 1.69,
            'volume': 50000000
        },
        'TSLA': {
            'symbol': 'TSLA',
            'price': 250.00,
            'change': -5.00,
            'change_percent': -1.96,
            'volume': 30000000
        }
    }


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before each test."""
    env_vars_to_clean = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'POSITIVE_THRESHOLD',
        'NEGATIVE_THRESHOLD',
        'STOCK_SYMBOL',
        'STOCK_SYMBOLS'
    ]
    
    original_values = {}
    for var in env_vars_to_clean:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]
    
    yield
    
    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
