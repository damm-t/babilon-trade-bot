# config.py

import os
import logging
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

# API Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Validate required API keys (allow dummy values for testing)
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    # Set dummy values for testing/development
    ALPACA_API_KEY = ALPACA_API_KEY or "test_key"
    ALPACA_SECRET_KEY = ALPACA_SECRET_KEY or "test_secret"
    
    # Only warn, don't raise error - let the application handle missing keys
    import warnings
    warnings.warn(
        "ALPACA_API_KEY and ALPACA_SECRET_KEY not set. Using dummy values. "
        "Set these environment variables for actual trading.",
        UserWarning
    )

# Trading Configuration
POSITIVE_THRESHOLD = float(os.getenv("POSITIVE_THRESHOLD", 0.65))
NEGATIVE_THRESHOLD = float(os.getenv("NEGATIVE_THRESHOLD", 0.7))
TRADE_THRESHOLD = float(os.getenv("TRADE_THRESHOLD", 0.65))  # For backward compatibility

# Default stock symbol for single-stock operations (used by main.py)
STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "AAPL")

# Default stock symbols for analysis
# Can be overridden via env var STOCK_SYMBOLS as a comma-separated list, e.g. "TSLA,XHR,XLK,RKT"
_default_symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "XHR", "XLK", "RKT"]
_env_symbols_raw = os.getenv("STOCK_SYMBOLS")
if _env_symbols_raw:
    STOCK_SYMBOLS = [s.strip().upper() for s in _env_symbols_raw.split(",") if s.strip()]
else:
    STOCK_SYMBOLS = _default_symbols

# Risk Management Configuration
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", 1000.0))  # Maximum position size in dollars
MAX_PORTFOLIO_RISK = float(os.getenv("MAX_PORTFOLIO_RISK", 0.02))  # Maximum 2% portfolio risk per trade
STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", 0.05))  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = float(os.getenv("TAKE_PROFIT_PERCENTAGE", 0.10))  # 10% take profit

# API Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 1.0))  # seconds
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))  # seconds

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT
)